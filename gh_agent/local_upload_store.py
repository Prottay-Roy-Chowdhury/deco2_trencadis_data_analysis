import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from gh_agent.config import DATA_UPLOAD_ROOT


class LocalUploadStore:
    """
    Manages files prepared locally for upload to the Python master.

    Structure:

        DATA_UPLOAD/
        └── <session>/
            ├── session_upload_manifest.json
            ├── design_geometry_01.ghdata
            ├── design_preview_01.png
            └── ...

    Session folders are created only when explicitly requested.
    """

    MANIFEST_FILENAME = "session_upload_manifest.json"

    def __init__(
        self,
        upload_root: Path | str = DATA_UPLOAD_ROOT,
    ):
        self.upload_root = Path(upload_root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------

    def get_session_folder(
        self,
        session: str,
        create: bool = False,
    ) -> Path:
        session_name = self._validate_session(session)

        session_folder = (
            self.upload_root / session_name
        )

        if create:
            session_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        return session_folder

    def get_manifest_path(
        self,
        session: str,
        create_session_folder: bool = False,
    ) -> Path:
        return (
            self.get_session_folder(
                session=session,
                create=create_session_folder,
            )
            / self.MANIFEST_FILENAME
        )

    # ------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------

    def create_empty_manifest(
        self,
        session: str,
    ) -> dict[str, Any]:
        now = self._now()

        return {
            "session": self._validate_session(session),
            "created_at": now,
            "updated_at": now,
            "files": [],
        }

    def load_manifest(
        self,
        session: str,
    ) -> dict[str, Any]:
        manifest_path = self.get_manifest_path(
            session=session,
            create_session_folder=False,
        )

        if not manifest_path.exists():
            return self.create_empty_manifest(session)

        with self._lock:
            try:
                with manifest_path.open(
                    "r",
                    encoding="utf-8",
                ) as file:
                    manifest = json.load(file)
            except Exception:
                return self.create_empty_manifest(session)

        if not isinstance(manifest, dict):
            return self.create_empty_manifest(session)

        manifest["session"] = self._validate_session(
            session
        )

        if not isinstance(manifest.get("files"), list):
            manifest["files"] = []

        return manifest

    def save_manifest(
        self,
        session: str,
        manifest: dict[str, Any],
    ) -> Path:
        manifest_path = self.get_manifest_path(
            session=session,
            create_session_folder=True,
        )

        temporary_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )

        manifest["session"] = self._validate_session(
            session
        )

        manifest.setdefault(
            "created_at",
            self._now(),
        )

        manifest["updated_at"] = self._now()

        with self._lock:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    manifest,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(manifest_path)

        return manifest_path

    # ------------------------------------------------------------
    # File registration
    # ------------------------------------------------------------

    def register_file(
        self,
        session: str,
        local_path: str | Path,
        domain: str,
        category: str,
        design_output_index: int | None = None,
        motion_output_index: int | None = None,
        upload_status: str = "pending",
    ) -> dict[str, Any]:
        """
        Registers an existing local file in the session manifest.

        This does not copy or move the file. The file is expected to
        already live inside DATA_UPLOAD/<session>/.
        """
        session_folder = self.get_session_folder(
            session=session,
            create=True,
        )

        path = Path(local_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Local upload file does not exist: {path}"
            )

        try:
            path.relative_to(session_folder.resolve())
        except ValueError as exc:
            raise ValueError(
                "Registered files must be located inside "
                f"the upload session folder: {session_folder}"
            ) from exc

        normalized_domain = self._sanitize_name(
            domain,
            fallback="unknown",
        )

        normalized_category = self._sanitize_name(
            category,
            fallback="file",
        )

        record = {
            "domain": normalized_domain,
            "category": normalized_category,
            "filename": path.name,
            "local_path": str(path),
            "size_bytes": path.stat().st_size,
            "upload_status": str(
                upload_status or "pending"
            ),
            "registered_at": self._now(),
            "updated_at": self._now(),
            "uploaded_at": None,
            "master_filename": None,
            "error": None,
        }

        if design_output_index is not None:
            record["design_output_index"] = (
                self._validate_index(
                    design_output_index,
                    "design_output_index",
                )
            )

        if motion_output_index is not None:
            record["motion_output_index"] = (
                self._validate_index(
                    motion_output_index,
                    "motion_output_index",
                )
            )

        manifest = self.load_manifest(session)

        self._upsert_file_record(
            manifest["files"],
            record,
        )

        self.save_manifest(
            session=session,
            manifest=manifest,
        )

        return dict(record)

    def update_upload_status(
        self,
        session: str,
        filename: str,
        upload_status: str,
        master_filename: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        manifest = self.load_manifest(session)

        record = self._find_file_record(
            manifest["files"],
            filename,
        )

        if record is None:
            raise FileNotFoundError(
                f"Upload manifest entry not found: {filename}"
            )

        record["upload_status"] = str(upload_status)
        record["updated_at"] = self._now()
        record["error"] = error

        if master_filename is not None:
            record["master_filename"] = str(
                master_filename
            )

        if upload_status == "uploaded":
            record["uploaded_at"] = self._now()

        self.save_manifest(
            session=session,
            manifest=manifest,
        )

        return dict(record)

    def update_files_status(
        self,
        session: str,
        filenames: list[str],
        upload_status: str,
        master_files: dict[str, str] | None = None,
        error: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Updates several local upload-manifest entries atomically.

        master_files maps the original local filename to the filename
        assigned by the Python master.
        """
        if not isinstance(filenames, list) or not filenames:
            raise ValueError(
                "filenames must be a non-empty list."
            )

        requested_names = {
            str(filename).strip().lower()
            for filename in filenames
            if str(filename).strip()
        }

        if not requested_names:
            raise ValueError(
                "filenames must contain at least one valid name."
            )

        master_files = master_files or {}
        normalized_master_files = {
            str(key).strip().lower(): str(value)
            for key, value in master_files.items()
        }

        manifest = self.load_manifest(session)
        updated_records: list[dict[str, Any]] = []
        now = self._now()

        for record in manifest.get("files", []):
            filename = str(
                record.get("filename", "")
            ).strip()

            if filename.lower() not in requested_names:
                continue

            record["upload_status"] = str(upload_status)
            record["updated_at"] = now
            record["error"] = error

            master_filename = normalized_master_files.get(
                filename.lower()
            )

            if master_filename:
                record["master_filename"] = master_filename

            if upload_status == "uploaded":
                record["uploaded_at"] = now

            updated_records.append(dict(record))

        found_names = {
            str(record.get("filename", "")).lower()
            for record in updated_records
        }

        missing_names = requested_names - found_names

        if missing_names:
            raise FileNotFoundError(
                "Upload manifest entries were not found: "
                + ", ".join(sorted(missing_names))
            )

        self.save_manifest(
            session=session,
            manifest=manifest,
        )

        return updated_records

    def list_files(
        self,
        session: str,
        domain: str | None = None,
        design_output_index: int | None = None,
        upload_status: str | None = None,
    ) -> list[dict[str, Any]]:
        manifest = self.load_manifest(session)

        normalized_domain = None

        if domain:
            normalized_domain = self._sanitize_name(
                domain,
                fallback="",
            )

        requested_design_index = None

        if design_output_index is not None:
            requested_design_index = (
                self._validate_index(
                    design_output_index,
                    "design_output_index",
                )
            )

        results: list[dict[str, Any]] = []

        for record in manifest.get("files", []):
            if (
                normalized_domain is not None
                and record.get("domain")
                != normalized_domain
            ):
                continue

            if (
                requested_design_index is not None
                and record.get("design_output_index")
                != requested_design_index
            ):
                continue

            if (
                upload_status is not None
                and record.get("upload_status")
                != upload_status
            ):
                continue

            item = dict(record)

            path = Path(
                item.get("local_path", "")
            )

            item["exists"] = path.is_file()

            if item["exists"]:
                item["size_bytes"] = (
                    path.stat().st_size
                )

            results.append(item)

        return results

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _upsert_file_record(
        self,
        records: list[dict[str, Any]],
        new_record: dict[str, Any],
    ) -> None:
        new_path = str(
            new_record.get("local_path", "")
        ).lower()

        for index, record in enumerate(records):
            existing_path = str(
                record.get("local_path", "")
            ).lower()

            if existing_path == new_path:
                records[index] = new_record
                return

        records.append(new_record)

    def _find_file_record(
        self,
        records: list[dict[str, Any]],
        filename: str,
    ) -> dict[str, Any] | None:
        requested_name = str(filename).lower()

        for record in records:
            stored_name = str(
                record.get("filename", "")
            ).lower()

            if stored_name == requested_name:
                return record

        return None

    def _validate_session(
        self,
        session: str,
    ) -> str:
        session_name = str(session or "").strip()

        if not session_name:
            raise ValueError(
                "Session name cannot be empty."
            )

        if Path(session_name).name != session_name:
            raise ValueError(
                "Session must be a folder name, not a path."
            )

        return session_name

    def _validate_index(
        self,
        value: int,
        field_name: str,
    ) -> int:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be an integer."
            ) from exc

        if index < 1:
            raise ValueError(
                f"{field_name} must be at least 1."
            )

        return index

    def _sanitize_name(
        self,
        value: str,
        fallback: str,
    ) -> str:
        raw = str(value or "").strip().lower()

        cleaned = "".join(
            character
            if character.isalnum()
            else "_"
            for character in raw
        )

        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")

        cleaned = cleaned.strip("_")

        return cleaned or fallback

    def _now(self) -> str:
        return datetime.now().isoformat(
            timespec="seconds"
        )