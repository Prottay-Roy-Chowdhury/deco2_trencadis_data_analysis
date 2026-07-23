import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MotionOutputStore:
    """
    Manages motion outputs stored on the Python master.

    Expected structure:

    sessions/
    └── <session>/
        └── Motion_Output/
            ├── motion_output_manifest.json
            ├── motion_ghdata_01.ghdata
            ├── motion_program_01.mod
            └── ...

    Motion_Output is created only when a motion upload begins.
    """

    OUTPUT_FOLDER_NAME = "Motion_Output"
    MANIFEST_FILENAME = "motion_output_manifest.json"

    def __init__(self, sessions_root):
        self.sessions_root = Path(sessions_root)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def get_session_path(
        self,
        session: str,
    ) -> Path:
        session_name = self._validate_session_name(session)

        return self.sessions_root / session_name

    def get_output_folder(
        self,
        session: str,
        create: bool = False,
    ) -> Path:
        session_path = self.get_session_path(session)

        if not session_path.is_dir():
            raise FileNotFoundError(
                f"Session does not exist: {session_path}"
            )

        output_folder = (
            session_path /
            self.OUTPUT_FOLDER_NAME
        )

        if create:
            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        return output_folder

    def get_manifest_path(
        self,
        session: str,
        create_folder: bool = False,
    ) -> Path:
        return (
            self.get_output_folder(
                session=session,
                create=create_folder,
            ) /
            self.MANIFEST_FILENAME
        )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def load_manifest(
        self,
        session: str,
    ) -> dict[str, Any]:
        manifest_path = self.get_manifest_path(
            session=session,
            create_folder=False,
        )

        if not manifest_path.exists():
            return self._new_manifest(session)

        with self._lock:
            with manifest_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                manifest = json.load(file)

        if not isinstance(manifest, dict):
            raise ValueError(
                f"Invalid motion manifest: {manifest_path}"
            )

        manifest.setdefault(
            "session",
            str(session).strip(),
        )

        manifest.setdefault(
            "output_type",
            "motion",
        )

        manifest.setdefault(
            "outputs",
            [],
        )

        return manifest

    def save_manifest(
        self,
        session: str,
        manifest: dict[str, Any],
    ) -> Path:
        manifest_path = self.get_manifest_path(
            session=session,
            create_folder=True,
        )

        temporary_path = manifest_path.with_suffix(
            manifest_path.suffix + ".tmp"
        )

        manifest["session"] = str(session).strip()
        manifest["output_type"] = "motion"
        manifest["updated_at"] = self._utc_now()

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

    # ------------------------------------------------------------------
    # Upload lifecycle
    # ------------------------------------------------------------------

    def begin_upload(
        self,
        session: str,
        motion_output_index: int,
        source_design_output_index: int | None = None,
        message: str = "",
        created_by: str = "motion_pc",
    ) -> dict[str, Any]:
        """
        Creates Motion_Output if needed and records an uploading entry.

        No files are written by this method. The communication layer
        later streams files into the returned output folder.
        """

        motion_index = self._validate_index(
            motion_output_index,
            "motion_output_index",
        )

        source_index = None

        if source_design_output_index is not None:
            source_index = self._validate_index(
                source_design_output_index,
                "source_design_output_index",
            )

        output_folder = self.get_output_folder(
            session=session,
            create=True,
        )

        manifest = self.load_manifest(session)

        if self._find_output(
            manifest,
            motion_index,
        ) is not None:
            raise FileExistsError(
                "A motion output entry already exists for "
                f"motion_output_index={motion_index}."
            )

        timestamp = self._utc_now()

        entry = {
            "motion_output_index": motion_index,
            "source_design_output_index": source_index,
            "status": "uploading",
            "created_by": (
                str(created_by).strip() or
                "motion_pc"
            ),
            "created_at": timestamp,
            "updated_at": timestamp,
            "message": str(message or ""),
            "files": [],
            "error": None,
        }

        manifest["outputs"].append(entry)

        self.save_manifest(
            session,
            manifest,
        )

        return {
            "session": str(session).strip(),
            "motion_output_index": motion_index,
            "output_folder": str(output_folder),
            "manifest_path": str(
                self.get_manifest_path(session)
            ),
            "entry": dict(entry),
        }

    def complete_upload(
        self,
        session: str,
        motion_output_index: int,
        files: list[dict[str, Any]],
        message: str = "",
    ) -> dict[str, Any]:
        motion_index = self._validate_index(
            motion_output_index,
            "motion_output_index",
        )

        manifest = self.load_manifest(session)

        entry = self._require_output(
            manifest,
            motion_index,
        )

        entry["status"] = "finished"
        entry["updated_at"] = self._utc_now()

        entry["files"] = [
            dict(file_record)
            for file_record in files
        ]

        entry["error"] = None

        if message:
            entry["message"] = str(message)

        self.save_manifest(
            session,
            manifest,
        )

        return dict(entry)

    def fail_upload(
        self,
        session: str,
        motion_output_index: int,
        error: str,
        files: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        motion_index = self._validate_index(
            motion_output_index,
            "motion_output_index",
        )

        manifest = self.load_manifest(session)

        entry = self._require_output(
            manifest,
            motion_index,
        )

        entry["status"] = "failed"
        entry["updated_at"] = self._utc_now()
        entry["error"] = str(error)
        entry["message"] = (
            "Motion output upload failed."
        )

        if files is not None:
            entry["files"] = [
                dict(file_record)
                for file_record in files
            ]

        self.save_manifest(
            session,
            manifest,
        )

        return dict(entry)

    def get_output(
        self,
        session: str,
        motion_output_index: int,
    ) -> dict[str, Any]:
        motion_index = self._validate_index(
            motion_output_index,
            "motion_output_index",
        )

        manifest = self.load_manifest(session)

        entry = self._require_output(
            manifest,
            motion_index,
        )

        return dict(entry)

    def list_outputs(
        self,
        session: str,
    ) -> list[dict[str, Any]]:
        manifest = self.load_manifest(session)

        return [
            dict(entry)
            for entry in manifest.get(
                "outputs",
                [],
            )
        ]

    # ------------------------------------------------------------------
    # Filename generation
    # ------------------------------------------------------------------

    def build_stored_filename(
        self,
        category: str,
        motion_output_index: int,
        original_filename: str,
        category_sequence: int | None = None,
    ) -> str:
        """
        Examples:

        solution.ghdata + ghdata + 1
            -> motion_ghdata_01.ghdata

        robot.mod + program + 1
            -> motion_program_01.mod

        second_program.mod + program + 1 + sequence 2
            -> motion_program_01_02.mod
        """

        motion_index = self._validate_index(
            motion_output_index,
            "motion_output_index",
        )

        safe_category = self._sanitize_name(
            category,
            fallback="file",
        )

        suffix = Path(
            original_filename
        ).suffix.lower()

        if not suffix:
            suffix = ".bin"

        sequence_text = ""

        if category_sequence is not None:
            sequence = self._validate_index(
                category_sequence,
                "category_sequence",
            )

            sequence_text = f"_{sequence:02d}"

        return (
            f"motion_{safe_category}_"
            f"{motion_index:02d}"
            f"{sequence_text}"
            f"{suffix}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_manifest(
        self,
        session: str,
    ) -> dict[str, Any]:
        timestamp = self._utc_now()

        return {
            "session": str(session).strip(),
            "output_type": "motion",
            "created_at": timestamp,
            "updated_at": timestamp,
            "outputs": [],
        }

    def _find_output(
        self,
        manifest: dict[str, Any],
        motion_output_index: int,
    ) -> dict[str, Any] | None:
        for entry in manifest.get(
            "outputs",
            [],
        ):
            if (
                entry.get("motion_output_index") ==
                motion_output_index
            ):
                return entry

        return None

    def _require_output(
        self,
        manifest: dict[str, Any],
        motion_output_index: int,
    ) -> dict[str, Any]:
        entry = self._find_output(
            manifest,
            motion_output_index,
        )

        if entry is None:
            raise FileNotFoundError(
                "Motion output entry not found for "
                f"motion_output_index={motion_output_index}."
            )

        return entry

    def _validate_session_name(
        self,
        session: str,
    ) -> str:
        session_name = str(
            session or ""
        ).strip()

        if not session_name:
            raise ValueError(
                "Session cannot be empty."
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
        raw = str(
            value or ""
        ).strip().lower()

        cleaned = "".join(
            character
            if character.isalnum()
            else "_"
            for character in raw
        )

        while "__" in cleaned:
            cleaned = cleaned.replace(
                "__",
                "_",
            )

        cleaned = cleaned.strip("_")

        return cleaned or fallback

    def _utc_now(self) -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()