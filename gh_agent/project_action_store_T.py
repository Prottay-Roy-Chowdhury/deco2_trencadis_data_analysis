import json
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProjectActionStore:
    """
    Persists project actions claimed by this GH Agent.

    Expected structure:

    gh_agent_data/
    └── project_actions/
        └── current_action.json

    This store does not communicate with the Python master.
    It only owns the local durable copy of the currently claimed action.
    """

    ACTION_FOLDER_NAME = "project_actions"
    ACTION_FILENAME = "current_action.json"
    SCHEMA_VERSION = 1

    TERMINAL_STATUSES = {
        "completed",
        "failed",
        "cancelled",
        "cleared",
    }

    def __init__(
        self,
        root,
    ):
        self.root = Path(
            root
        ).resolve()

        self._lock = threading.RLock()

    # ============================================================
    # PATHS
    # ============================================================

    def get_action_folder(
        self,
        create: bool = False,
    ) -> Path:
        folder = (
            self.root /
            self.ACTION_FOLDER_NAME
        )

        if create:
            folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        return folder

    def get_action_path(
        self,
        create_folder: bool = False,
    ) -> Path:
        return (
            self.get_action_folder(
                create=create_folder,
            )
            /
            self.ACTION_FILENAME
        )

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def exists(
        self,
    ) -> bool:
        return self.get_action_path(
            create_folder=False,
        ).is_file()

    def save_claimed_action(
        self,
        session: str,
        action: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(
            action,
            dict,
        ):
            raise TypeError(
                "action must be a dictionary."
            )

        session_name = self._required_text(
            session,
            "session",
            lowercase=False,
        )

        action_id = self._required_text(
            action.get("action_id"),
            "action_id",
            lowercase=False,
        )

        action_name = self._required_text(
            action.get("action"),
            "action",
        )

        target = self._required_text(
            action.get("target"),
            "target",
        )

        source_output_index = self._validate_index(
            action.get(
                "source_output_index"
            ),
            "source_output_index",
        )

        solution_index = self._validate_index(
            action.get("solution_index") or 1,
            "solution_index",
        )

        with self._lock:
            existing = self.load(
                required=False,
            )

            if isinstance(
                existing,
                dict,
            ):
                existing_status = str(
                    existing.get("local_status")
                    or ""
                ).strip().lower()

                existing_action_id = str(
                    existing.get("action_id")
                    or ""
                ).strip()

                if (
                    existing_action_id == action_id
                    and existing_status
                    not in self.TERMINAL_STATUSES
                ):
                    return existing

                if (
                    existing_status
                    not in self.TERMINAL_STATUSES
                ):
                    raise RuntimeError(
                        "Another active local project action "
                        "already exists: "
                        f"{existing_action_id}."
                    )

            timestamp = self._utc_now()

            record = {
                "schema_version": (
                    self.SCHEMA_VERSION
                ),
                "session": session_name,
                "action_id": action_id,
                "action": action_name,
                "target": target,
                "source_output_index": (
                    source_output_index
                ),
                "solution_index": solution_index,
                "master_status": str(
                    action.get("status")
                    or "claimed"
                ).strip().lower(),
                "local_status": "claimed",
                "claimed_by": str(
                    action.get("claimed_by")
                    or target
                ).strip().lower(),
                "created_at": str(
                    action.get("created_at")
                    or timestamp
                ),
                "claimed_at": str(
                    action.get("claimed_at")
                    or timestamp
                ),
                "updated_at": timestamp,
                "consumed_at": None,
                "reported_running_at": None,
                "finished_at": None,
                "message": (
                    "Project action claimed and "
                    "stored locally."
                ),
                "result": None,
                "error": None,
                "metadata": dict(
                    action.get("metadata")
                    or {}
                ),
                "history": [
                    self._history_entry(
                        event="stored",
                        status="claimed",
                        message=(
                            "Claimed master action "
                            "stored locally."
                        ),
                    )
                ],
            }

            self._save_record(
                record
            )

        return dict(
            record
        )

    def load(
        self,
        required: bool = True,
    ) -> dict[str, Any] | None:
        path = self.get_action_path(
            create_folder=False,
        )

        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    "No local project action exists."
                )

            return None

        with self._lock:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                record = json.load(
                    file
                )

        if not isinstance(
            record,
            dict,
        ):
            raise ValueError(
                "Invalid local project action file."
            )

        return record

    def get_active_action(
        self,
        target: str | None = None,
    ) -> dict[str, Any] | None:
        record = self.load(
            required=False,
        )

        if not isinstance(
            record,
            dict,
        ):
            return None

        local_status = str(
            record.get("local_status")
            or ""
        ).strip().lower()

        if local_status in self.TERMINAL_STATUSES:
            return None

        if target is not None:
            normalized_target = self._required_text(
                target,
                "target",
            )

            if (
                record.get("target")
                != normalized_target
            ):
                return None

        return dict(
            record
        )

    # ============================================================
    # LOCAL ACTION STATE
    # ============================================================

    def mark_consumed(
        self,
        action_id: str,
        message: str = (
            "Project action consumed by Grasshopper."
        ),
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require_matching_action(
                action_id
            )

            current_status = str(
                record.get("local_status")
                or ""
            ).strip().lower()

            if current_status == "consumed":
                return dict(
                    record
                )

            if current_status != "claimed":
                raise RuntimeError(
                    "Local action cannot be consumed "
                    f"from status '{current_status}'."
                )

            timestamp = self._utc_now()

            record["local_status"] = (
                "consumed"
            )

            record["consumed_at"] = (
                timestamp
            )

            record["updated_at"] = (
                timestamp
            )

            record["message"] = str(
                message
            )

            record.setdefault(
                "history",
                [],
            ).append(
                self._history_entry(
                    event="consumed",
                    status="consumed",
                    message=message,
                )
            )

            self._save_record(
                record
            )

        return dict(
            record
        )

    def mark_running_reported(
        self,
        action_id: str,
        message: str = (
            "Running status reported to master."
        ),
    ) -> dict[str, Any]:
        with self._lock:
            record = self._require_matching_action(
                action_id
            )

            current_status = str(
                record.get("local_status")
                or ""
            ).strip().lower()

            if current_status == "running":
                return dict(
                    record
                )

            if current_status not in {
                "claimed",
                "consumed",
            }:
                raise RuntimeError(
                    "Local action cannot become running "
                    f"from status '{current_status}'."
                )

            timestamp = self._utc_now()

            record["local_status"] = (
                "running"
            )

            record["master_status"] = (
                "running"
            )

            record["reported_running_at"] = (
                timestamp
            )

            record["updated_at"] = (
                timestamp
            )

            record["message"] = str(
                message
            )

            record.setdefault(
                "history",
                [],
            ).append(
                self._history_entry(
                    event="running_reported",
                    status="running",
                    message=message,
                )
            )

            self._save_record(
                record
            )

        return dict(
            record
        )

    def mark_terminal(
        self,
        action_id: str,
        status: str,
        message: str = "",
        result: Any = None,
        error: Any = None,
    ) -> dict[str, Any]:
        normalized_status = str(
            status or ""
        ).strip().lower()

        if normalized_status not in {
            "completed",
            "failed",
            "cancelled",
        }:
            raise ValueError(
                "Terminal status must be completed, "
                "failed, or cancelled."
            )

        with self._lock:
            record = self._require_matching_action(
                action_id
            )

            current_status = str(
                record.get("local_status")
                or ""
            ).strip().lower()

            if current_status == normalized_status:
                return dict(
                    record
                )

            timestamp = self._utc_now()

            record["local_status"] = (
                normalized_status
            )

            record["master_status"] = (
                normalized_status
            )

            record["finished_at"] = (
                timestamp
            )

            record["updated_at"] = (
                timestamp
            )

            record["message"] = (
                str(message)
                if message
                else (
                    "Project action "
                    f"{normalized_status}."
                )
            )

            record["result"] = result
            record["error"] = error

            record.setdefault(
                "history",
                [],
            ).append(
                self._history_entry(
                    event=normalized_status,
                    status=normalized_status,
                    message=record["message"],
                    error=error,
                )
            )

            self._save_record(
                record
            )

        return dict(
            record
        )

    def clear(
        self,
        expected_action_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self.load(
                required=False,
            )

            if not isinstance(
                record,
                dict,
            ):
                return None

            if expected_action_id is not None:
                normalized_action_id = (
                    self._required_text(
                        expected_action_id,
                        "expected_action_id",
                        lowercase=False,
                    )
                )

                if (
                    record.get("action_id")
                    != normalized_action_id
                ):
                    raise RuntimeError(
                        "Local project action does not "
                        "match expected_action_id."
                    )

            current_status = str(
                record.get("local_status")
                or ""
            ).strip().lower()

            if current_status not in (
                self.TERMINAL_STATUSES
            ):
                raise RuntimeError(
                    "An active local project action "
                    "cannot be cleared."
                )

            path = self.get_action_path(
                create_folder=False,
            )

            if path.exists():
                path.unlink()

        return dict(
            record
        )

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _save_record(
        self,
        record: dict[str, Any],
    ) -> Path:
        path = self.get_action_path(
            create_folder=True,
        )

        temporary_path = (
            path.with_suffix(
                path.suffix + ".tmp"
            )
        )

        record["schema_version"] = (
            self.SCHEMA_VERSION
        )

        record["updated_at"] = (
            self._utc_now()
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                record,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(
            path
        )

        return path

    def _require_matching_action(
        self,
        action_id: str,
    ) -> dict[str, Any]:
        normalized_action_id = (
            self._required_text(
                action_id,
                "action_id",
                lowercase=False,
            )
        )

        record = self.load(
            required=True,
        )

        if (
            record.get("action_id")
            != normalized_action_id
        ):
            raise RuntimeError(
                "Local project action ID does not match."
            )

        return record

    def _history_entry(
        self,
        event: str,
        status: str,
        message: str,
        error: Any = None,
    ) -> dict[str, Any]:
        return {
            "created_at": self._utc_now(),
            "event": str(
                event or ""
            ).strip().lower(),
            "status": str(
                status or ""
            ).strip().lower(),
            "message": str(
                message or ""
            ),
            "error": error,
        }

    def _required_text(
        self,
        value: Any,
        field_name: str,
        lowercase: bool = True,
    ) -> str:
        normalized = str(
            value or ""
        ).strip()

        if lowercase:
            normalized = (
                normalized.lower()
            )

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized

    def _validate_index(
        self,
        value: Any,
        field_name: str,
    ) -> int:
        try:
            index = int(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                f"{field_name} must be an integer."
            ) from exc

        if index < 1:
            raise ValueError(
                f"{field_name} must be at least 1."
            )

        return index

    def _utc_now(
        self,
    ) -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()