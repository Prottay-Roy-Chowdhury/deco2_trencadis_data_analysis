import json
import threading

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProjectPipelineStateStore:
    """
    Persists the distributed project-pipeline state for one session.

    Expected structure:

    sessions/
    └── <session>/
        └── Project_Pipeline/
            └── project_pipeline_state.json

    This store does not execute any workflow stages. It only owns the
    durable coordination state connecting:

        processing
        → design
        → motion
        → robot execution
    """

    FOLDER_NAME = "Project_Pipeline"
    STATE_FILENAME = "project_pipeline_state.json"
    SCHEMA_VERSION = 1

    INITIAL_STATUS = "waiting_for_processing"

    VALID_STATUSES = {
        "waiting_for_processing",
        "design_pending",
        "design_requested",
        "design_running",
        "design_finished",
        "motion_pending",
        "motion_requested",
        "motion_running",
        "motion_finished",
        "robot_pending",
        "robot_requested",
        "robot_running",
        "robot_finished",
        "pipeline_finished",
        "failed",
        "cancelled",
        "paused",
    }

    def __init__(
        self,
        sessions_root,
    ):
        self.sessions_root = Path(
            sessions_root
        )

        self._lock = threading.RLock()

    # ============================================================
    # PATHS
    # ============================================================

    def get_session_path(
        self,
        session: str,
    ) -> Path:
        session_name = (
            self._validate_session_name(
                session
            )
        )

        return (
            self.sessions_root /
            session_name
        )

    def get_pipeline_folder(
        self,
        session: str,
        create: bool = False,
    ) -> Path:
        session_path = (
            self.get_session_path(
                session
            )
        )

        if not session_path.is_dir():
            raise FileNotFoundError(
                "Session does not exist: "
                f"{session_path}"
            )

        pipeline_folder = (
            session_path /
            self.FOLDER_NAME
        )

        if create:
            pipeline_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

        return pipeline_folder

    def get_state_path(
        self,
        session: str,
        create_folder: bool = False,
    ) -> Path:
        return (
            self.get_pipeline_folder(
                session=session,
                create=create_folder,
            ) /
            self.STATE_FILENAME
        )

    # ============================================================
    # STATE LIFECYCLE
    # ============================================================

    def exists(
        self,
        session: str,
    ) -> bool:
        return self.get_state_path(
            session=session,
            create_folder=False,
        ).is_file()

    def create(
        self,
        session: str,
        processing_output_index: int | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """
        Creates the initial project-pipeline state.

        The Project_Pipeline folder is created only when this method
        is called.
        """

        state_path = self.get_state_path(
            session=session,
            create_folder=True,
        )

        if (
            state_path.exists()
            and not overwrite
        ):
            raise FileExistsError(
                "Project pipeline state already "
                f"exists: {state_path}"
            )

        timestamp = self._utc_now()

        processing_index = (
            self._optional_index(
                processing_output_index,
                "processing_output_index",
            )
        )

        state = {
            "schema_version": (
                self.SCHEMA_VERSION
            ),
            "session": (
                self._validate_session_name(
                    session
                )
            ),
            "status": (
                self.INITIAL_STATUS
            ),
            "previous_status": None,
            "created_at": timestamp,
            "updated_at": timestamp,
            "started_at": None,
            "finished_at": None,
            "workflow_id": (
                str(
                    workflow_id or ""
                ).strip() or None
            ),
            "processing_output_index": (
                processing_index
            ),
            "design_output_index": None,
            "motion_output_index": None,
            "robot_execution_index": None,
            "active_stage": "processing",
            "active_job_id": None,
            "message": (
                "Waiting for processing output."
            ),
            "error": None,
            "retry_count": 0,
            "metadata": dict(
                metadata or {}
            ),
            "history": [
                self._history_entry(
                    previous_status=None,
                    status=(
                        self.INITIAL_STATUS
                    ),
                    message=(
                        "Project pipeline state created."
                    ),
                    job_id=None,
                    error=None,
                )
            ],
        }

        self.save(
            session=session,
            state=state,
        )

        return dict(state)

    def load(
        self,
        session: str,
    ) -> dict[str, Any]:
        state_path = self.get_state_path(
            session=session,
            create_folder=False,
        )

        if not state_path.is_file():
            raise FileNotFoundError(
                "Project pipeline state not "
                f"found: {state_path}"
            )

        with self._lock:
            with state_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                state = json.load(
                    file
                )

        if not isinstance(
            state,
            dict,
        ):
            raise ValueError(
                "Invalid project pipeline "
                f"state: {state_path}"
            )

        self._apply_defaults(
            session=session,
            state=state,
        )

        return state

    def load_or_create(
        self,
        session: str,
        processing_output_index: int | None = None,
        workflow_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self.exists(session):
            return self.load(
                session
            )

        return self.create(
            session=session,
            processing_output_index=(
                processing_output_index
            ),
            workflow_id=workflow_id,
            metadata=metadata,
        )

    def save(
        self,
        session: str,
        state: dict[str, Any],
    ) -> Path:
        if not isinstance(
            state,
            dict,
        ):
            raise TypeError(
                "state must be a dictionary."
            )

        session_name = (
            self._validate_session_name(
                session
            )
        )

        state_path = self.get_state_path(
            session=session_name,
            create_folder=True,
        )

        temporary_path = (
            state_path.with_suffix(
                state_path.suffix +
                ".tmp"
            )
        )

        state["schema_version"] = (
            self.SCHEMA_VERSION
        )

        state["session"] = (
            session_name
        )

        state["updated_at"] = (
            self._utc_now()
        )

        state.setdefault(
            "history",
            [],
        )

        state.setdefault(
            "metadata",
            {},
        )

        with self._lock:
            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    state,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            temporary_path.replace(
                state_path
            )

        return state_path

    # ============================================================
    # STATE UPDATES
    # ============================================================

    def update(
        self,
        session: str,
        **changes,
    ) -> dict[str, Any]:
        """
        Updates fields without enforcing a status transition.

        Use transition() when changing the pipeline status.
        """

        state = self.load(
            session
        )

        protected_fields = {
            "session",
            "schema_version",
            "created_at",
            "history",
        }

        for key, value in changes.items():
            if key in protected_fields:
                raise ValueError(
                    f"{key} cannot be updated directly."
                )

            if key == "status":
                raise ValueError(
                    "Use transition() to change status."
                )

            if key.endswith("_output_index"):
                value = self._optional_index(
                    value,
                    key,
                )

            if key == "robot_execution_index":
                value = self._optional_index(
                    value,
                    key,
                )

            state[key] = value

        self.save(
            session=session,
            state=state,
        )

        return dict(state)

    def transition(
        self,
        session: str,
        new_status: str,
        expected_status: str | None = None,
        message: str = "",
        active_stage: str | None = None,
        job_id: str | None = None,
        error: Any = None,
        processing_output_index: int | None = None,
        design_output_index: int | None = None,
        motion_output_index: int | None = None,
        robot_execution_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Atomically transitions the project pipeline.

        expected_status prevents duplicate or stale transitions.
        """

        normalized_status = (
            self._validate_status(
                new_status
            )
        )

        normalized_expected = None

        if expected_status is not None:
            normalized_expected = (
                self._validate_status(
                    expected_status
                )
            )

        with self._lock:
            state = self.load(
                session
            )

            current_status = str(
                state.get("status") or ""
            ).strip().lower()

            if (
                normalized_expected is not None
                and current_status
                != normalized_expected
            ):
                raise RuntimeError(
                    "Project pipeline transition "
                    "rejected: expected status "
                    f"'{normalized_expected}', "
                    f"found '{current_status}'."
                )

            timestamp = self._utc_now()

            state["previous_status"] = (
                current_status or None
            )

            state["status"] = (
                normalized_status
            )

            if active_stage is not None:
                state["active_stage"] = (
                    str(
                        active_stage or ""
                    ).strip() or None
                )

            state["active_job_id"] = (
                str(
                    job_id or ""
                ).strip() or None
            )

            if message:
                state["message"] = str(
                    message
                )

            state["error"] = error

            if processing_output_index is not None:
                state[
                    "processing_output_index"
                ] = self._validate_index(
                    processing_output_index,
                    "processing_output_index",
                )

            if design_output_index is not None:
                state[
                    "design_output_index"
                ] = self._validate_index(
                    design_output_index,
                    "design_output_index",
                )

            if motion_output_index is not None:
                state[
                    "motion_output_index"
                ] = self._validate_index(
                    motion_output_index,
                    "motion_output_index",
                )

            if robot_execution_index is not None:
                state[
                    "robot_execution_index"
                ] = self._validate_index(
                    robot_execution_index,
                    "robot_execution_index",
                )

            if metadata:
                current_metadata = (
                    state.setdefault(
                        "metadata",
                        {},
                    )
                )

                current_metadata.update(
                    dict(metadata)
                )

            if (
                state.get("started_at") is None
                and normalized_status
                != self.INITIAL_STATUS
            ):
                state["started_at"] = (
                    timestamp
                )

            if normalized_status in {
                "pipeline_finished",
                "robot_finished",
                "cancelled",
            }:
                state["finished_at"] = (
                    timestamp
                )

            state.setdefault(
                "history",
                [],
            ).append(
                self._history_entry(
                    previous_status=(
                        current_status or None
                    ),
                    status=normalized_status,
                    message=message,
                    job_id=job_id,
                    error=error,
                    created_at=timestamp,
                )
            )

            self.save(
                session=session,
                state=state,
            )

        return dict(state)

    def mark_failed(
        self,
        session: str,
        error: Any,
        message: str = (
            "Project pipeline failed."
        ),
        job_id: str | None = None,
    ) -> dict[str, Any]:
        return self.transition(
            session=session,
            new_status="failed",
            message=message,
            active_stage=None,
            job_id=job_id,
            error=error,
        )

    def increment_retry(
        self,
        session: str,
        message: str = (
            "Project pipeline retry requested."
        ),
    ) -> dict[str, Any]:
        with self._lock:
            state = self.load(
                session
            )

            state["retry_count"] = (
                int(
                    state.get(
                        "retry_count",
                        0,
                    )
                ) +
                1
            )

            state.setdefault(
                "history",
                [],
            ).append(
                self._history_entry(
                    previous_status=(
                        state.get(
                            "status"
                        )
                    ),
                    status=(
                        state.get(
                            "status"
                        )
                    ),
                    message=message,
                    job_id=(
                        state.get(
                            "active_job_id"
                        )
                    ),
                    error=None,
                )
            )

            self.save(
                session=session,
                state=state,
            )

        return dict(state)

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _apply_defaults(
        self,
        session: str,
        state: dict[str, Any],
    ) -> None:
        state.setdefault(
            "schema_version",
            self.SCHEMA_VERSION,
        )

        state.setdefault(
            "session",
            self._validate_session_name(
                session
            ),
        )

        state.setdefault(
            "status",
            self.INITIAL_STATUS,
        )

        state.setdefault(
            "previous_status",
            None,
        )

        state.setdefault(
            "created_at",
            self._utc_now(),
        )

        state.setdefault(
            "updated_at",
            state["created_at"],
        )

        state.setdefault(
            "started_at",
            None,
        )

        state.setdefault(
            "finished_at",
            None,
        )

        state.setdefault(
            "workflow_id",
            None,
        )

        state.setdefault(
            "processing_output_index",
            None,
        )

        state.setdefault(
            "design_output_index",
            None,
        )

        state.setdefault(
            "motion_output_index",
            None,
        )

        state.setdefault(
            "robot_execution_index",
            None,
        )

        state.setdefault(
            "active_stage",
            "processing",
        )

        state.setdefault(
            "active_job_id",
            None,
        )

        state.setdefault(
            "message",
            "",
        )

        state.setdefault(
            "error",
            None,
        )

        state.setdefault(
            "retry_count",
            0,
        )

        state.setdefault(
            "metadata",
            {},
        )

        state.setdefault(
            "history",
            [],
        )

    def _history_entry(
        self,
        previous_status: str | None,
        status: str | None,
        message: str,
        job_id: str | None,
        error: Any,
        created_at: str | None = None,
    ) -> dict[str, Any]:
        return {
            "created_at": (
                created_at or
                self._utc_now()
            ),
            "previous_status": (
                previous_status
            ),
            "status": status,
            "message": str(
                message or ""
            ),
            "job_id": (
                str(
                    job_id or ""
                ).strip() or None
            ),
            "error": error,
        }

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

        if (
            Path(session_name).name
            != session_name
        ):
            raise ValueError(
                "Session must be a folder "
                "name, not a path."
            )

        return session_name

    def _validate_status(
        self,
        value: str,
    ) -> str:
        normalized = str(
            value or ""
        ).strip().lower()

        if normalized not in self.VALID_STATUSES:
            valid_values = ", ".join(
                sorted(
                    self.VALID_STATUSES
                )
            )

            raise ValueError(
                "Invalid project pipeline status: "
                f"'{normalized}'. Valid values: "
                f"{valid_values}."
            )

        return normalized

    def _validate_index(
        self,
        value: int,
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

    def _optional_index(
        self,
        value: int | None,
        field_name: str,
    ) -> int | None:
        if value is None:
            return None

        return self._validate_index(
            value,
            field_name,
        )

    def _utc_now(
        self,
    ) -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()