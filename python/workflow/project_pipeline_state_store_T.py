import json
import threading
import uuid

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

    VALID_ACTION_STATUSES = {
        "pending",
        "claimed",
        "running",
        "completed",
        "failed",
        "cancelled",
    }

    ACTIVE_ACTION_STATUSES = {
        "pending",
        "claimed",
        "running",
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
            "requested_solution_count": 1,
            "current_solution_index": 1,
            "completed_solution_count": 0,
            "solution_history": [],
            "design_output_index": None,
            "motion_output_index": None,
            "robot_execution_index": None,
            "active_stage": "processing",
            "active_job_id": None,
            "pending_action": None,
            "action_history": [],
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


    def configure_solution_count(
        self,
        session: str,
        requested_solution_count: int,
    ) -> dict[str, Any]:
        """Sets the total number of sequential design-motion solutions."""
        count = self._validate_index(
            requested_solution_count,
            "requested_solution_count",
        )

        with self._lock:
            state = self.load(session)
            completed = int(
                state.get("completed_solution_count") or 0
            )
            if count < completed:
                raise ValueError(
                    "requested_solution_count cannot be lower than "
                    "completed_solution_count."
                )
            state["requested_solution_count"] = count
            self.save(session=session, state=state)

        return dict(state)

    def complete_solution_iteration(
        self,
        session: str,
        design_output_index: int,
        motion_output_index: int,
        expected_status: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Records one completed solution and either starts the next or finishes."""
        design_index = self._validate_index(
            design_output_index, "design_output_index"
        )
        motion_index = self._validate_index(
            motion_output_index, "motion_output_index"
        )
        expected = self._validate_status(expected_status)

        with self._lock:
            state = self.load(session)
            current_status = str(state.get("status") or "").strip().lower()
            if current_status != expected:
                raise RuntimeError(
                    "Solution completion rejected: expected pipeline status "
                    f"'{expected}', found '{current_status}'."
                )

            solution_index = self._validate_index(
                state.get("current_solution_index") or 1,
                "current_solution_index",
            )
            history = state.setdefault("solution_history", [])
            if not any(
                isinstance(item, dict)
                and item.get("solution_index") == solution_index
                for item in history
            ):
                history.append({
                    "solution_index": solution_index,
                    "processing_output_index": state.get("processing_output_index"),
                    "design_output_index": design_index,
                    "motion_output_index": motion_index,
                    "status": "completed",
                    "completed_at": self._utc_now(),
                    "metadata": dict(metadata or {}),
                })

            completed = max(
                int(state.get("completed_solution_count") or 0),
                solution_index,
            )
            requested = self._validate_index(
                state.get("requested_solution_count") or 1,
                "requested_solution_count",
            )
            state["completed_solution_count"] = completed
            state["motion_output_index"] = motion_index

            if completed < requested:
                state["previous_status"] = current_status
                state["status"] = "design_pending"
                state["active_stage"] = "design"
                state["active_job_id"] = None
                state["current_solution_index"] = solution_index + 1
                state["design_output_index"] = None
                state["motion_output_index"] = None
                state["pending_action"] = None
                state["message"] = (
                    f"Solution {solution_index} completed. "
                    f"Solution {solution_index + 1} design is pending."
                )
                state["finished_at"] = None
            else:
                state["previous_status"] = current_status
                state["status"] = "pipeline_finished"
                state["active_stage"] = None
                state["active_job_id"] = None
                state["message"] = (
                    f"All {requested} requested solution(s) completed."
                )
                state["finished_at"] = self._utc_now()

            state.setdefault("history", []).append(
                self._history_entry(
                    previous_status=current_status,
                    status=state["status"],
                    message=state["message"],
                    job_id=None,
                    error=None,
                )
            )
            self.save(session=session, state=state)

        return dict(state)

    # ============================================================
    # PROJECT ACTIONS
    # ============================================================

    def create_pending_action(
        self,
        session: str,
        action: str,
        target: str,
        source_output_index: int,
        expected_pipeline_status: str,
        metadata: dict[str, Any] | None = None,
        solution_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Creates one durable action for a remote project stage.

        The method is idempotent. If the same active action already
        exists, that action is returned instead of creating a duplicate.
        """

        normalized_action = self._normalize_required_text(
            action,
            "action",
        )

        normalized_target = self._normalize_required_text(
            target,
            "target",
        )

        normalized_expected_status = (
            self._validate_status(
                expected_pipeline_status
            )
        )

        normalized_source_index = (
            self._validate_index(
                source_output_index,
                "source_output_index",
            )
        )

        normalized_solution_index = (
            self._validate_index(
                solution_index,
                "solution_index",
            )
            if solution_index is not None
            else None
        )

        with self._lock:
            state = self.load(
                session
            )

            current_pipeline_status = str(
                state.get("status") or ""
            ).strip().lower()

            if (
                current_pipeline_status
                != normalized_expected_status
            ):
                raise RuntimeError(
                    "Project action creation rejected: "
                    "expected pipeline status "
                    f"'{normalized_expected_status}', "
                    f"found '{current_pipeline_status}'."
                )

            existing_action = state.get(
                "pending_action"
            )

            if isinstance(
                existing_action,
                dict,
            ):
                existing_status = str(
                    existing_action.get(
                        "status"
                    ) or ""
                ).strip().lower()

                same_action = (
                    existing_action.get("action")
                    == normalized_action
                    and existing_action.get("target")
                    == normalized_target
                    and existing_action.get(
                        "source_output_index"
                    )
                    == normalized_source_index
                    and existing_action.get(
                        "solution_index"
                    )
                    == normalized_solution_index
                )

                if (
                    existing_status
                    in self.ACTIVE_ACTION_STATUSES
                ):
                    if same_action:
                        return dict(
                            existing_action
                        )

                    raise RuntimeError(
                        "Another active project action "
                        "already exists: "
                        f"{existing_action.get('action_id')}."
                    )

            timestamp = self._utc_now()

            action_id = (
                f"{normalized_action}-"
                f"{self._validate_session_name(session)}-"
                f"s{normalized_solution_index or 1}-"
                f"{normalized_source_index}-"
                f"{uuid.uuid4().hex[:8]}"
            )

            action_record = {
                "action_id": action_id,
                "action": normalized_action,
                "target": normalized_target,
                "source_output_index": (
                    normalized_source_index
                ),
                "solution_index": (
                    normalized_solution_index
                ),
                "status": "pending",
                "created_at": timestamp,
                "updated_at": timestamp,
                "claimed_at": None,
                "started_at": None,
                "finished_at": None,
                "claimed_by": None,
                "message": (
                    "Project action is pending."
                ),
                "result": None,
                "error": None,
                "metadata": dict(
                    metadata or {}
                ),
            }

            state["pending_action"] = (
                action_record
            )

            state.setdefault(
                "action_history",
                [],
            ).append(
                self._action_history_entry(
                    action_record=action_record,
                    event="created",
                    message=(
                        "Project action created."
                    ),
                )
            )

            self.save(
                session=session,
                state=state,
            )

        return dict(
            action_record
        )

    def get_pending_action(
        self,
        session: str,
        target: str | None = None,
    ) -> dict[str, Any] | None:
        state = self.load(
            session
        )

        action_record = state.get(
            "pending_action"
        )

        if not isinstance(
            action_record,
            dict,
        ):
            return None

        action_status = str(
            action_record.get("status") or ""
        ).strip().lower()

        if (
            action_status
            not in self.ACTIVE_ACTION_STATUSES
        ):
            return None

        if target is not None:
            normalized_target = (
                self._normalize_required_text(
                    target,
                    "target",
                )
            )

            if (
                action_record.get("target")
                != normalized_target
            ):
                return None

        return dict(
            action_record
        )

    def claim_pending_action(
        self,
        session: str,
        action_id: str,
        claimed_by: str,
    ) -> dict[str, Any]:
        normalized_action_id = (
            self._normalize_required_identifier(
                action_id,
                "action_id",
            )
        )

        normalized_claimed_by = (
            self._normalize_required_text(
                claimed_by,
                "claimed_by",
            )
        )

        with self._lock:
            state = self.load(
                session
            )

            action_record = state.get(
                "pending_action"
            )

            if not isinstance(
                action_record,
                dict,
            ):
                raise FileNotFoundError(
                    "No pending project action exists."
                )

            if (
                action_record.get("action_id")
                != normalized_action_id
            ):
                raise RuntimeError(
                    "Project action ID does not match "
                    "the current pending action."
                )

            current_status = str(
                action_record.get("status") or ""
            ).strip().lower()

            existing_claimed_by = str(
                action_record.get(
                    "claimed_by"
                ) or ""
            ).strip()

            # Idempotent retry from the same agent.
            if (
                current_status == "claimed"
                and existing_claimed_by
                == normalized_claimed_by
            ):
                return dict(
                    action_record
                )

            if current_status != "pending":
                raise RuntimeError(
                    "Project action cannot be claimed "
                    f"from status '{current_status}'."
                )

            timestamp = self._utc_now()

            action_record["status"] = (
                "claimed"
            )

            action_record["claimed_by"] = (
                normalized_claimed_by
            )

            action_record["claimed_at"] = (
                timestamp
            )

            action_record["updated_at"] = (
                timestamp
            )

            action_record["message"] = (
                "Project action claimed."
            )

            state.setdefault(
                "action_history",
                [],
            ).append(
                self._action_history_entry(
                    action_record=action_record,
                    event="claimed",
                    message=(
                        "Project action claimed by "
                        f"{normalized_claimed_by}."
                    ),
                )
            )

            self.save(
                session=session,
                state=state,
            )

        return dict(
            action_record
        )

    def report_pending_action(
        self,
        session: str,
        action_id: str,
        status: str,
        message: str = "",
        result: Any = None,
        error: Any = None,
    ) -> dict[str, Any]:
        normalized_action_id = (
            self._normalize_required_identifier(
                action_id,
                "action_id",
            )
        )

        normalized_status = str(
            status or ""
        ).strip().lower()

        allowed_report_statuses = {
            "running",
            "completed",
            "failed",
            "cancelled",
        }

        if (
            normalized_status
            not in allowed_report_statuses
        ):
            raise ValueError(
                "Action report status must be one of: "
                + ", ".join(
                    sorted(
                        allowed_report_statuses
                    )
                )
                + "."
            )

        with self._lock:
            state = self.load(
                session
            )

            action_record = state.get(
                "pending_action"
            )

            if not isinstance(
                action_record,
                dict,
            ):
                raise FileNotFoundError(
                    "No current project action exists."
                )

            if (
                action_record.get("action_id")
                != normalized_action_id
            ):
                raise RuntimeError(
                    "Project action ID does not match "
                    "the current project action."
                )

            current_status = str(
                action_record.get("status") or ""
            ).strip().lower()

            valid_previous_statuses = {
                "running": {
                    "claimed",
                    "running",
                },
                "completed": {
                    "pending",
                    "claimed",
                    "running",
                    "completed",
                },
                "failed": {
                    "pending",
                    "claimed",
                    "running",
                    "failed",
                },
                "cancelled": {
                    "pending",
                    "claimed",
                    "running",
                    "cancelled",
                },
            }

            if (
                current_status
                not in valid_previous_statuses[
                    normalized_status
                ]
            ):
                raise RuntimeError(
                    "Project action cannot transition "
                    f"from '{current_status}' to "
                    f"'{normalized_status}'."
                )

            # Idempotent terminal report.
            if (
                current_status
                == normalized_status
                and normalized_status
                in {
                    "completed",
                    "failed",
                    "cancelled",
                }
            ):
                return dict(
                    action_record
                )

            timestamp = self._utc_now()

            action_record["status"] = (
                normalized_status
            )

            action_record["updated_at"] = (
                timestamp
            )

            if normalized_status == "running":
                if (
                    action_record.get(
                        "started_at"
                    )
                    is None
                ):
                    action_record[
                        "started_at"
                    ] = timestamp

            if normalized_status in {
                "completed",
                "failed",
                "cancelled",
            }:
                action_record["finished_at"] = (
                    timestamp
                )

            if message:
                action_record["message"] = str(
                    message
                )
            else:
                action_record["message"] = (
                    "Project action "
                    f"{normalized_status}."
                )

            action_record["result"] = (
                result
            )

            action_record["error"] = (
                error
            )

            state.setdefault(
                "action_history",
                [],
            ).append(
                self._action_history_entry(
                    action_record=action_record,
                    event=normalized_status,
                    message=(
                        action_record["message"]
                    ),
                )
            )

            self.save(
                session=session,
                state=state,
            )

        return dict(
            action_record
        )

    def clear_pending_action(
        self,
        session: str,
        expected_action_id: str | None = None,
    ) -> dict[str, Any] | None:
        """
        Removes the current action after the pipeline has consumed its
        completed/failed/cancelled result.
        """

        with self._lock:
            state = self.load(
                session
            )

            action_record = state.get(
                "pending_action"
            )

            if not isinstance(
                action_record,
                dict,
            ):
                return None

            if expected_action_id is not None:
                normalized_action_id = (
                    self._normalize_required_identifier(
                        expected_action_id,
                        "expected_action_id",
                    )
                )

                if (
                    action_record.get("action_id")
                    != normalized_action_id
                ):
                    raise RuntimeError(
                        "Current project action does not "
                        "match expected_action_id."
                    )

            current_status = str(
                action_record.get("status") or ""
            ).strip().lower()

            if current_status in self.ACTIVE_ACTION_STATUSES:
                raise RuntimeError(
                    "An active project action cannot "
                    "be cleared."
                )

            state.setdefault(
                "action_history",
                [],
            ).append(
                self._action_history_entry(
                    action_record=action_record,
                    event="cleared",
                    message=(
                        "Project action cleared."
                    ),
                )
            )

            state["pending_action"] = None

            self.save(
                session=session,
                state=state,
            )

        return dict(
            action_record
        )

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
            "requested_solution_count",
            1,
        )

        state.setdefault(
            "current_solution_index",
            1,
        )

        state.setdefault(
            "completed_solution_count",
            0,
        )

        state.setdefault(
            "solution_history",
            [],
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
            "pending_action",
            None,
        )

        state.setdefault(
            "action_history",
            [],
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

    def _action_history_entry(
        self,
        action_record: dict[str, Any],
        event: str,
        message: str,
    ) -> dict[str, Any]:
        return {
            "created_at": self._utc_now(),
            "event": str(
                event or ""
            ).strip().lower(),
            "action_id": action_record.get(
                "action_id"
            ),
            "action": action_record.get(
                "action"
            ),
            "target": action_record.get(
                "target"
            ),
            "source_output_index": (
                action_record.get(
                    "source_output_index"
                )
            ),
            "status": action_record.get(
                "status"
            ),
            "claimed_by": action_record.get(
                "claimed_by"
            ),
            "message": str(
                message or ""
            ),
            "error": action_record.get(
                "error"
            ),
        }

    def _normalize_required_text(
        self,
        value: Any,
        field_name: str,
    ) -> str:
        normalized = str(
            value or ""
        ).strip().lower()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized

    def _normalize_required_identifier(
        self,
        value: Any,
        field_name: str,
    ) -> str:
        normalized = str(
            value or ""
        ).strip()

        if not normalized:
            raise ValueError(
                f"{field_name} cannot be empty."
            )

        return normalized