from copy import deepcopy
from pathlib import Path
from typing import Any

from helpers.design_output_store import DesignOutputStore
from helpers.motion_output_store import MotionOutputStore

from .project_pipeline_state_store import (
    ProjectPipelineStateStore,
)
from .workflow_store import WorkflowStore


class ProjectPipelineOrchestrator:
    """
    Observes durable project state and advances the distributed pipeline.

    First-phase responsibility:

        vision workflow finished
            -> design_pending

        finished design output found
            -> motion_pending

        finished motion output found
            -> pipeline_finished

    This version does not yet send commands to the Design PC,
    Motion PC, GH Agent, or robot executor.
    """

    TERMINAL_STATUSES = {
        "pipeline_finished",
        "robot_finished",
        "failed",
        "cancelled",
    }

    DESIGN_WAITING_STATUSES = {
        "design_pending",
        "design_requested",
        "design_running",
        "design_finished",
    }

    MOTION_WAITING_STATUSES = {
        "motion_pending",
        "motion_requested",
        "motion_running",
        "motion_finished",
    }

    def __init__(
        self,
        sessions_root,
        workflow_folder_name: str = "Workflows",
    ):
        self.sessions_root = Path(
            sessions_root
        )

        self.workflow_store = WorkflowStore(
            sessions_root=self.sessions_root,
            workflow_folder_name=(
                workflow_folder_name
            ),
        )

        self.state_store = (
            ProjectPipelineStateStore(
                sessions_root=self.sessions_root,
            )
        )

        self.design_store = DesignOutputStore(
            sessions_root=self.sessions_root,
        )

        self.motion_store = MotionOutputStore(
            sessions_root=self.sessions_root,
        )

    # ============================================================
    # PUBLIC API
    # ============================================================

    def evaluate_session(
        self,
        session: str,
        workflow_id: str | None = None,
        processing_output_index: int | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates one session and performs at most one state transition.

        Calling this repeatedly is safe. The expected-status checks in
        ProjectPipelineStateStore prevent stale transitions.

        processing_output_index may be provided when the vision workflow
        does not currently expose that index in its persisted context.
        """

        session_name = self._validate_session(
            session
        )

        session_path = (
            self.sessions_root /
            session_name
        )

        if not session_path.is_dir():
            raise FileNotFoundError(
                "Session does not exist: "
                f"{session_path}"
            )

        state = self.state_store.load_or_create(
            session=session_name,
            processing_output_index=(
                processing_output_index
            ),
            workflow_id=workflow_id,
        )

        if processing_output_index is not None:
            normalized_processing_index = (
                self._validate_index(
                    processing_output_index,
                    "processing_output_index",
                )
            )

            if (
                state.get(
                    "processing_output_index"
                )
                != normalized_processing_index
            ):
                state = self.state_store.update(
                    session=session_name,
                    processing_output_index=(
                        normalized_processing_index
                    ),
                )

        current_status = self._normalize(
            state.get("status")
        )

        observation = {
            "session": session_name,
            "state_before": deepcopy(state),
            "transitioned": False,
            "transition": None,
            "workflow": None,
            "design_output": None,
            "motion_output": None,
        }

        if current_status in self.TERMINAL_STATUSES:
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Project pipeline is already in "
                f"terminal status '{current_status}'."
            )

            return observation

        workflow_record = self._resolve_workflow(
            session=session_name,
            workflow_id=(
                workflow_id
                or state.get("workflow_id")
            ),
        )

        observation["workflow"] = (
            self._workflow_summary(
                workflow_record
            )
        )

        # --------------------------------------------------------
        # 1. Observe the vision/processing workflow
        # --------------------------------------------------------

        if current_status == "waiting_for_processing":
            return self._evaluate_processing_stage(
                session=session_name,
                state=state,
                workflow_record=workflow_record,
                observation=observation,
            )

        # --------------------------------------------------------
        # 2. Observe Design_Output
        # --------------------------------------------------------

        if current_status in self.DESIGN_WAITING_STATUSES:
            return self._evaluate_design_stage(
                session=session_name,
                state=state,
                observation=observation,
            )

        # --------------------------------------------------------
        # 3. Observe Motion_Output
        # --------------------------------------------------------

        if current_status in self.MOTION_WAITING_STATUSES:
            return self._evaluate_motion_stage(
                session=session_name,
                state=state,
                observation=observation,
            )

        observation["state_after"] = deepcopy(
            state
        )

        observation["message"] = (
            "No evaluator is defined for project "
            f"pipeline status '{current_status}'."
        )

        return observation

    def get_status(
        self,
        session: str,
    ) -> dict[str, Any]:
        return self.state_store.load(
            self._validate_session(
                session
            )
        )

    def initialize_session(
        self,
        session: str,
        workflow_id: str | None = None,
        processing_output_index: int | None = None,
        overwrite: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.state_store.create(
            session=self._validate_session(
                session
            ),
            workflow_id=workflow_id,
            processing_output_index=(
                processing_output_index
            ),
            metadata=metadata,
            overwrite=overwrite,
        )

    # ============================================================
    # STAGE EVALUATION
    # ============================================================

    def _evaluate_processing_stage(
        self,
        session: str,
        state: dict[str, Any],
        workflow_record: dict[str, Any] | None,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        processing_index = state.get(
            "processing_output_index"
        )

        # ---------------------------------------------------------
        # Explicit processing-output mode
        # ---------------------------------------------------------

        if processing_index is not None:
            processing_index = self._validate_index(
                processing_index,
                "processing_output_index",
            )

            updated = self.state_store.transition(
                session=session,
                new_status="design_pending",
                expected_status="waiting_for_processing",
                message=(
                    "Processing output is available. "
                    "Design stage is pending."
                ),
                active_stage="design",
                job_id=None,
                processing_output_index=processing_index,
                metadata={
                    "processing_confirmation": (
                        "explicit_output_index"
                    ),
                },
            )

            return self._transition_result(
                observation=observation,
                previous_state=state,
                updated_state=updated,
                transition=(
                    "waiting_for_processing"
                    " -> design_pending"
                ),
            )

        # ---------------------------------------------------------
        # Persisted workflow mode
        # ---------------------------------------------------------

        if workflow_record is None:
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "No persisted vision workflow or "
                "processing output index was found."
            )

            return observation

        workflow_status = self._normalize(
            workflow_record.get("status")
        )

        workflow_id = str(
            workflow_record.get("workflow_id") or ""
        ).strip()

        if workflow_status == "failed":
            return self._transition_failed(
                session=session,
                state=state,
                observation=observation,
                message="Vision workflow failed.",
                error=workflow_record.get("error"),
                job_id=workflow_id or None,
            )

        if workflow_status == "cancelled":
            updated = self.state_store.transition(
                session=session,
                new_status="cancelled",
                expected_status="waiting_for_processing",
                message="Vision workflow was cancelled.",
                active_stage=None,
                job_id=workflow_id or None,
                error=workflow_record.get("error"),
            )

            return self._transition_result(
                observation=observation,
                previous_state=state,
                updated_state=updated,
                transition=(
                    "waiting_for_processing"
                    " -> cancelled"
                ),
            )

        if workflow_status != "finished":
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Waiting for the vision workflow "
                f"to finish. Current status: "
                f"{workflow_status or 'unknown'}."
            )

            return observation

        processing_index = (
            self._extract_processing_output_index(
                workflow_record
            )
        )

        if processing_index is None:
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Vision workflow finished, but no "
                "processing output index was found."
            )

            return observation

        updated = self.state_store.transition(
            session=session,
            new_status="design_pending",
            expected_status="waiting_for_processing",
            message=(
                "Processing finished. "
                "Design stage is pending."
            ),
            active_stage="design",
            job_id=None,
            processing_output_index=processing_index,
            metadata={
                "vision_workflow_id": (
                    workflow_id or None
                ),
                "vision_workflow_status": (
                    workflow_status
                ),
                "processing_confirmation": (
                    "persisted_workflow"
                ),
            },
        )

        if (
            not updated.get("workflow_id")
            and workflow_id
        ):
            updated = self.state_store.update(
                session=session,
                workflow_id=workflow_id,
            )

        return self._transition_result(
            observation=observation,
            previous_state=state,
            updated_state=updated,
            transition=(
                "waiting_for_processing"
                " -> design_pending"
            ),
        )

    def _evaluate_design_stage(
        self,
        session: str,
        state: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        processing_index = state.get(
            "processing_output_index"
        )

        design_output = (
            self._select_design_output(
                session=session,
                source_processing_output_index=(
                    processing_index
                ),
            )
        )

        observation["design_output"] = (
            deepcopy(design_output)
            if design_output is not None
            else None
        )

        if design_output is None:
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Waiting for a matching design "
                "output manifest entry."
            )

            return observation

        design_status = self._normalize(
            design_output.get("status")
        )

        design_index = design_output.get(
            "design_output_index"
        )

        if design_status == "failed":
            return self._transition_failed(
                session=session,
                state=state,
                observation=observation,
                message=(
                    "Design output upload failed."
                ),
                error=design_output.get(
                    "error"
                ),
                job_id=None,
            )

        if design_status != "finished":
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Design output exists but is not "
                f"finished. Current status: "
                f"{design_status or 'unknown'}."
            )

            return observation

        updated = self.state_store.transition(
            session=session,
            new_status="motion_pending",
            expected_status=self._normalize(
                state.get("status")
            ),
            message=(
                "Design output finished. "
                "Motion stage is pending."
            ),
            active_stage="motion",
            job_id=None,
            design_output_index=(
                design_index
            ),
            metadata={
                "design_output_status": (
                    design_status
                ),
                "design_output_created_by": (
                    design_output.get(
                        "created_by"
                    )
                ),
            },
        )

        return self._transition_result(
            observation=observation,
            previous_state=state,
            updated_state=updated,
            transition=(
                f"{state.get('status')}"
                " -> motion_pending"
            ),
        )

    def _evaluate_motion_stage(
        self,
        session: str,
        state: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        design_index = state.get(
            "design_output_index"
        )

        motion_output = (
            self._select_motion_output(
                session=session,
                source_design_output_index=(
                    design_index
                ),
            )
        )

        observation["motion_output"] = (
            deepcopy(motion_output)
            if motion_output is not None
            else None
        )

        if motion_output is None:
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Waiting for a matching motion "
                "output manifest entry."
            )

            return observation

        motion_status = self._normalize(
            motion_output.get("status")
        )

        motion_index = motion_output.get(
            "motion_output_index"
        )

        if motion_status == "failed":
            return self._transition_failed(
                session=session,
                state=state,
                observation=observation,
                message=(
                    "Motion output upload failed."
                ),
                error=motion_output.get(
                    "error"
                ),
                job_id=None,
            )

        if motion_status != "finished":
            observation["state_after"] = deepcopy(
                state
            )

            observation["message"] = (
                "Motion output exists but is not "
                f"finished. Current status: "
                f"{motion_status or 'unknown'}."
            )

            return observation

        updated = self.state_store.transition(
            session=session,
            new_status="pipeline_finished",
            expected_status=self._normalize(
                state.get("status")
            ),
            message=(
                "Motion output finished. "
                "Project pipeline completed."
            ),
            active_stage=None,
            job_id=None,
            motion_output_index=(
                motion_index
            ),
            metadata={
                "motion_output_status": (
                    motion_status
                ),
                "motion_output_created_by": (
                    motion_output.get(
                        "created_by"
                    )
                ),
            },
        )

        return self._transition_result(
            observation=observation,
            previous_state=state,
            updated_state=updated,
            transition=(
                f"{state.get('status')}"
                " -> pipeline_finished"
            ),
        )

    # ============================================================
    # WORKFLOW LOOKUP
    # ============================================================

    def _resolve_workflow(
        self,
        session: str,
        workflow_id: str | None,
    ) -> dict[str, Any] | None:
        normalized_workflow_id = str(
            workflow_id or ""
        ).strip()

        if normalized_workflow_id:
            try:
                return self.workflow_store.load(
                    session=session,
                    workflow_id=(
                        normalized_workflow_id
                    ),
                )
            except FileNotFoundError:
                return None

        records = self.workflow_store.list_workflows(
            session
        )

        if not records:
            return None

        # WorkflowStore returns newest modified records first.
        return dict(
            records[0]
        )

    # ============================================================
    # OUTPUT LOOKUP
    # ============================================================

    def _select_design_output(
        self,
        session: str,
        source_processing_output_index: int | None,
    ) -> dict[str, Any] | None:
        outputs = self.design_store.list_outputs(
            session
        )

        candidates = []

        for output in outputs:
            if not isinstance(
                output,
                dict,
            ):
                continue

            if (
                source_processing_output_index
                is not None
                and output.get(
                    "source_processing_output_index"
                )
                != int(
                    source_processing_output_index
                )
            ):
                continue

            candidates.append(
                output
            )

        return self._select_latest_output(
            candidates,
            index_field="design_output_index",
        )

    def _select_motion_output(
        self,
        session: str,
        source_design_output_index: int | None,
    ) -> dict[str, Any] | None:
        outputs = self.motion_store.list_outputs(
            session
        )

        candidates = []

        for output in outputs:
            if not isinstance(
                output,
                dict,
            ):
                continue

            if (
                source_design_output_index
                is not None
                and output.get(
                    "source_design_output_index"
                )
                != int(
                    source_design_output_index
                )
            ):
                continue

            candidates.append(
                output
            )

        return self._select_latest_output(
            candidates,
            index_field="motion_output_index",
        )

    def _select_latest_output(
        self,
        outputs: list[dict[str, Any]],
        index_field: str,
    ) -> dict[str, Any] | None:
        if not outputs:
            return None

        def sort_key(
            output: dict[str, Any],
        ):
            return (
                str(
                    output.get("updated_at")
                    or output.get("created_at")
                    or ""
                ),
                self._safe_integer(
                    output.get(index_field)
                ),
            )

        selected = max(
            outputs,
            key=sort_key,
        )

        return dict(
            selected
        )

    # ============================================================
    # PROCESSING INDEX DISCOVERY
    # ============================================================

    def _extract_processing_output_index(
        self,
        workflow_record: dict[str, Any],
    ) -> int | None:
        """
        Attempts to discover the processing output index from persisted
        workflow context without depending on one exact executor schema.

        Explicit state/configuration remains preferred.
        """

        preferred_keys = {
            "processing_output_index",
            "processing_index",
        }

        value = self._find_integer_by_keys(
            workflow_record.get(
                "context",
                {}
            ),
            preferred_keys,
        )

        if value is not None:
            return value

        value = self._find_integer_by_keys(
            workflow_record.get(
                "stage_states",
                {}
            ),
            preferred_keys,
        )

        if value is not None:
            return value

        return None

    def _find_integer_by_keys(
        self,
        value: Any,
        keys: set[str],
    ) -> int | None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized_key = str(
                    key
                ).strip().lower()

                if normalized_key in keys:
                    parsed = self._positive_integer_or_none(
                        child
                    )

                    if parsed is not None:
                        return parsed

            for child in value.values():
                found = self._find_integer_by_keys(
                    child,
                    keys,
                )

                if found is not None:
                    return found

        elif isinstance(value, list):
            for child in value:
                found = self._find_integer_by_keys(
                    child,
                    keys,
                )

                if found is not None:
                    return found

        return None

    # ============================================================
    # RESULT HELPERS
    # ============================================================

    def _transition_failed(
        self,
        session: str,
        state: dict[str, Any],
        observation: dict[str, Any],
        message: str,
        error: Any,
        job_id: str | None,
    ) -> dict[str, Any]:
        updated = self.state_store.transition(
            session=session,
            new_status="failed",
            expected_status=self._normalize(
                state.get("status")
            ),
            message=message,
            active_stage=None,
            job_id=job_id,
            error=error,
        )

        return self._transition_result(
            observation=observation,
            previous_state=state,
            updated_state=updated,
            transition=(
                f"{state.get('status')}"
                " -> failed"
            ),
        )

    def _transition_result(
        self,
        observation: dict[str, Any],
        previous_state: dict[str, Any],
        updated_state: dict[str, Any],
        transition: str,
    ) -> dict[str, Any]:
        observation["transitioned"] = True
        observation["transition"] = transition
        observation["state_before"] = deepcopy(
            previous_state
        )
        observation["state_after"] = deepcopy(
            updated_state
        )
        observation["message"] = (
            updated_state.get("message")
            or transition
        )

        return observation

    def _workflow_summary(
        self,
        record: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if record is None:
            return None

        return {
            "workflow_id": record.get(
                "workflow_id"
            ),
            "workflow_name": record.get(
                "workflow_name"
            ),
            "status": record.get(
                "status"
            ),
            "progress": record.get(
                "progress"
            ),
            "current_stage": record.get(
                "current_stage"
            ),
            "message": record.get(
                "message"
            ),
            "error": record.get(
                "error"
            ),
            "created_at": record.get(
                "created_at"
            ),
            "updated_at": record.get(
                "updated_at"
            ),
            "finished_at": record.get(
                "finished_at"
            ),
        }

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_session(
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
                "Session must be a folder name, "
                "not a path."
            )

        return session_name

    def _validate_index(
        self,
        value: Any,
        field_name: str,
    ) -> int:
        parsed = self._positive_integer_or_none(
            value
        )

        if parsed is None:
            raise ValueError(
                f"{field_name} must be an "
                "integer of at least 1."
            )

        return parsed

    def _positive_integer_or_none(
        self,
        value: Any,
    ) -> int | None:
        try:
            parsed = int(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            return None

        if parsed < 1:
            return None

        return parsed

    def _safe_integer(
        self,
        value: Any,
    ) -> int:
        parsed = self._positive_integer_or_none(
            value
        )

        return parsed or 0

    def _normalize(
        self,
        value: Any,
    ) -> str:
        return str(
            value or ""
        ).strip().lower()