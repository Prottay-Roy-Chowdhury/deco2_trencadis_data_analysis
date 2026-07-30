import os
import threading
from typing import Dict, Any
from pathlib import Path
from helpers.session_manager import load_session
from helpers.design_output_store import DesignOutputStore

from workflow.bootstrap import (
    build_default_workflow_manager,
)

from workflow.project_pipeline_orchestrator_T import (
    ProjectPipelineOrchestrator,
)

from agent.protocol import ok_response, error_response
from agent.job_manager import JobManager
# from agent.file_sender import FileSender


class CommandHandler:
    def __init__(self):
        # Manual job system.
        self.jobs = JobManager()

        # Workflow system.
        python_root = Path(__file__).resolve().parents[1]
        project_root = python_root.parent

        sessions_root = project_root / "sessions"
        definitions_root = python_root / "workflow_definitions"

        self.design_output_store = (
            DesignOutputStore(
                sessions_root=sessions_root,
            )
        )

        self.workflows = build_default_workflow_manager(
            sessions_root=sessions_root,
            definitions_root=definitions_root,
        )

        self.project_pipeline = (
            ProjectPipelineOrchestrator(
                sessions_root=sessions_root,
            )
        )

    def handle(self, message: Dict[str, Any]) -> Dict[str, Any]:
        command = message.get("command")

        if not command:
            return error_response("Missing command.")

        if command == "ping":
            return self.handle_ping()

        if command == "get_status":
            return self.handle_get_status(message)

        if command == "list_jobs":
            return self.handle_list_jobs()

        if command == "get_file_metadata":
            return self.handle_get_file_metadata(message)

        if command == "capture":
            return self.handle_capture(message)

        if command == "transform":
            return self.handle_transform(message)

        if command == "process":
            return self.handle_process(message)
        
        if command == "list_downloadable_outputs":
            return self.handle_list_downloadable_outputs(message)

        if command == "get_design_output_file":
            return self.handle_get_design_output_file(
                message
            )
        
        # Workflow commands.
        if command == "submit_workflow":
            return self.handle_submit_workflow(message)

        if command == "get_workflow_status":
            return self.handle_get_workflow_status(message)

        if command == "cancel_workflow":
            return self.handle_cancel_workflow(message)

        if command == "list_workflows":
            return self.handle_list_workflows(message)

        if command == "initialize_project_pipeline":
            return (
                self.handle_initialize_project_pipeline(
                    message
                )
            )

        if command == "evaluate_project_pipeline":
            return (
                self.handle_evaluate_project_pipeline(
                    message
                )
            )

        if command == "get_project_pipeline_status":
            return (
                self.handle_get_project_pipeline_status(
                    message
                )
            )

        if command == "get_pending_project_action":
            return self.handle_get_pending_project_action(
                message
            )

        if command == "claim_project_action":
            return self.handle_claim_project_action(
                message
            )

        if command == "report_project_action":
            return self.handle_report_project_action(
                message
            )

        return error_response(f"Unknown command: {command}")

    def handle_ping(self):
        return ok_response(message="Python Agent is alive.")

    def handle_get_status(self, message):
        job_id = message.get("job_id")

        if not job_id:
            return error_response("Missing job_id.")

        try:
            job = self.jobs.get_job(job_id)
            return ok_response(job=job)
        except KeyError as e:
            return error_response(str(e))

    def handle_list_jobs(self):
        return ok_response(jobs=self.jobs.list_jobs())
    
    # for Workflow manager
    # def handle_submit_workflow(self, message):
    #     workflow_name = message.get("workflow_name")
    #     session = message.get("session")

    #     if not workflow_name:
    #         return error_response("Missing workflow_name.")

    #     if not session:
    #         return error_response("Missing session.")

    #     stage_configs = message.get("stage_configs", {})
    #     selected_stages = message.get("selected_stages")
    #     start_stage = message.get("start_stage")
    #     runtime = message.get("runtime", {})

    #     if not isinstance(stage_configs, dict):
    #         return error_response("stage_configs must be an object.")

    #     if not isinstance(runtime, dict):
    #         return error_response("runtime must be an object.")

    #     try:
    #         workflow = self.workflows.submit_workflow(
    #             workflow_name=workflow_name,
    #             session=session,
    #             stage_configs=stage_configs,
    #             selected_stages=selected_stages,
    #             start_stage=start_stage,
    #             runtime=runtime,
    #         )

    #         # ---------------------------------------------------------
    #         # Initialize the project pipeline immediately after the
    #         # workflow record and workflow manifest have been created.
    #         #
    #         # This creates:
    #         #
    #         #     <session>/Project_Pipeline/
    #         #         project_pipeline_state.json
    #         #
    #         # The initial pipeline status remains:
    #         #
    #         #     waiting_for_processing
    #         #
    #         # The ProjectPipelineMonitor will then detect and evaluate
    #         # this session automatically.
    #         # ---------------------------------------------------------

    #         normalized_session = str(
    #             workflow.get("session")
    #             or session
    #         ).strip()

    #         workflow_id = str(
    #             workflow.get("workflow_id")
    #             or ""
    #         ).strip()

    #         if not self.project_pipeline.state_store.exists(
    #             normalized_session
    #         ):
    #             self.project_pipeline.initialize_session(
    #                 session=normalized_session,
    #                 workflow_id=workflow_id,
    #                 processing_output_index=None,
    #                 overwrite=False,
    #                 metadata={
    #                     "workflow_name": workflow.get(
    #                         "workflow_name"
    #                     ),
    #                     "workflow_version": workflow.get(
    #                         "workflow_version"
    #                     ),
    #                     "initialized_by": (
    #                         "workflow_submission"
    #                     ),
    #                 },
    #             )

    #         return ok_response(
    #             workflow_id=workflow["workflow_id"],
    #             message=(
    #                 "Workflow submitted and project "
    #                 "pipeline initialized."
    #             ),
    #             workflow=workflow,
    #         )

    #     except Exception as e:
    #         return error_response(
    #             message=f"Could not submit workflow: {e}"
    #         )

    def handle_submit_workflow(self, message):
        workflow_name = message.get("workflow_name")
        session = message.get("session")

        if not workflow_name:
            return error_response("Missing workflow_name.")

        if not session:
            return error_response("Missing session.")

        stage_configs = message.get("stage_configs", {})
        selected_stages = message.get("selected_stages")
        start_stage = message.get("start_stage")
        runtime = message.get("runtime", {})

        if not isinstance(stage_configs, dict):
            return error_response(
                "stage_configs must be an object."
            )

        if not isinstance(runtime, dict):
            return error_response(
                "runtime must be an object."
            )

        requested_solution_count = runtime.get(
            "requested_solution_count",
            1,
        )

        try:
            requested_solution_count = int(
                requested_solution_count
            )
        except (TypeError, ValueError):
            return error_response(
                "runtime.requested_solution_count "
                "must be an integer."
            )

        if requested_solution_count < 1:
            return error_response(
                "runtime.requested_solution_count "
                "must be at least 1."
            )

        try:
            workflow = self.workflows.submit_workflow(
                workflow_name=workflow_name,
                session=session,
                stage_configs=stage_configs,
                selected_stages=selected_stages,
                start_stage=start_stage,
                runtime=runtime,
            )

            normalized_session = str(
                workflow.get("session")
                or session
            ).strip()

            workflow_id = str(
                workflow.get("workflow_id")
                or ""
            ).strip()

            if not self.project_pipeline.state_store.exists(
                normalized_session
            ):
                self.project_pipeline.initialize_session(
                    session=normalized_session,
                    workflow_id=workflow_id,
                    processing_output_index=None,
                    overwrite=False,
                    metadata={
                        "workflow_name": workflow.get(
                            "workflow_name"
                        ),
                        "workflow_version": workflow.get(
                            "workflow_version"
                        ),
                        "initialized_by": (
                            "workflow_submission"
                        ),
                    },
                )

                self.project_pipeline.state_store.configure_solution_count(
                    session=normalized_session,
                    requested_solution_count=(
                        requested_solution_count
                    ),
                )

            return ok_response(
                workflow_id=workflow["workflow_id"],
                requested_solution_count=(
                    requested_solution_count
                ),
                message=(
                    "Workflow submitted and project "
                    "pipeline initialized."
                ),
                workflow=workflow,
            )

        except Exception as e:
            return error_response(
                message=f"Could not submit workflow: {e}"
            )


    def handle_get_workflow_status(self, message):
        workflow_id = message.get("workflow_id")
        session = message.get("session")

        if not workflow_id:
            return error_response("Missing workflow_id.")

        try:
            workflow = self.workflows.get_workflow_status(
                workflow_id=workflow_id,
                session=session,
            )

            return ok_response(
                workflow_id=workflow_id,
                workflow=workflow,
            )

        except Exception as e:
            return error_response(
                message=f"Could not get workflow status: {e}"
            )


    def handle_cancel_workflow(self, message):
        workflow_id = message.get("workflow_id")
        session = message.get("session")

        if not workflow_id:
            return error_response("Missing workflow_id.")

        try:
            workflow = self.workflows.cancel_workflow(
                workflow_id=workflow_id,
                session=session,
            )

            return ok_response(
                workflow_id=workflow_id,
                message="Workflow cancellation requested.",
                workflow=workflow,
            )

        except Exception as e:
            return error_response(
                message=f"Could not cancel workflow: {e}"
            )


    def handle_list_workflows(self, message):
        session = message.get("session")

        if not session:
            return error_response("Missing session.")

        try:
            workflows = self.workflows.list_workflows(
                session=session,
            )

            return ok_response(
                session=session,
                workflows=workflows,
            )

        except Exception as e:
            return error_response(
                message=f"Could not list workflows: {e}"
            )

    def handle_initialize_project_pipeline(
        self,
        message,
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        if not session:
            return error_response(
                "Missing session."
            )

        workflow_id = str(
            message.get("workflow_id") or ""
        ).strip() or None

        processing_output_index = (
            message.get(
                "processing_output_index"
            )
        )

        overwrite = bool(
            message.get(
                "overwrite",
                False,
            )
        )

        metadata = message.get(
            "metadata",
            {},
        )

        if not isinstance(
            metadata,
            dict,
        ):
            return error_response(
                "metadata must be an object."
            )

        if processing_output_index is not None:
            try:
                processing_output_index = int(
                    processing_output_index
                )
            except (
                TypeError,
                ValueError,
            ):
                return error_response(
                    "processing_output_index "
                    "must be an integer."
                )

            if processing_output_index < 1:
                return error_response(
                    "processing_output_index "
                    "must be at least 1."
                )

        try:
            state = (
                self.project_pipeline
                .initialize_session(
                    session=session,
                    workflow_id=workflow_id,
                    processing_output_index=(
                        processing_output_index
                    ),
                    overwrite=overwrite,
                    metadata=metadata,
                )
            )

            return ok_response(
                message=(
                    "Project pipeline initialized."
                ),
                session=session,
                project_pipeline=state,
            )

        except Exception as exc:
            return error_response(
                "Could not initialize project "
                f"pipeline: {exc}"
            )


    def handle_evaluate_project_pipeline(
        self,
        message,
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        if not session:
            return error_response(
                "Missing session."
            )

        workflow_id = str(
            message.get("workflow_id") or ""
        ).strip() or None

        processing_output_index = (
            message.get(
                "processing_output_index"
            )
        )

        if processing_output_index is not None:
            try:
                processing_output_index = int(
                    processing_output_index
                )
            except (
                TypeError,
                ValueError,
            ):
                return error_response(
                    "processing_output_index "
                    "must be an integer."
                )

            if processing_output_index < 1:
                return error_response(
                    "processing_output_index "
                    "must be at least 1."
                )

        try:
            result = (
                self.project_pipeline
                .evaluate_session(
                    session=session,
                    workflow_id=workflow_id,
                    processing_output_index=(
                        processing_output_index
                    ),
                )
            )

            state_after = result.get(
                "state_after",
                {},
            )

            return ok_response(
                message=result.get(
                    "message",
                    "Project pipeline evaluated.",
                ),
                session=session,
                transitioned=result.get(
                    "transitioned",
                    False,
                ),
                transition=result.get(
                    "transition"
                ),
                project_pipeline_status=(
                    state_after.get(
                        "status"
                    )
                ),
                evaluation=result,
            )

        except Exception as exc:
            return error_response(
                "Could not evaluate project "
                f"pipeline: {exc}"
            )


    def handle_get_project_pipeline_status(
        self,
        message,
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        if not session:
            return error_response(
                "Missing session."
            )

        try:
            state = (
                self.project_pipeline
                .get_status(
                    session=session,
                )
            )

            return ok_response(
                message=(
                    "Project pipeline status loaded."
                ),
                session=session,
                project_pipeline=state,
            )

        except FileNotFoundError as exc:
            return error_response(
                str(exc)
            )

        except Exception as exc:
            return error_response(
                "Could not get project pipeline "
                f"status: {exc}"
            )

    def handle_get_pending_project_action(
        self,
        message,
    ):
        target = str(
            message.get("target") or ""
        ).strip().lower()

        if not target:
            return error_response(
                "Missing target."
            )

        requested_session = str(
            message.get("session") or ""
        ).strip()

        try:
            # -----------------------------------------------------
            # Explicit session lookup
            # -----------------------------------------------------

            if requested_session:
                action = (
                    self.project_pipeline
                    .state_store
                    .get_pending_action(
                        session=requested_session,
                        target=target,
                    )
                )

                if action is None:
                    return ok_response(
                        message=(
                            "No pending project action "
                            "was found."
                        ),
                        target=target,
                        session=requested_session,
                        action=None,
                    )

                return ok_response(
                    message=(
                        "Pending project action found."
                    ),
                    target=target,
                    session=requested_session,
                    action={
                        **action,
                        "session": requested_session,
                    },
                )

            # -----------------------------------------------------
            # Search all persisted pipelines
            # -----------------------------------------------------

            sessions_root = (
                self.project_pipeline.sessions_root
            )

            if not sessions_root.is_dir():
                return ok_response(
                    message=(
                        "No pending project action "
                        "was found."
                    ),
                    target=target,
                    action=None,
                )

            candidates = []

            for session_path in (
                sessions_root.iterdir()
            ):
                if not session_path.is_dir():
                    continue

                state_path = (
                    session_path /
                    "Project_Pipeline" /
                    "project_pipeline_state.json"
                )

                if not state_path.is_file():
                    continue

                try:
                    action = (
                        self.project_pipeline
                        .state_store
                        .get_pending_action(
                            session=(
                                session_path.name
                            ),
                            target=target,
                        )
                    )
                except Exception:
                    continue

                if action is None:
                    continue

                if (
                    str(
                        action.get("status") or ""
                    ).strip().lower()
                    != "pending"
                ):
                    # Only unclaimed actions are offered
                    # through this endpoint.
                    continue

                candidates.append(
                    {
                        **action,
                        "session": (
                            session_path.name
                        ),
                    }
                )

            if not candidates:
                return ok_response(
                    message=(
                        "No pending project action "
                        "was found."
                    ),
                    target=target,
                    action=None,
                )

            candidates.sort(
                key=lambda item: (
                    str(
                        item.get("created_at")
                        or ""
                    ),
                    str(
                        item.get("action_id")
                        or ""
                    ),
                )
            )

            selected_action = (
                candidates[0]
            )

            return ok_response(
                message=(
                    "Pending project action found."
                ),
                target=target,
                session=selected_action[
                    "session"
                ],
                action=selected_action,
            )

        except Exception as exc:
            return error_response(
                "Could not get pending project "
                f"action: {exc}"
            )


    def handle_claim_project_action(
        self,
        message,
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        action_id = str(
            message.get("action_id") or ""
        ).strip()

        claimed_by = str(
            message.get("claimed_by") or ""
        ).strip().lower()

        if not session:
            return error_response(
                "Missing session."
            )

        if not action_id:
            return error_response(
                "Missing action_id."
            )

        if not claimed_by:
            return error_response(
                "Missing claimed_by."
            )

        try:
            action = (
                self.project_pipeline
                .state_store
                .claim_pending_action(
                    session=session,
                    action_id=action_id,
                    claimed_by=claimed_by,
                )
            )

            action_name = str(
                action.get("action") or ""
            ).strip().lower()

            state = (
                self.project_pipeline
                .state_store
                .load(
                    session
                )
            )

            current_status = str(
                state.get("status") or ""
            ).strip().lower()

            requested_status = None
            active_stage = None

            if (
                action_name == "run_design"
                and current_status
                == "design_pending"
            ):
                requested_status = (
                    "design_requested"
                )

                active_stage = "design"

            elif (
                action_name == "run_motion"
                and current_status
                == "motion_pending"
            ):
                requested_status = (
                    "motion_requested"
                )

                active_stage = "motion"

            if requested_status is not None:
                state = (
                    self.project_pipeline
                    .state_store
                    .transition(
                        session=session,
                        new_status=requested_status,
                        expected_status=(
                            current_status
                        ),
                        message=(
                            f"Project action "
                            f"{action_name} was claimed "
                            f"by {claimed_by}."
                        ),
                        active_stage=active_stage,
                        job_id=action_id,
                    )
                )

            return ok_response(
                message=(
                    "Project action claimed."
                ),
                session=session,
                action=action,
                project_pipeline=state,
            )

        except Exception as exc:
            return error_response(
                "Could not claim project "
                f"action: {exc}"
            )

    def handle_report_project_action(
        self,
        message,
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        action_id = str(
            message.get("action_id") or ""
        ).strip()

        report_status = str(
            message.get("action_status")
            or message.get("report_status")
            or ""
        ).strip().lower()

        report_message = str(
            message.get("message") or ""
        )

        result = message.get(
            "result"
        )

        error = message.get(
            "error"
        )

        if not session:
            return error_response(
                "Missing session."
            )

        if not action_id:
            return error_response(
                "Missing action_id."
            )

        if not report_status:
            return error_response(
                "Missing action_status."
            )

        try:
            action = (
                self.project_pipeline
                .state_store
                .report_pending_action(
                    session=session,
                    action_id=action_id,
                    status=report_status,
                    message=report_message,
                    result=result,
                    error=error,
                )
            )

            action_name = str(
                action.get("action") or ""
            ).strip().lower()

            state = (
                self.project_pipeline
                .state_store
                .load(
                    session
                )
            )

            current_status = str(
                state.get("status") or ""
            ).strip().lower()

            running_status = None
            active_stage = None

            if (
                report_status == "running"
                and action_name == "run_design"
                and current_status
                in {
                    "design_pending",
                    "design_requested",
                }
            ):
                running_status = (
                    "design_running"
                )

                active_stage = "design"

            elif (
                report_status == "running"
                and action_name == "run_motion"
                and current_status
                in {
                    "motion_pending",
                    "motion_requested",
                }
            ):
                running_status = (
                    "motion_running"
                )

                active_stage = "motion"

            if running_status is not None:
                state = (
                    self.project_pipeline
                    .state_store
                    .transition(
                        session=session,
                        new_status=running_status,
                        expected_status=(
                            current_status
                        ),
                        message=(
                            report_message
                            or (
                                f"Project action "
                                f"{action_name} is running."
                            )
                        ),
                        active_stage=active_stage,
                        job_id=action_id,
                    )
                )

            elif report_status == "failed":
                state = (
                    self.project_pipeline
                    .state_store
                    .transition(
                        session=session,
                        new_status="failed",
                        expected_status=(
                            current_status
                        ),
                        message=(
                            report_message
                            or (
                                f"Project action "
                                f"{action_name} failed."
                            )
                        ),
                        active_stage=None,
                        job_id=action_id,
                        error=error,
                    )
                )

            elif report_status == "cancelled":
                state = (
                    self.project_pipeline
                    .state_store
                    .transition(
                        session=session,
                        new_status="cancelled",
                        expected_status=(
                            current_status
                        ),
                        message=(
                            report_message
                            or (
                                f"Project action "
                                f"{action_name} "
                                "was cancelled."
                            )
                        ),
                        active_stage=None,
                        job_id=action_id,
                        error=error,
                    )
                )

            return ok_response(
                message=(
                    "Project action report accepted."
                ),
                session=session,
                action=action,
                project_pipeline=state,
            )

        except Exception as exc:
            return error_response(
                "Could not report project "
                f"action: {exc}"
            )

    # def handle_get_file_metadata(self, message):
    #     path = message.get("path")

    #     if not path:
    #         return error_response("Missing file path.")

    #     try:
    #         return ok_response(file=file_metadata(path))
    #     except Exception as e:
    #         return error_response(str(e))

    def handle_capture(self, message):
        job_id = self.jobs.create_job("capture")

        self.jobs.update_job(
            job_id,
            status="queued",
            progress=0,
            message="Capture job submitted."
        )

        thread = threading.Thread(
            target=self._run_capture_job,
            args=(job_id, message),
            daemon=True
        )
        thread.start()

        return ok_response(
            job_id=job_id,
            message="Capture job submitted.",
            job=self.jobs.get_job(job_id)
        )
    
    def _run_capture_job(self, job_id, message):
        try:
            self.jobs.update_job(
                job_id,
                status="running",
                progress=10,
                message="Starting capture."
            )

            from capture import CaptureTexturedPointCloud

            app = CaptureTexturedPointCloud()

            self.jobs.update_job(
                job_id,
                progress=25,
                message="Camera capture script initialized."
            )

            result = app.run_capture_from_config(message)

            if result.get("status") != "ok":
                self.jobs.update_job(
                    job_id,
                    status="failed",
                    progress=100,
                    message=result.get("message", "Capture failed."),
                    error=result.get("message", "Capture failed."),
                    result=result
                )
                return

            self.jobs.update_job(
                job_id,
                status="finished",
                progress=100,
                message="Capture finished.",
                result=result
            )

        except Exception as e:
            self.jobs.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Capture exception.",
                error=str(e)
            )

    def handle_transform(self, message):
        job_id = self.jobs.create_job("transform")

        self.jobs.update_job(
            job_id,
            status="queued",
            progress=0,
            message="Transform job submitted."
        )

        thread = threading.Thread(
            target=self._run_transform_job,
            args=(job_id, message),
            daemon=True
        )
        thread.start()

        return ok_response(
            job_id=job_id,
            message="Transform job submitted.",
            job=self.jobs.get_job(job_id)
        )
    
    def _run_transform_job(self, job_id, message):
        try:
            self.jobs.update_job(
                job_id,
                status="running",
                progress=10,
                message="Starting transform."
            )

            from transform import run_transform_from_config

            result = run_transform_from_config(message)

            if result.get("status") != "ok":
                self.jobs.update_job(
                    job_id,
                    status="failed",
                    progress=100,
                    message=result.get("message", "Transform failed."),
                    error=result.get("message", "Transform failed."),
                    result=result
                )
                return

            self.jobs.update_job(
                job_id,
                status="finished",
                progress=100,
                message="Transform finished.",
                result=result
            )

        except Exception as e:
            import traceback
            err = traceback.format_exc()

            self.jobs.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Transform exception.",
                error=err
            )

            print(err)

    def handle_process(self, message):
        """
        Creates a background processing job.

        The received message is passed directly to:
            processing.run_processing_from_config(message)

        Expected message fields may include:
            session
            output_index
            input_kind
            show_preview
            preview_time_sec
            color_grouping_method
            gmm_max_groups
            gmm_merge_distance
            params
        """
        job_id = self.jobs.create_job("process")

        self.jobs.update_job(
            job_id,
            status="queued",
            progress=0,
            message="Processing job submitted."
        )

        thread = threading.Thread(
            target=self._run_process_job,
            args=(job_id, dict(message)),
            daemon=True
        )
        thread.start()

        return ok_response(
            job_id=job_id,
            message="Processing job submitted.",
            job=self.jobs.get_job(job_id)
        )


    def _run_process_job(self, job_id, message):
        """
        Executes processing.py through its non-interactive API.
        """
        try:
            self.jobs.update_job(
                job_id,
                status="running",
                progress=5,
                message="Starting processing."
            )

            # processing.py must be importable from the Python project root.
            from processing import run_processing_from_config

            self.jobs.update_job(
                job_id,
                progress=15,
                message="Processing module loaded."
            )

            # Remove the agent command before passing the config.
            # processing.py does not need it, although leaving it would not
            # normally affect config.get(...) calls.
            processing_config = dict(message)
            processing_config.pop("command", None)

            result = run_processing_from_config(
                processing_config
            )

            if not isinstance(result, dict):
                raise RuntimeError(
                    "processing.run_processing_from_config() "
                    "did not return a dictionary."
                )

            if result.get("status") != "ok":
                failure_message = result.get(
                    "message",
                    "Processing failed."
                )

                self.jobs.update_job(
                    job_id,
                    status="failed",
                    progress=100,
                    message=failure_message,
                    error=failure_message,
                    result=result
                )
                return

            self.jobs.update_job(
                job_id,
                status="finished",
                progress=100,
                message="Processing finished.",
                result=result
            )

        except Exception:
            import traceback

            error_text = traceback.format_exc()

            self.jobs.update_job(
                job_id,
                status="failed",
                progress=100,
                message="Processing exception.",
                error=error_text
            )

            print(error_text)
    
    def handle_list_downloadable_outputs(self, message):
        

        session_name = message.get("session")
        output_index = int(message.get("output_index", 1))

        requested_types = message.get("file_types", ["pointcloud", "image", "json"])

        if isinstance(requested_types, str):
            requested_types = [requested_types]

        requested_types = {
            str(t).strip().lower()
            for t in requested_types
        }

        category_aliases = {
            "pointcloud": "pointclouds",
            "pointclouds": "pointclouds",
            "pcd": "pointclouds",
            "ply": "pointclouds",
            "image": "images",
            "images": "images",
            "img": "images",
            "png": "images",
            "json": "json",
        }

        allowed_categories = {
            category_aliases[t]
            for t in requested_types
            if t in category_aliases
        }

        paths = load_session(".", session_name)

        files = {
            "pointclouds": [],
            "images": [],
            "json": []
        }

        candidates = [        
            # Point clouds
            (
                "pointclouds",
                paths.merged_point_clouds
                / f"merged{output_index:02d}.ply"
            ),
            (
                "pointclouds",
                paths.merged_point_clouds
                / f"eye_to_base_point_cloud_{output_index:02d}.ply"
            ),
            (
                "pointclouds",
                paths.initial_point_clouds
                / f"point_cloud_{output_index:02d}.ply"
            ),

            # Images
            (
                "images",
                paths.merged_images
                / f"stitched_rgb_{output_index:02d}.png"
            ),
            (
                "images",
                paths.merged_depth_images
                / f"stitched_height_{output_index:02d}.png"
            ),
            (
                "images",
                paths.merged_images
                / f"eye_to_base_rgb_{output_index:02d}.png"
            ),
            (
                "images",
                paths.merged_depth_images
                / f"eye_to_base_height_{output_index:02d}.png"
            ),
            (
                "images",
                paths.initial_images
                / f"image_{output_index:02d}.png"
            ),
            (
                "images",
                paths.initial_depth_images
                / f"depth_{output_index:02d}.png"
            ),
            (
                "images",
                paths.initial_depth_images
                / f"depth_rendered_{output_index:02d}.png"
            ),

            # Processing outputs
            (
                "json",
                paths.exported_data
                / f"processed_clusters_{output_index:02d}.json"
            ),
            (
                "json",
                paths.exported_data
                / f"processing_params_used_{output_index:02d}.json"
            ),

            # The report is text, but it can temporarily travel in the JSON/data
            # category until you add a separate "report" file type.
            (
                "json",
                paths.exported_data
                / f"processing_report_{output_index:02d}.txt"
            ),
        ]

        for category, path in candidates:
            if allowed_categories and category not in allowed_categories:
                continue

            print(f"[download-list] checking {category}: {path} exists={path.exists()}")

            if path.exists():
                files[category].append({
                    "name": path.name,
                    "path": str(path.resolve()),
                    "size": os.path.getsize(path)
                })

        return ok_response(
            message="Downloadable outputs listed.",
            session=paths.session_name,
            output_index=output_index,
            files=files
        )

    def handle_get_design_output_file(
        self,
        message: Dict[str, Any],
    ):
        session = str(
            message.get("session") or ""
        ).strip()

        if not session:
            return error_response(
                "Missing session."
            )

        try:
            design_output_index = int(
                message.get(
                    "design_output_index"
                )
            )
        except (TypeError, ValueError):
            return error_response(
                "design_output_index must be an integer."
            )

        if design_output_index < 1:
            return error_response(
                "design_output_index must be at least 1."
            )

        requested_category = str(
            message.get("category") or
            "ghdata"
        ).strip().lower()

        try:
            output_entry = (
                self.design_output_store
                .get_output(
                    session=session,
                    design_output_index=(
                        design_output_index
                    ),
                )
            )

            output_status = str(
                output_entry.get("status") or ""
            ).strip().lower()

            if output_status != "finished":
                return error_response(
                    "Design output is not finished: "
                    f"session={session}, "
                    "design_output_index="
                    f"{design_output_index}, "
                    f"status={output_status or 'unknown'}."
                )

            matching_files = []

            for file_record in output_entry.get(
                "files",
                [],
            ):
                if not isinstance(
                    file_record,
                    dict,
                ):
                    continue

                category = str(
                    file_record.get("category") or ""
                ).strip().lower()

                if category != requested_category:
                    continue

                filename = str(
                    file_record.get("filename") or ""
                ).strip()

                if not filename:
                    continue

                matching_files.append(
                    {
                        "category": category,
                        "filename": filename,
                        "original_filename": str(
                            file_record.get(
                                "original_filename"
                            ) or ""
                        ).strip(),
                        "size_bytes": file_record.get(
                            "size_bytes"
                        ),
                    }
                )

            if not matching_files:
                return error_response(
                    "No design output file was found "
                    f"for category={requested_category}, "
                    f"session={session}, "
                    "design_output_index="
                    f"{design_output_index}."
                )

            if len(matching_files) > 1:
                return error_response(
                    "Multiple design output files matched "
                    f"category={requested_category}, "
                    f"session={session}, "
                    "design_output_index="
                    f"{design_output_index}. "
                    "The result is ambiguous."
                )

            selected_file = matching_files[0]

            output_folder = (
                self.design_output_store
                .get_output_folder(
                    session=session,
                    create=False,
                )
            )

            file_path = (
                output_folder /
                selected_file["filename"]
            )

            if not file_path.is_file():
                return error_response(
                    "Design output file is listed in the "
                    "manifest but does not exist: "
                    f"{file_path}"
                )

            actual_size = (
                file_path.stat().st_size
            )

            return ok_response(
                message=(
                    "Design output file resolved."
                ),
                session=session,
                design_output_index=(
                    design_output_index
                ),
                category=requested_category,
                file={
                    "name": file_path.name,
                    "path": str(
                        file_path.resolve()
                    ),
                    "size": actual_size,
                    "category": (
                        requested_category
                    ),
                    "original_filename": (
                        selected_file[
                            "original_filename"
                        ]
                    ),
                },
            )

        except Exception as exc:
            return error_response(
                "Could not resolve design output file: "
                f"{exc}"
            )