import os
import threading
from typing import Dict, Any
from pathlib import Path
from helpers.session_manager import load_session
from helpers.design_output_store import DesignOutputStore
from workflow.bootstrap import build_default_workflow_manager

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
            return error_response("stage_configs must be an object.")

        if not isinstance(runtime, dict):
            return error_response("runtime must be an object.")

        try:
            workflow = self.workflows.submit_workflow(
                workflow_name=workflow_name,
                session=session,
                stage_configs=stage_configs,
                selected_stages=selected_stages,
                start_stage=start_stage,
                runtime=runtime,
            )

            return ok_response(
                workflow_id=workflow["workflow_id"],
                message="Workflow submitted.",
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