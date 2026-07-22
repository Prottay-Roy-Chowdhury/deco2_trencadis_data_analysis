import socket
import traceback
import threading
import time
import json
from pathlib import Path
from datetime import datetime

from gh_agent.config import (
    LOCAL_GH_AGENT_HOST,
    LOCAL_GH_AGENT_PORT,
    GH_AGENT_NAME,
    GH_AGENT_VERSION,
    POLL_INTERVAL_SEC,
    LOG_DIR,
    GH_AGENT_LOG_FILE,
    RECEIVED_ROOT
)

from gh_agent.protocol import (
    receive_message,
    send_message,
    ok_response,
    error_response,
)

from gh_agent.python_agent_client import PythonAgentClient
from gh_agent.local_upload_store import LocalUploadStore


class GrasshopperAgent:
    def __init__(
            self,
            host: str = LOCAL_GH_AGENT_HOST,
            port: int = LOCAL_GH_AGENT_PORT):
        self.host = host
        self.port = port
        self.python_client = PythonAgentClient()
        self.local_upload_store = LocalUploadStore()
        self.running = False

        self.lock = threading.Lock()

        self.latest_job_id = None
        self.latest_job_status = "idle"
        self.latest_message = "No job submitted yet."
        self.latest_result = None
        self.latest_error = None
        self.latest_log = []
        self.manifest_lock = threading.Lock()
        # Local GH-Agent jobs, currently used for downloads.
        # Python-side capture/transform/process jobs remain in the Python Agent.
        self.local_jobs_lock = threading.Lock()
        self.local_jobs = {}

        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def write_log(self, message):
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"

        with self.lock:
            self.latest_log.append(line)
            self.latest_log = self.latest_log[-50:]

        with open(GH_AGENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        print(f"[gh-agent-log] {line}")

    def set_latest(
        self,
        job_id=None,
        status=None,
        message=None,
        result=None,
        error=None,
        clear_error=False
    ):
        """
        Updates the latest GH Agent state.

        Set clear_error=True after a successful job to remove an older error.
        """
        with self.lock:
            if job_id is not None:
                self.latest_job_id = job_id

            if status is not None:
                self.latest_job_status = status

            if message is not None:
                self.latest_message = message

            if result is not None:
                self.latest_result = result

            if clear_error:
                self.latest_error = None

            elif error is not None:
                self.latest_error = error      
        

    def get_latest_status(self):
        with self.lock:
            return ok_response(
                latest_job_id=self.latest_job_id,
                latest_job_status=self.latest_job_status,
                latest_message=self.latest_message,
                latest_error=self.latest_error,
                latest_result=self.latest_result,
                recent_log=self.latest_log[-20:],
                log_file=str(GH_AGENT_LOG_FILE),
            )

    def start_polling_job(self, job_id):
        thread = threading.Thread(
            target=self.poll_job_until_done,
            args=(job_id,),
            daemon=True
        )
        thread.start()
    
    def start_polling_workflow(self, workflow_id, session):
        thread = threading.Thread(
            target=self.poll_workflow_until_done,
            args=(workflow_id, session),
            daemon=True,
        )
        thread.start()

    def poll_workflow_until_done(self, workflow_id, session):
        self.write_log(f"Started polling workflow: {workflow_id}")

        while self.running:
            try:
                response = self.python_client.send_command({
                    "command": "get_workflow_status",
                    "workflow_id": workflow_id,
                    "session": session,
                })

                if response.get("status") != "ok":
                    message = response.get(
                        "message",
                        "Failed to get workflow status.",
                    )

                    self.set_latest(
                        job_id=workflow_id,
                        status="unknown",
                        message=message,
                        error=message,
                    )

                    self.write_log(message)
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                workflow = response.get("workflow", {})
                workflow_status = workflow.get("status", "unknown")
                current_stage = workflow.get("current_stage")
                progress = workflow.get("progress", 0)
                workflow_error = workflow.get("error")

                if current_stage:
                    message = (
                        f"Workflow {workflow_status}: "
                        f"{current_stage} ({progress}%)."
                    )
                else:
                    message = (
                        f"Workflow {workflow_status} "
                        f"({progress}%)."
                    )

                self.set_latest(
                    job_id=workflow_id,
                    status=workflow_status,
                    message=message,
                    result=workflow,
                    error=workflow_error,
                    clear_error=not workflow_error,
                )

                self.write_log(
                    f"Workflow {workflow_id}: "
                    f"{workflow_status} - {message}"
                )

                if workflow_status in {
                    "finished",
                    "failed",
                    "cancelled",
                    "error",
                }:
                    self.write_log(
                        f"Workflow {workflow_id} completed "
                        f"with status: {workflow_status}"
                    )
                    break

            except Exception as exc:
                self.set_latest(
                    job_id=workflow_id,
                    status="error",
                    message=str(exc),
                    error=str(exc),
                )

                self.write_log(
                    f"Workflow polling error for "
                    f"{workflow_id}: {exc}"
                )

            time.sleep(POLL_INTERVAL_SEC)

    def poll_job_until_done(self, job_id):
        self.write_log(f"Started polling job: {job_id}")

        while self.running:
            try:
                response = self.python_client.send_command({
                    "command": "get_status",
                    "job_id": job_id
                })

                if response.get("status") != "ok":
                    msg = response.get("message", "Failed to get job status.")
                    self.set_latest(
                        job_id=job_id,
                        status="unknown",
                        message=msg,
                        error=msg
                    )
                    self.write_log(msg)
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                job = response.get("job", {})
                job_status = job.get("status", "unknown")
                job_message = job.get("message", "")
                job_result = job.get("result")
                job_error = job.get("error")

                self.set_latest(
                    job_id=job_id,
                    status=job_status,
                    message=job_message,
                    result=job_result,
                    error=job_error
                )

                self.write_log(
                    f"Job {job_id}: {job_status} - {job_message}"
                )

                if job_status in (
                    "finished",
                    "failed",
                    "cancelled",
                    "error"
                ):
                    self.write_log(
                        f"Job {job_id} completed with status: "
                        f"{job_status}"
                    )
                    break

            except Exception as e:
                self.set_latest(
                    job_id=job_id,
                    status="error",
                    message=str(e),
                    error=str(e)
                )
                self.write_log(f"Polling error for job {job_id}: {e}")

            time.sleep(POLL_INTERVAL_SEC)

    def submit_python_job(self, payload):
        response = self.python_client.send_command(payload)

        job_id = response.get("job_id")

        if not job_id:
            return response

        self.set_latest(
            job_id=job_id,
            status="submitted",
            message=response.get("message", "Job submitted."),
            result=None,
            clear_error=True
        )

        self.write_log(f"Submitted job {job_id}: {payload}")

        self.start_polling_job(job_id)

        return ok_response(
            message="Job submitted to Python Agent.",
            job_id=job_id,
            latest_status="submitted"
        )
    

    def submit_python_workflow(self, payload):
        response = self.python_client.send_command(payload)

        if response.get("status") != "ok":
            return response

        workflow_id = response.get("workflow_id")
        session = payload.get("session")

        if not workflow_id:
            return error_response(
                "Python Agent did not return a workflow_id."
            )

        self.set_latest(
            job_id=workflow_id,
            status="submitted",
            message=response.get(
                "message",
                "Workflow submitted.",
            ),
            result=response.get("workflow"),
            clear_error=True,
        )

        self.write_log(
            f"Submitted workflow {workflow_id}: {payload}"
        )

        self.start_polling_workflow(
            workflow_id=workflow_id,
            session=session,
        )

        return ok_response(
            message="Workflow submitted to Python Agent.",
            workflow_id=workflow_id,
            session=session,
            latest_status="submitted",
            workflow=response.get("workflow"),
        )
    

    def create_local_job(
        self,
        job_id,
        job_type,
        status="queued",
        progress=0,
        message="",
        result=None,
        error=None
    ):
        """
        Creates or replaces a locally managed GH-Agent job record.
        """
        now = datetime.now().isoformat(
            timespec="seconds"
        )

        job = {
            "job_id": str(job_id),
            "job_type": str(job_type),
            "status": str(status),
            "progress": int(progress),
            "message": str(message),
            "result": result,
            "error": error,
            "created_at": now,
            "updated_at": now
        }

        with self.local_jobs_lock:
            self.local_jobs[str(job_id)] = job

        return dict(job)


    def update_local_job(
        self,
        job_id,
        status=None,
        progress=None,
        message=None,
        result=None,
        error=None,
        clear_error=False
    ):
        """
        Updates one locally managed GH-Agent job.
        """
        job_key = str(job_id)

        with self.local_jobs_lock:
            if job_key not in self.local_jobs:
                raise KeyError(
                    f"Unknown local job ID: {job_key}"
                )

            job = self.local_jobs[job_key]

            if status is not None:
                job["status"] = str(status)

            if progress is not None:
                job["progress"] = int(progress)

            if message is not None:
                job["message"] = str(message)

            if result is not None:
                job["result"] = result

            if clear_error:
                job["error"] = None

            elif error is not None:
                job["error"] = error

            job["updated_at"] = (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )

            return dict(job)


    def get_local_job(self, job_id):
        """
        Returns one local GH-Agent job record.
        """
        job_key = str(job_id)

        with self.local_jobs_lock:
            if job_key not in self.local_jobs:
                raise KeyError(
                    f"Unknown local job ID: {job_key}"
                )

            return dict(
                self.local_jobs[job_key]
            )
    
    def get_session_manifest_path(
        self,
        session_name,
        create_session_folder=False
    ):
        """
        Returns the per-session manifest path.

        The folder is created only when explicitly requested, normally
        during a real download or manifest save.
        """
        safe_session = str(session_name).strip()

        if not safe_session:
            raise ValueError(
                "Session name cannot be empty."
            )

        session_root = RECEIVED_ROOT / safe_session

        if create_session_folder:
            session_root.mkdir(
                parents=True,
                exist_ok=True
            )

        return session_root / "download_manifest.json"


    def create_empty_session_manifest(self, session_name):
        """
        Creates the default in-memory structure for one session.
        """
        return {
            "session": str(session_name),
            "updated_at": None,
            "outputs": {}
        }


    def load_session_manifest(self, session_name):
        """
        Loads one session's local download manifest.

        Missing or invalid manifests return an empty structure.
        """
        manifest_path = self.get_session_manifest_path(
            session_name,
            create_session_folder=False
        )

        if not manifest_path.exists():
            return self.create_empty_session_manifest(
                session_name
            )

        try:
            with open(
                manifest_path,
                "r",
                encoding="utf-8"
            ) as file:
                manifest = json.load(file)

        except Exception as exc:
            self.write_log(
                f"Failed to read manifest for "
                f"{session_name}: {exc}"
            )

            return self.create_empty_session_manifest(
                session_name
            )

        if not isinstance(manifest, dict):
            manifest = self.create_empty_session_manifest(
                session_name
            )

        manifest["session"] = str(session_name)

        if not isinstance(
            manifest.get("outputs"),
            dict
        ):
            manifest["outputs"] = {}

        return manifest


    def save_session_manifest(
        self,
        session_name,
        manifest
    ):
        """
        Saves one session manifest atomically.
        """
        manifest_path = self.get_session_manifest_path(
            session_name,
            create_session_folder=True
        )

        manifest["session"] = str(session_name)
        manifest["updated_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        temporary_path = manifest_path.with_suffix(
            ".json.tmp"
        )

        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                manifest,
                file,
                ensure_ascii=False,
                indent=2
            )

        temporary_path.replace(
            manifest_path
        )
    
    def infer_download_category(
        self,
        file_type,
        filename
    ):
        """
        Infers the logical category from the downloaded filename.
        """
        file_type = str(
            file_type
        ).strip().lower()

        name = str(
            filename
        ).strip().lower()

        if file_type == "pointclouds":
            if name.startswith(
                "eye_to_base_point_cloud_"
            ):
                return "eye_to_base_pointcloud"

            if name.startswith("merged"):
                return "merged_pointcloud"

            if name.startswith(
                "point_cloud_"
            ):
                return "initial_pointcloud"

        elif file_type == "images":
            if name.startswith(
                "stitched_rgb_"
            ):
                return "stitched_rgb"

            if name.startswith(
                "stitched_height_"
            ):
                return "stitched_height"

            if name.startswith(
                "eye_to_base_rgb_"
            ):
                return "eye_to_base_rgb"

            if name.startswith(
                "eye_to_base_height_"
            ):
                return "eye_to_base_height"

            if name.startswith(
                "depth_rendered_"
            ):
                return "initial_depth_rendered"

            if name.startswith(
                "depth_"
            ):
                return "initial_depth"

            if name.startswith(
                "image_"
            ):
                return "initial_rgb"

        elif file_type == "json":
            if name.startswith(
                "processed_clusters_"
            ):
                return "processed_clusters"

            if name.startswith(
                "processing_params_used_"
            ):
                return "processing_params"

            if name.startswith(
                "processing_report_"
            ):
                return "processing_report"

        return "unknown"
    
    def build_manifest_file_entry(
        self,
        file_type,
        downloaded_item
    ):
        """
        Converts one successful download result into a manifest entry.
        """
        local_path = str(
            downloaded_item.get(
                "local_path",
                ""
            )
        )

        path = Path(local_path)

        name = str(
            downloaded_item.get(
                "name",
                path.name
            )
        )

        remote_path = str(
            downloaded_item.get(
                "remote_path",
                ""
            )
        )

        exists = path.exists()

        size = (
            path.stat().st_size
            if exists
            else None
        )

        return {
            "name": name,
            "category": (
                self.infer_download_category(
                    file_type,
                    name
                )
            ),
            "local_path": local_path,
            "remote_path": remote_path,
            "size": size,
            "exists": bool(exists),
            "downloaded_at": (
                datetime.now().isoformat(
                    timespec="seconds"
                )
            )
        }


    def upsert_manifest_entry(
        self,
        entries,
        new_entry
    ):
        """
        Updates an existing file entry or appends a new one.

        Uniqueness is based on category + filename.
        """
        new_name = str(
            new_entry.get(
                "name",
                ""
            )
        ).lower()

        new_category = str(
            new_entry.get(
                "category",
                ""
            )
        ).lower()

        for index, existing in enumerate(
            entries
        ):
            existing_name = str(
                existing.get(
                    "name",
                    ""
                )
            ).lower()

            existing_category = str(
                existing.get(
                    "category",
                    ""
                )
            ).lower()

            if (
                existing_name == new_name
                and existing_category
                == new_category
            ):
                entries[index] = new_entry
                return

        entries.append(new_entry)


    def update_session_manifest_from_downloads(
        self,
        session_name,
        output_index,
        downloaded_items
    ):
        """
        Merges successful downloads into one session/output manifest.
        """
        output_key = str(
            int(output_index)
        )

        with self.manifest_lock:
            manifest = self.load_session_manifest(
                session_name
            )

            outputs = manifest.setdefault(
                "outputs",
                {}
            )

            output_entry = outputs.setdefault(
                output_key,
                {
                    "pointclouds": [],
                    "images": [],
                    "json": []
                }
            )

            for file_type in (
                "pointclouds",
                "images",
                "json"
            ):
                if not isinstance(
                    output_entry.get(file_type),
                    list
                ):
                    output_entry[file_type] = []

            for item in downloaded_items:
                file_type = str(
                    item.get(
                        "category",
                        ""
                    )
                ).strip().lower()

                if file_type not in (
                    "pointclouds",
                    "images",
                    "json"
                ):
                    continue

                local_path = Path(
                    item.get(
                        "local_path",
                        ""
                    )
                )

                if (
                    not local_path.exists()
                    or not local_path.is_file()
                ):
                    self.write_log(
                        f"Manifest skipped missing local file: "
                        f"{local_path}"
                    )
                    continue

                manifest_entry = (
                    self.build_manifest_file_entry(
                        file_type,
                        item
                    )
                )

                self.upsert_manifest_entry(
                    output_entry[file_type],
                    manifest_entry
                )

            self.save_session_manifest(
                session_name,
                manifest
            )

    def list_local_downloads(
        self,
        session_name,
        output_index,
        file_types=None,
        categories=None
    ):
        """
        Reads locally downloaded files for one session and output.

        file_types:
            broad types such as pointcloud, image, json

        categories:
            specific selectors such as stitched_rgb,
            processed_clusters, eye_to_base_pointcloud
        """
        manifest = self.load_session_manifest(
            session_name
        )

        output_key = str(
            int(output_index)
        )

        output_entry = manifest.get(
            "outputs",
            {}
        ).get(
            output_key,
            {}
        )

        type_aliases = {
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

        if isinstance(file_types, str):
            file_types = [
                file_types
            ]

        requested_types = set()

        for value in (
            file_types
            or [
                "pointcloud",
                "image",
                "json"
            ]
        ):
            normalized = str(
                value
            ).strip().lower()

            mapped = type_aliases.get(
                normalized
            )

            if mapped:
                requested_types.add(mapped)

        if isinstance(categories, str):
            categories = [
                value.strip()
                for value in categories.split(",")
                if value.strip()
            ]

        requested_categories = {
            str(value).strip().lower()
            for value in (
                categories
                or []
            )
            if str(value).strip()
        }

        files = {
            "pointclouds": [],
            "images": [],
            "json": []
        }

        for file_type in files.keys():
            if file_type not in requested_types:
                continue

            stored_entries = output_entry.get(
                file_type,
                []
            )

            if not isinstance(
                stored_entries,
                list
            ):
                continue

            for stored_entry in stored_entries:
                item = dict(stored_entry)

                local_path = Path(
                    item.get(
                        "local_path",
                        ""
                    )
                )

                item["exists"] = (
                    local_path.exists()
                )

                if item["exists"]:
                    item["size"] = (
                        local_path.stat().st_size
                    )

                category = str(
                    item.get(
                        "category",
                        "unknown"
                    )
                ).lower()

                if (
                    requested_categories
                    and category
                    not in requested_categories
                ):
                    continue

                files[file_type].append(
                    item
                )

        return files

    def start_download_job(self, message):
        download_job_id = (
            f"download-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )

        local_job = self.create_local_job(
            job_id=download_job_id,
            job_type="download",
            status="download_queued",
            progress=0,
            message="Download job submitted.",
            result=None,
            error=None
        )

        # Keep global latest status for the future workflow manager.
        self.set_latest(
            job_id=download_job_id,
            status="download_queued",
            message="Download job submitted.",
            result=None,
            clear_error=True
        )

        self.write_log(
            f"Download job submitted: {message}"
        )

        thread = threading.Thread(
            target=self._run_download_job,
            args=(
                download_job_id,
                dict(message)
            ),
            daemon=True
        )

        thread.start()

        return ok_response(
            message="Download job submitted to GH Agent.",
            job_id=download_job_id,
            latest_status="download_queued",
            job=local_job
        )
    
    def _run_download_job(self, job_id, message):
        try:
            session = message.get("session")

            output_index = int(
                message.get(
                    "output_index",
                    1
                )
            )

            requested_types = message.get(
                "file_types",
                [
                    "pointcloud",
                    "image",
                    "json"
                ]
            )

            if isinstance(requested_types, str):
                requested_types = [
                    requested_types
                ]

            requested_types = [
                str(file_type).strip().lower()
                for file_type in requested_types
            ]

            # ---------------------------------------------------------
            # Start the local and global download states
            # ---------------------------------------------------------

            self.update_local_job(
                job_id=job_id,
                status="download_running",
                progress=5,
                message="Listing downloadable files.",
                clear_error=True
            )

            self.set_latest(
                job_id=job_id,
                status="download_running",
                message="Listing downloadable files.",
                clear_error=True
            )

            self.write_log(
                f"Download running: "
                f"session={session}, "
                f"output_index={output_index}, "
                f"file_types={requested_types}"
            )

            # ---------------------------------------------------------
            # Ask the Python Agent which files are available
            # ---------------------------------------------------------

            listing = self.python_client.send_command({
                "command": "list_downloadable_outputs",
                "session": session,
                "output_index": output_index,
                "file_types": requested_types
            })

            if listing.get("status") != "ok":
                failure_message = listing.get(
                    "message",
                    "Failed to list downloadable outputs."
                )

                self.update_local_job(
                    job_id=job_id,
                    status="download_failed",
                    progress=100,
                    message=failure_message,
                    error=listing
                )

                self.set_latest(
                    job_id=job_id,
                    status="download_failed",
                    message=failure_message,
                    error=listing
                )

                self.write_log(
                    f"Download failed while listing files: "
                    f"{listing}"
                )

                return

            saved = []

            files = listing.get(
                "files",
                {}
            )

            total = sum(
                len(items)
                for items in files.values()
            )

            count = 0

            # ---------------------------------------------------------
            # Download every returned file
            # ---------------------------------------------------------

            for category, items in files.items():
                for item in items:
                    count += 1

                    remote_path = item["path"]
                    name = item["name"]

                    local_path = (
                        RECEIVED_ROOT
                        / listing["session"]
                        / category
                        / name
                    )

                    if total > 0:
                        progress = 10 + int(
                            80.0 * count / total
                        )
                    else:
                        progress = 90

                    download_message = (
                        f"Downloading "
                        f"{count}/{total}: "
                        f"{name}"
                    )

                    self.update_local_job(
                        job_id=job_id,
                        status="download_running",
                        progress=progress,
                        message=download_message
                    )

                    self.set_latest(
                        job_id=job_id,
                        status="download_running",
                        message=download_message
                    )

                    self.write_log(
                        f"Downloading {name} "
                        f"→ {local_path}"
                    )

                    result = (
                        self.python_client.download_file(
                            remote_path,
                            local_path
                        )
                    )

                    saved.append({
                        "category": category,
                        "name": name,
                        "remote_path": remote_path,
                        "local_path": str(local_path),
                        "result": result
                    })

            # ---------------------------------------------------------
            # Update the persistent per-session manifest
            # ---------------------------------------------------------

            # This stays outside both download loops.
            self.update_session_manifest_from_downloads(
                session_name=listing["session"],
                output_index=output_index,
                downloaded_items=saved
            )

            manifest_path = self.get_session_manifest_path(
                listing["session"]
            )

            final_result = {
                "session": listing["session"],
                "output_index": output_index,
                "saved": saved,
                "manifest_path": str(
                    manifest_path
                )
            }

            finished_message = (
                f"Downloaded {len(saved)} file(s) "
                f"and updated the session manifest."
            )

            # ---------------------------------------------------------
            # Finish both the local job and global latest state
            # ---------------------------------------------------------

            self.update_local_job(
                job_id=job_id,
                status="download_finished",
                progress=100,
                message=finished_message,
                result=final_result,
                clear_error=True
            )

            self.set_latest(
                job_id=job_id,
                status="download_finished",
                message=finished_message,
                result=final_result,
                clear_error=True
            )

            self.write_log(
                f"Download finished: "
                f"{len(saved)} file(s). "
                f"Manifest: {manifest_path}"
            )

        except Exception:
            error_text = traceback.format_exc()

            # Update the local download job if it was already created.
            try:
                self.update_local_job(
                    job_id=job_id,
                    status="download_failed",
                    progress=100,
                    message="Download exception.",
                    error=error_text
                )
            except KeyError:
                pass

            self.set_latest(
                job_id=job_id,
                status="download_failed",
                message="Download exception.",
                error=error_text
            )

            self.write_log(
                f"Download exception:\n"
                f"{error_text}"
            )

            print(error_text)

    def get_upload_session_info(
        self,
        session,
    ):
        """
        Creates the local upload session folder and manifest on demand,
        then returns their paths.
        """
        session_name = str(session or "").strip()

        if not session_name:
            raise ValueError("Missing session.")

        session_folder = (
            self.local_upload_store.get_session_folder(
                session=session_name,
                create=True,
            )
        )

        manifest = (
            self.local_upload_store.load_manifest(
                session_name
            )
        )

        manifest_path = (
            self.local_upload_store.save_manifest(
                session=session_name,
                manifest=manifest,
            )
        )

        return {
            "session": session_name,
            "session_folder": str(session_folder.resolve()),
            "manifest_path": str(manifest_path.resolve()),
        }

    
    def start_design_upload_job(self, message):
        upload_job_id = (
            f"design-upload-"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}"
        )

        local_job = self.create_local_job(
            job_id=upload_job_id,
            job_type="design_upload",
            status="upload_queued",
            progress=0,
            message="Design upload job submitted.",
            result=None,
            error=None,
        )

        self.set_latest(
            job_id=upload_job_id,
            status="upload_queued",
            message="Design upload job submitted.",
            result=None,
            clear_error=True,
        )

        self.write_log(
            f"Design upload job submitted: {message}"
        )

        thread = threading.Thread(
            target=self._run_design_upload_job,
            args=(
                upload_job_id,
                dict(message),
            ),
            daemon=True,
        )
        thread.start()

        return ok_response(
            message="Design upload job submitted to GH Agent.",
            job_id=upload_job_id,
            latest_status="upload_queued",
            job=local_job,
        )
    

    def _run_design_upload_job(
        self,
        job_id,
        message,
    ):
        try:
            session = str(
                message.get("session") or ""
            ).strip()

            if not session:
                raise ValueError("Missing session.")

            try:
                design_output_index = int(
                    message.get("design_output_index")
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "design_output_index must be an integer."
                ) from exc

            if design_output_index < 1:
                raise ValueError(
                    "design_output_index must be at least 1."
                )

            source_processing_output_index = (
                message.get(
                    "source_processing_output_index"
                )
            )

            if source_processing_output_index is not None:
                try:
                    source_processing_output_index = int(
                        source_processing_output_index
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "source_processing_output_index "
                        "must be an integer."
                    ) from exc

                if source_processing_output_index < 1:
                    raise ValueError(
                        "source_processing_output_index "
                        "must be at least 1."
                    )

            files = message.get("files")

            if not isinstance(files, list) or not files:
                raise ValueError(
                    "files must be a non-empty array."
                )

            prepared_files = []

            for index, file_record in enumerate(files):
                if not isinstance(file_record, dict):
                    raise ValueError(
                        f"files[{index}] must be an object."
                    )

                local_path = Path(
                    file_record.get("path") or ""
                ).expanduser()

                category = str(
                    file_record.get("category") or ""
                ).strip()

                if not local_path.is_file():
                    raise FileNotFoundError(
                        f"Design file does not exist: "
                        f"{local_path}"
                    )

                if not category:
                    raise ValueError(
                        f"files[{index}] is missing category."
                    )

                prepared_files.append(
                    {
                        "path": str(
                            local_path.resolve()
                        ),
                        "category": category,
                    }
                )

            created_by = str(
                message.get("created_by")
                or "design_pc"
            ).strip()

            upload_message = str(
                message.get("message") or ""
            )

            self.update_local_job(
                job_id=job_id,
                status="upload_running",
                progress=10,
                message=(
                    "Validated design files. "
                    "Connecting to Python master."
                ),
                clear_error=True,
            )

            self.set_latest(
                job_id=job_id,
                status="upload_running",
                message=(
                    "Validated design files. "
                    "Connecting to Python master."
                ),
                clear_error=True,
            )

            self.write_log(
                "Design upload running: "
                f"session={session}, "
                f"design_output_index="
                f"{design_output_index}, "
                f"file_count={len(prepared_files)}"
            )

            self.update_local_job(
                job_id=job_id,
                status="upload_running",
                progress=25,
                message=(
                    f"Uploading {len(prepared_files)} "
                    "design file(s)."
                ),
            )

            self.set_latest(
                job_id=job_id,
                status="upload_running",
                message=(
                    f"Uploading {len(prepared_files)} "
                    "design file(s)."
                ),
            )

            result = self.python_client.upload_design_output(
                session=session,
                design_output_index=design_output_index,
                files=prepared_files,
                source_processing_output_index=(
                    source_processing_output_index
                ),
                created_by=created_by,
                message=upload_message,
            )

            if result.get("status") != "ok":
                failure_message = result.get(
                    "message",
                    "Design upload failed.",
                )

                self.update_local_job(
                    job_id=job_id,
                    status="upload_failed",
                    progress=100,
                    message=failure_message,
                    result=result,
                    error=result,
                )

                self.set_latest(
                    job_id=job_id,
                    status="upload_failed",
                    message=failure_message,
                    result=result,
                    error=result,
                )

                self.write_log(
                    "Design upload failed: "
                    f"{result}"
                )

                return

            finished_message = (
                "Design output uploaded successfully: "
                f"session={session}, "
                f"design_output_index="
                f"{design_output_index}."
            )

            final_result = {
                "session": session,
                "design_output_index": (
                    design_output_index
                ),
                "source_processing_output_index": (
                    source_processing_output_index
                ),
                "uploaded_files": prepared_files,
                "master_response": result,
            }

            self.update_local_job(
                job_id=job_id,
                status="upload_finished",
                progress=100,
                message=finished_message,
                result=final_result,
                clear_error=True,
            )

            self.set_latest(
                job_id=job_id,
                status="upload_finished",
                message=finished_message,
                result=final_result,
                clear_error=True,
            )

            self.write_log(finished_message)

        except Exception:
            error_text = traceback.format_exc()

            try:
                self.update_local_job(
                    job_id=job_id,
                    status="upload_failed",
                    progress=100,
                    message="Design upload exception.",
                    error=error_text,
                )
            except KeyError:
                pass

            self.set_latest(
                job_id=job_id,
                status="upload_failed",
                message="Design upload exception.",
                error=error_text,
            )

            self.write_log(
                "Design upload exception:\n"
                f"{error_text}"
            )

            print(error_text)
    

    def handle_local_message(self, message):
        command = message.get("command")

        if not command:
            return error_response("Missing command.")

        if command == "ping":
            return ok_response(
                message="Grasshopper Agent is alive."
            )

        if command == "ping_python":
            return self.python_client.send_command(
                {
                    "command": "ping"
                }
            )
        if command == "get_job_status":
            job_id = message.get(
                "job_id"
            )

            if not job_id:
                return error_response(
                    "Missing job_id."
                )

            job_id = str(job_id)

            # Download & Upload jobs are managed locally by the GH Agent.
            if (
                job_id.startswith("download-")
                or job_id.startswith("design-upload-")
            ):
                try:
                    local_job = self.get_local_job(job_id)

                    return ok_response(
                        message="Local job status returned.",
                        job=local_job,
                    )
                except KeyError as exc:
                    return error_response(str(exc))

            # Capture, transform and process jobs belong to the Python Agent.
            python_response = (
                self.python_client.send_command({
                    "command": "get_status",
                    "job_id": job_id
                })
            )

            return python_response

        if command == "get_latest_status":
            return self.get_latest_status()

        if command == "forward":
            payload = message.get("payload")

            if not isinstance(payload, dict):
                return error_response("Missing or invalid payload.")

            return self.submit_python_job(payload)
        
        if command == "downloadable_outputs":
            return self.start_download_job(message)

        if command == "get_upload_session_path":
            session = message.get("session")

            if not session:
                return error_response("Missing session.")

            try:
                session_info = self.get_upload_session_info(
                    session=session,
                )

                return ok_response(
                    message="Local upload session is ready.",
                    **session_info,
                )

            except Exception as exc:
                return error_response(str(exc))

        if command == "register_local_upload":
            session = message.get("session")
            local_path = message.get("local_path")
            domain = message.get("domain", "design")
            category = message.get("category", "geometry")

            if not session:
                return error_response("Missing session.")

            if not local_path:
                return error_response("Missing local_path.")

            try:
                record = self.local_upload_store.register_file(
                    session=session,
                    local_path=local_path,
                    domain=domain,
                    category=category,
                    design_output_index=message.get(
                        "design_output_index"
                    ),
                    motion_output_index=message.get(
                        "motion_output_index"
                    ),
                    upload_status=message.get(
                        "upload_status",
                        "pending",
                    ),
                )

                return ok_response(
                    message="Local upload file registered.",
                    session=str(session),
                    file=record,
                    manifest_path=str(
                        self.local_upload_store
                        .get_manifest_path(session)
                        .resolve()
                    ),
                )

            except Exception as exc:
                return error_response(str(exc))

        if command == "list_local_uploads":
            session = message.get("session")

            if not session:
                return error_response("Missing session.")

            try:
                files = self.local_upload_store.list_files(
                    session=session,
                    domain=message.get("domain"),
                    design_output_index=message.get(
                        "design_output_index"
                    ),
                    upload_status=message.get(
                        "upload_status"
                    ),
                )

                return ok_response(
                    message="Local upload files listed.",
                    session=str(session),
                    manifest_path=str(
                        self.local_upload_store
                        .get_manifest_path(
                            session=session,
                            create_session_folder=False,
                        )
                        .resolve()
                    ),
                    files=files,
                )

            except Exception as exc:
                return error_response(str(exc))
        
        if command == "upload_design_output":
            return self.start_design_upload_job(message)
        
        if command == "list_local_downloads":
            session = message.get(
                "session"
            )

            if not session:
                return error_response(
                    "Missing session."
                )

            output_index = int(
                message.get(
                    "output_index",
                    1
                )
            )

            file_types = message.get(
                "file_types",
                [
                    "pointcloud",
                    "image",
                    "json"
                ]
            )

            categories = message.get(
                "categories",
                message.get(
                    "category",
                    []
                )
            )

            files = self.list_local_downloads(
                session_name=session,
                output_index=output_index,
                file_types=file_types,
                categories=categories
            )

            return ok_response(
                message=(
                    "Local session downloads listed."
                ),
                session=str(session),
                output_index=output_index,
                manifest_path=str(
                    self.get_session_manifest_path(
                        session,
                        create_session_folder=False
                    )
                ),
                files=files
            )
        
        if command == "submit_workflow":
            payload = message.get("payload")

            if payload is None:
                payload = {
                    key: value
                    for key, value in message.items()
                    if key != "command"
                }

            if not isinstance(payload, dict):
                return error_response(
                    "Missing or invalid workflow payload."
                )

            payload = dict(payload)
            payload["command"] = "submit_workflow"

            return self.submit_python_workflow(payload)
        
        if command == "get_workflow_status":
            workflow_id = message.get("workflow_id")

            if not workflow_id:
                return error_response("Missing workflow_id.")

            payload = {
                "command": "get_workflow_status",
                "workflow_id": workflow_id,
            }

            if message.get("session"):
                payload["session"] = message["session"]

            return self.python_client.send_command(payload)
        
        if command == "cancel_workflow":
            workflow_id = message.get("workflow_id")

            if not workflow_id:
                return error_response("Missing workflow_id.")

            payload = {
                "command": "cancel_workflow",
                "workflow_id": workflow_id,
            }

            if message.get("session"):
                payload["session"] = message["session"]

            return self.python_client.send_command(payload)
        
        if command == "list_workflows":
            session = message.get("session")

            if not session:
                return error_response("Missing session.")

            return self.python_client.send_command({
                "command": "list_workflows",
                "session": session,
            })

        return error_response(f"Unknown GH Agent command: {command}")

    def start(self):
        print(f"[gh-agent] {GH_AGENT_NAME} v{GH_AGENT_VERSION}")
        print(f"[gh-agent] Listening locally on {self.host}:{self.port}")
        print(f"[gh-agent] Forwarding to Python Agent at "
              f"{self.python_client.host}:{self.python_client.port}")
        print("[gh-agent] Press Ctrl+C to stop.")

        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(1.0)

            try:
                while self.running:
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        continue

                    with client:
                        print(f"[gh-agent] Local connection: {address}")

                        try:
                            message = receive_message(client)
                            print(f"[gh-agent] Received local message: {message}")

                            response = self.handle_local_message(message)

                        except Exception as e:
                            traceback.print_exc()
                            response = error_response(str(e))

                        send_message(client, response)
                        print("[gh-agent] Response sent.")

            except KeyboardInterrupt:
                print("\n[gh-agent] Ctrl+C received.")

            finally:
                self.running = False
                print("[gh-agent] Shutting down...")


def main():
    agent = GrasshopperAgent()
    agent.start()


if __name__ == "__main__":
    main()