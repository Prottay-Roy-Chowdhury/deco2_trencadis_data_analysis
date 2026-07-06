from datetime import datetime
from typing import Dict, Any
import uuid


class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, command: str) -> str:
        job_id = str(uuid.uuid4())

        self.jobs[job_id] = {
            "job_id": job_id,
            "command": command,
            "status": "created",
            "progress": 0,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "message": "",
            "log": [],
            "result": None,
            "error": None,
        }

        return job_id

    def update_job(
            self,
            job_id: str,
            status: str | None = None,
            progress: int | None = None,
            message: str | None = None,
            result: Any | None = None,
            error: str | None = None):

        if job_id not in self.jobs:
            raise KeyError(f"Unknown job_id: {job_id}")

        job = self.jobs[job_id]

        if status is not None:
            job["status"] = status

        if progress is not None:
            job["progress"] = progress

        if message is not None:
            job["message"] = message
            job["log"].append(
                f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
            )

        if result is not None:
            job["result"] = result

        if error is not None:
            job["error"] = error
            job["log"].append(
                f"[{datetime.now().isoformat(timespec='seconds')}] ERROR: {error}"
            )

        job["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def get_job(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            raise KeyError(f"Unknown job_id: {job_id}")

        return self.jobs[job_id]

    def list_jobs(self):
        return list(self.jobs.values())