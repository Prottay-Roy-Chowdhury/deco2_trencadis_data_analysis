import json
import threading
from pathlib import Path


class WorkflowStore:
    def __init__(self, sessions_root, workflow_folder_name="Workflows"):
        self.sessions_root = Path(sessions_root)
        self.workflow_folder_name = workflow_folder_name
        self._lock = threading.RLock()

    def get_workflow_path(self, session, workflow_id, create_folder=False):
        session = str(session).strip()
        if not session:
            raise ValueError("Session cannot be empty.")
        folder = self.sessions_root / session / self.workflow_folder_name
        if create_folder:
            folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{workflow_id}.json"

    def save(self, record):
        path = self.get_workflow_path(
            record["session"], record["workflow_id"], create_folder=True
        )
        temp = path.with_suffix(".json.tmp")
        with self._lock:
            with temp.open("w", encoding="utf-8") as file:
                json.dump(dict(record), file, ensure_ascii=False, indent=2)
            temp.replace(path)
        return path

    def load(self, session, workflow_id):
        path = self.get_workflow_path(session, workflow_id)
        if not path.exists():
            raise FileNotFoundError(f"Workflow record not found: {path}")
        with self._lock, path.open("r", encoding="utf-8") as file:
            return dict(json.load(file))

    def list_workflows(self, session):
        folder = self.sessions_root / str(session).strip() / self.workflow_folder_name
        if not folder.exists():
            return []
        records = []
        for path in sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                with path.open("r", encoding="utf-8") as file:
                    records.append(dict(json.load(file)))
            except Exception:
                continue
        return records
