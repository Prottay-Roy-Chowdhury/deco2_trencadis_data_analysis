import threading
import traceback
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .workflow_context import WorkflowContext
from .workflow_definition import WorkflowDefinition


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkflowCancelled(Exception):
    pass


class WorkflowManager:
    def __init__(self, store, registry, definitions_root):
        self.store = store
        self.registry = registry
        self.definitions_root = Path(definitions_root)
        self._lock = threading.RLock()
        self._records = {}
        self._threads = {}
        self._cancel_events = {}

    def load_definition(self, workflow_name):
        path = self.definitions_root / f"{workflow_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Workflow definition not found: {path}")
        return WorkflowDefinition.load(path)

    def submit_workflow(
        self,
        workflow_name,
        session,
        stage_configs=None,
        selected_stages=None,
        start_stage=None,
        runtime=None,
    ):
        definition = self.load_definition(workflow_name)
        stages = definition.select_stages(selected_stages, start_stage)
        workflow_id = self._create_workflow_id(definition.name)
        context = WorkflowContext(session, workflow_id, runtime)

        record = {
            "workflow_id": workflow_id,
            "workflow_name": definition.name,
            "workflow_version": definition.version,
            "session": str(session),
            "status": "queued",
            "progress": 0,
            "message": "Workflow submitted.",
            "current_stage": None,
            "current_stage_index": -1,
            "stage_count": len(stages),
            "stages": deepcopy(stages),
            "stage_configs": deepcopy(dict(stage_configs or {})),
            "stage_states": {
                stage["id"]: {
                    "status": "pending",
                    "attempt": 0,
                    "max_retries": int(stage.get("max_retries", 0)),
                    "started_at": None,
                    "finished_at": None,
                    "message": "",
                    "result": None,
                    "error": None,
                }
                for stage in stages
            },
            "context": context.to_dict(),
            "error": None,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "finished_at": None,
        }

        with self._lock:
            self._records[workflow_id] = record
            self._cancel_events[workflow_id] = threading.Event()
            self.store.save(record)
            thread = threading.Thread(
                target=self._run_workflow,
                args=(workflow_id,),
                daemon=True,
            )
            self._threads[workflow_id] = thread
            thread.start()

        return self.get_workflow_status(workflow_id, session)

    def get_workflow_status(self, workflow_id, session=None):
        with self._lock:
            if workflow_id in self._records:
                return deepcopy(self._records[workflow_id])
        if session is None:
            raise KeyError("Session is required for a workflow not held in memory.")
        record = self.store.load(session, workflow_id)
        with self._lock:
            self._records[workflow_id] = record
        return deepcopy(record)

    def list_workflows(self, session):
        return self.store.list_workflows(session)

    def cancel_workflow(self, workflow_id, session=None):
        with self._lock:
            event = self._cancel_events.get(workflow_id)
            if event is None:
                return self.get_workflow_status(workflow_id, session)
            event.set()
            self._update_record(
                workflow_id,
                status="cancelling",
                message="Cancellation requested.",
            )
        return self.get_workflow_status(workflow_id, session)

    def _run_workflow(self, workflow_id):
        try:
            self._update_record(
                workflow_id,
                status="running",
                message="Workflow started.",
                error=None,
            )
            record = self._get_record(workflow_id)
            context = WorkflowContext.from_dict(record["context"])
            stages = record["stages"]

            for stage_index, stage in enumerate(stages):
                self._raise_if_cancelled(workflow_id)
                self._ensure_dependencies_finished(workflow_id, stage)

                stage_id = stage["id"]
                executor = self.registry.get(stage["executor"])
                config = deepcopy(record.get("stage_configs", {}).get(stage_id, {}))
                attempts = int(stage.get("max_retries", 0)) + 1

                for attempt in range(1, attempts + 1):
                    self._raise_if_cancelled(workflow_id)
                    self._set_stage_state(
                        workflow_id,
                        stage_id,
                        {
                            "status": "running",
                            "attempt": attempt,
                            "started_at": utc_now(),
                            "finished_at": None,
                            "message": f"Running {stage_id}, attempt {attempt}/{attempts}.",
                            "result": None,
                            "error": None,
                        },
                    )
                    self._update_record(
                        workflow_id,
                        current_stage=stage_id,
                        current_stage_index=stage_index,
                        progress=int(100 * stage_index / max(len(stages), 1)),
                        message=f"Running stage: {stage_id}",
                    )

                    try:
                        executor.validate_inputs(context.to_dict(), config)
                        result = executor.run(context.to_dict(), config)
                        executor.validate_outputs(context.to_dict(), config, result)
                        context.set_stage_result(stage_id, result)
                        self._register_result_artifacts(context, result)
                        self._set_stage_state(
                            workflow_id,
                            stage_id,
                            {
                                "status": "finished",
                                "finished_at": utc_now(),
                                "message": f"Stage {stage_id} finished.",
                                "result": result,
                                "error": None,
                            },
                        )
                        self._update_record(
                            workflow_id,
                            context=context.to_dict(),
                            progress=int(100 * (stage_index + 1) / max(len(stages), 1)),
                            message=f"Stage finished: {stage_id}",
                        )
                        break
                    except Exception:
                        error = traceback.format_exc()
                        self._set_stage_state(
                            workflow_id,
                            stage_id,
                            {
                                "status": "retrying" if attempt < attempts else "failed",
                                "finished_at": utc_now(),
                                "message": f"Stage {stage_id} failed.",
                                "error": error,
                            },
                        )
                        if attempt >= attempts:
                            raise

            self._update_record(
                workflow_id,
                status="finished",
                progress=100,
                current_stage=None,
                current_stage_index=len(stages),
                message="Workflow finished successfully.",
                finished_at=utc_now(),
                error=None,
                context=context.to_dict(),
            )

        except WorkflowCancelled:
            self._update_record(
                workflow_id,
                status="cancelled",
                message="Workflow cancelled.",
                finished_at=utc_now(),
            )
        except Exception:
            self._update_record(
                workflow_id,
                status="failed",
                message="Workflow failed.",
                error=traceback.format_exc(),
                finished_at=utc_now(),
            )

    def _register_result_artifacts(self, context, result):
        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict):
            for key, value in artifacts.items():
                context.register_artifact(key, value)
        for key, value in result.items():
            if key.endswith("_path") and value:
                context.register_artifact(key, value)

    def _ensure_dependencies_finished(self, workflow_id, stage):
        record = self._get_record(workflow_id)
        for dependency in stage.get("depends_on", []):
            state = record["stage_states"].get(dependency, {})
            if state.get("status") != "finished":
                raise RuntimeError(
                    f"Stage {stage['id']} cannot run because {dependency} is not finished."
                )

    def _set_stage_state(self, workflow_id, stage_id, values):
        with self._lock:
            record = self._records[workflow_id]
            record["stage_states"][stage_id].update(deepcopy(dict(values)))
            record["updated_at"] = utc_now()
            self.store.save(record)

    def _update_record(self, workflow_id, **values):
        with self._lock:
            record = self._records[workflow_id]
            record.update(deepcopy(values))
            record["updated_at"] = utc_now()
            self.store.save(record)

    def _get_record(self, workflow_id):
        with self._lock:
            return deepcopy(self._records[workflow_id])

    def _raise_if_cancelled(self, workflow_id):
        with self._lock:
            event = self._cancel_events.get(workflow_id)
        if event is not None and event.is_set():
            raise WorkflowCancelled

    def _create_workflow_id(self, workflow_name):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"wf-{workflow_name}-{timestamp}-{uuid.uuid4().hex[:8]}"
