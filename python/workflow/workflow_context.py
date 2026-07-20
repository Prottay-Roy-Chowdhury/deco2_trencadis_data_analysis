from copy import deepcopy


class WorkflowContext:
    def __init__(self, session, workflow_id, runtime=None):
        self._data = {
            "workflow_id": str(workflow_id),
            "session": str(session),
            "runtime": deepcopy(dict(runtime or {})),
            "stage_results": {},
            "artifacts": {},
            "metadata": {},
        }

    @classmethod
    def from_dict(cls, data):
        context = cls(
            session=data.get("session", ""),
            workflow_id=data.get("workflow_id", ""),
            runtime=data.get("runtime", {}),
        )
        context._data = deepcopy(dict(data))
        context._data.setdefault("stage_results", {})
        context._data.setdefault("artifacts", {})
        context._data.setdefault("metadata", {})
        return context

    def to_dict(self):
        return deepcopy(self._data)

    def set_stage_result(self, stage_id, result):
        self._data["stage_results"][stage_id] = deepcopy(dict(result))

    def register_artifact(self, artifact_type, value):
        self._data["artifacts"][artifact_type] = deepcopy(value)
