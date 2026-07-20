import json
from pathlib import Path


class WorkflowDefinition:
    def __init__(self, data, source=None):
        self.data = dict(data)
        self.source = source
        self.validate()

    @classmethod
    def load(cls, path):
        path = Path(path)
        with path.open("r", encoding="utf-8") as file:
            return cls(json.load(file), source=path)

    @property
    def name(self):
        return str(self.data["name"])

    @property
    def version(self):
        return int(self.data.get("version", 1))

    @property
    def stages(self):
        return [dict(stage) for stage in self.data["stages"]]

    def validate(self):
        if not isinstance(self.data.get("name"), str) or not self.data["name"].strip():
            raise ValueError("Workflow definition requires a non-empty name.")
        stages = self.data.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError("Workflow definition requires stages.")
        ids = []
        for stage in stages:
            if not isinstance(stage, dict):
                raise ValueError("Every stage must be an object.")
            stage_id = stage.get("id")
            executor = stage.get("executor")
            if not isinstance(stage_id, str) or not stage_id.strip():
                raise ValueError("Every stage requires an id.")
            if not isinstance(executor, str) or not executor.strip():
                raise ValueError(f"Stage {stage_id} requires an executor.")
            if stage_id in ids:
                raise ValueError(f"Duplicate stage id: {stage_id}")
            ids.append(stage_id)
            if not isinstance(stage.get("depends_on", []), list):
                raise ValueError(f"Stage {stage_id} depends_on must be a list.")
            retries = stage.get("max_retries", 0)
            if not isinstance(retries, int) or retries < 0:
                raise ValueError(f"Stage {stage_id} has invalid max_retries.")
        known = set(ids)
        for stage in stages:
            for dependency in stage.get("depends_on", []):
                if dependency not in known:
                    raise ValueError(
                        f"Stage {stage['id']} depends on unknown stage {dependency}."
                    )

    def select_stages(self, selected_stage_ids=None, start_stage=None):
        stages = self.stages
        if selected_stage_ids is not None:
            lookup = {stage["id"]: stage for stage in stages}
            requested = [str(value) for value in selected_stage_ids]
            unknown = [value for value in requested if value not in lookup]
            if unknown:
                raise ValueError(f"Unknown requested stages: {unknown}")
            stages = [lookup[value] for value in requested]
        if start_stage is not None:
            positions = {stage["id"]: index for index, stage in enumerate(stages)}
            if start_stage not in positions:
                raise ValueError(f"Unknown start stage: {start_stage}")
            stages = stages[positions[start_stage]:]
        return stages
