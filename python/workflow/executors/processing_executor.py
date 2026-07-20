from copy import deepcopy
from workflow.base_executor import BaseExecutor
from processing import run_processing_from_config


class ProcessingExecutor(BaseExecutor):
    executor_name = "process"

    def run(self, context, stage_config):
        config = deepcopy(dict(stage_config))
        config.setdefault("session", context["session"])
        result = run_processing_from_config(config)
        self.validate_outputs(context, config, result)
        return dict(result)
