from copy import deepcopy
from workflow.base_executor import BaseExecutor
from capture import CaptureTexturedPointCloud


class CaptureExecutor(BaseExecutor):
    executor_name = "capture"

    def run(self, context, stage_config):
        config = deepcopy(dict(stage_config))
        config.setdefault("session", context["session"])
        capture = CaptureTexturedPointCloud()
        result = capture.run_capture_from_config(config)
        self.validate_outputs(context, config, result)
        return dict(result)
