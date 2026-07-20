from abc import ABC, abstractmethod


class BaseExecutor(ABC):
    executor_name = "base"

    def validate_inputs(self, context, stage_config):
        pass

    @abstractmethod
    def run(self, context, stage_config):
        raise NotImplementedError

    def validate_outputs(self, context, stage_config, result):
        if not isinstance(result, dict):
            raise RuntimeError(
                f"{self.executor_name} executor returned a non-dictionary result."
            )
        if result.get("status") != "ok":
            raise RuntimeError(
                str(result.get("message", f"{self.executor_name} stage failed."))
            )
