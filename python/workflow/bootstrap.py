from .executor_registry import ExecutorRegistry
from .workflow_manager import WorkflowManager
from .workflow_store import WorkflowStore
from .executors import CaptureExecutor, TransformExecutor, ProcessingExecutor


def build_default_workflow_manager(sessions_root, definitions_root):
    registry = ExecutorRegistry()
    registry.register("capture", CaptureExecutor())
    registry.register("transform", TransformExecutor())
    registry.register("process", ProcessingExecutor())

    return WorkflowManager(
        store=WorkflowStore(sessions_root),
        registry=registry,
        definitions_root=definitions_root,
    )
