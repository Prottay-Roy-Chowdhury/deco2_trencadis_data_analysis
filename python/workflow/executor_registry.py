class ExecutorRegistry:
    def __init__(self):
        self._executors = {}

    def register(self, name, executor, replace=False):
        name = str(name).strip().lower()
        if not name:
            raise ValueError("Executor name cannot be empty.")
        if name in self._executors and not replace:
            raise KeyError(f"Executor already registered: {name}")
        self._executors[name] = executor

    def get(self, name):
        name = str(name).strip().lower()
        if name not in self._executors:
            raise KeyError(
                f"Unknown executor {name}. Available: {sorted(self._executors)}"
            )
        return self._executors[name]

    def names(self):
        return tuple(sorted(self._executors))
