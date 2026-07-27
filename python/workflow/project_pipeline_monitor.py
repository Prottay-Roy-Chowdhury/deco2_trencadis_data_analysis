import threading
import traceback

from pathlib import Path
from typing import Any

from .project_pipeline_orchestrator import (
    ProjectPipelineOrchestrator,
)


class ProjectPipelineMonitor:
    """
    Periodically evaluates active project pipelines.

    A session is monitored only when it already contains:

        Project_Pipeline/
        └── project_pipeline_state.json

    This first version observes and advances durable state only.
    It does not yet send stage requests to Design, Motion, or Robot
    agents.
    """

    TERMINAL_STATUSES = {
        "pipeline_finished",
        "robot_finished",
        "failed",
        "cancelled",
    }

    def __init__(
        self,
        sessions_root,
        evaluation_interval_seconds: float = 1.0,
    ):
        self.sessions_root = Path(
            sessions_root
        ).resolve()

        self.evaluation_interval_seconds = (
            self._validate_interval(
                evaluation_interval_seconds
            )
        )

        self.orchestrator = (
            ProjectPipelineOrchestrator(
                sessions_root=self.sessions_root,
            )
        )

        self._lock = threading.RLock()

        self._stop_event = (
            threading.Event()
        )

        self._thread: (
            threading.Thread | None
        ) = None

        self._running = False

        self._last_results: dict[
            str,
            dict[str, Any],
        ] = {}

        self._last_errors: dict[
            str,
            str,
        ] = {}

    # ============================================================
    # LIFECYCLE
    # ============================================================

    def start(
        self,
    ) -> bool:
        """
        Starts the background evaluator.

        Returns True when a new monitor thread is started.
        Returns False when the monitor is already running.
        """

        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
            ):
                return False

            self._stop_event.clear()

            self._running = True

            self._thread = threading.Thread(
                target=self._run,
                name=(
                    "project-pipeline-monitor"
                ),
                daemon=True,
            )

            self._thread.start()

        return True

    def stop(
        self,
        join_timeout_seconds: float = 2.0,
    ) -> bool:
        """
        Requests the monitor to stop.

        Returns True when a running monitor was stopped.
        Returns False when it was already stopped.
        """

        with self._lock:
            thread = self._thread

            if (
                thread is None
                or not thread.is_alive()
            ):
                self._running = False
                return False

            self._stop_event.set()

        thread.join(
            timeout=max(
                0.0,
                float(
                    join_timeout_seconds
                ),
            )
        )

        with self._lock:
            self._running = False

            if not thread.is_alive():
                self._thread = None

        return True

    def is_running(
        self,
    ) -> bool:
        with self._lock:
            return bool(
                self._running
                and self._thread is not None
                and self._thread.is_alive()
            )

    # ============================================================
    # BACKGROUND LOOP
    # ============================================================

    def _run(
        self,
    ) -> None:
        print(
            "[project-pipeline-monitor] "
            "Started."
        )

        try:
            while not self._stop_event.is_set():
                try:
                    self.evaluate_all_sessions()
                except Exception:
                    print(
                        "[project-pipeline-monitor] "
                        "Evaluation cycle failed:\n"
                        f"{traceback.format_exc()}"
                    )

                self._stop_event.wait(
                    self.evaluation_interval_seconds
                )

        finally:
            with self._lock:
                self._running = False

            print(
                "[project-pipeline-monitor] "
                "Stopped."
            )

    # ============================================================
    # EVALUATION
    # ============================================================

    def evaluate_all_sessions(
        self,
    ) -> list[dict[str, Any]]:
        """
        Evaluates each active persisted project pipeline once.

        One orchestrator call performs at most one transition.
        """

        results: list[
            dict[str, Any]
        ] = []

        for session in self.list_monitored_sessions():
            try:
                result = (
                    self.evaluate_session(
                        session
                    )
                )

                results.append(
                    result
                )

            except Exception:
                error_text = (
                    traceback.format_exc()
                )

                with self._lock:
                    self._last_errors[
                        session
                    ] = error_text

                print(
                    "[project-pipeline-monitor] "
                    "Session evaluation failed: "
                    f"{session}\n"
                    f"{error_text}"
                )

        return results

    def evaluate_session(
        self,
        session: str,
    ) -> dict[str, Any]:
        """
        Evaluates one persisted project pipeline once.
        """

        state = (
            self.orchestrator
            .get_status(
                session
            )
        )

        current_status = str(
            state.get("status") or ""
        ).strip().lower()

        if current_status in self.TERMINAL_STATUSES:
            result = {
                "session": session,
                "transitioned": False,
                "transition": None,
                "message": (
                    "Project pipeline is already "
                    "terminal."
                ),
                "state_before": state,
                "state_after": state,
                "workflow": None,
                "design_output": None,
                "motion_output": None,
            }

            with self._lock:
                self._last_results[
                    session
                ] = result

                self._last_errors.pop(
                    session,
                    None,
                )

            return result

        result = (
            self.orchestrator
            .evaluate_session(
                session=session,
            )
        )

        with self._lock:
            self._last_results[
                session
            ] = result

            self._last_errors.pop(
                session,
                None,
            )

        if result.get(
            "transitioned",
            False,
        ):
            print(
                "[project-pipeline-monitor] "
                f"{session}: "
                f"{result.get('transition')}"
            )

        return result

    # ============================================================
    # SESSION DISCOVERY
    # ============================================================

    def list_monitored_sessions(
        self,
        include_terminal: bool = False,
    ) -> list[str]:
        """
        Returns sessions containing a persisted project-pipeline state.

        By default, terminal pipelines are excluded.
        """

        if not self.sessions_root.is_dir():
            return []

        sessions: list[str] = []

        for session_path in self.sessions_root.iterdir():
            if not session_path.is_dir():
                continue

            state_path = (
                session_path /
                "Project_Pipeline" /
                "project_pipeline_state.json"
            )

            if not state_path.is_file():
                continue

            if not include_terminal:
                try:
                    state = (
                        self.orchestrator
                        .get_status(
                            session_path.name
                        )
                    )

                    status = str(
                        state.get("status") or ""
                    ).strip().lower()

                    if status in self.TERMINAL_STATUSES:
                        continue

                except Exception:
                    # Keep it discoverable so the evaluation cycle
                    # can report the underlying error.
                    pass

            sessions.append(
                session_path.name
            )

        sessions.sort()

        return sessions

    # ============================================================
    # STATUS
    # ============================================================

    def get_monitor_status(
        self,
    ) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.is_running(),
                "evaluation_interval_seconds": (
                    self.evaluation_interval_seconds
                ),
                "active_sessions": (
                    self.list_monitored_sessions(
                        include_terminal=False
                    )
                ),
                "all_sessions": (
                    self.list_monitored_sessions(
                        include_terminal=True
                    )
                ),
                "last_results": dict(
                    self._last_results
                ),
                "last_errors": dict(
                    self._last_errors
                ),
            }

    def get_last_result(
        self,
        session: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            result = self._last_results.get(
                session
            )

            return (
                dict(result)
                if isinstance(
                    result,
                    dict,
                )
                else None
            )

    # ============================================================
    # VALIDATION
    # ============================================================

    def _validate_interval(
        self,
        value: float,
    ) -> float:
        try:
            interval = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "evaluation_interval_seconds "
                "must be numeric."
            ) from exc

        if interval < 0.1:
            raise ValueError(
                "evaluation_interval_seconds "
                "must be at least 0.1."
            )

        return interval