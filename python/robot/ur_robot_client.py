from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import rtde_control
import rtde_receive


ROBOT_MODE_NAMES = {
    -1: "no_controller",
    0: "disconnected",
    1: "confirm_safety",
    2: "booting",
    3: "power_off",
    4: "power_on",
    5: "idle",
    6: "backdrive",
    7: "running",
    8: "updating_firmware",
}

# ur_rtde RuntimeState values.
RUNTIME_STATE_NAMES = {
    0: "stopping",
    1: "stopped",
    2: "playing",
    3: "pausing",
    4: "paused",
    5: "resuming",
}


@dataclass
class RobotSnapshot:
    connected: bool
    robot_mode: int | None
    robot_mode_name: str
    runtime_state: int | None
    runtime_state_name: str
    program_running: bool
    protective_stopped: bool
    emergency_stopped: bool
    safety_mode: int | None
    safety_status_bits: int | None
    actual_q: list[float]
    actual_tcp_pose: list[float]
    speed_scaling: float | None
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScriptExecutionResult:
    accepted: bool
    started: bool
    finished: bool
    message: str
    elapsed_seconds: float
    final_snapshot: RobotSnapshot | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)

        if self.final_snapshot is not None:
            result["final_snapshot"] = (
                self.final_snapshot.to_dict()
            )

        return result


class URRobotClient:
    """
    Initial UR10e client for:

    - receiving robot state;
    - checking basic safety conditions;
    - sending a complete URScript program;
    - observing program start and completion.

    All RTDEControlInterface calls are serialized because the
    ur_rtde control interface is not thread-safe.
    """

    def __init__(
        self,
        robot_ip: str,
        *,
        frequency: float = 10.0,
        verbose: bool = False,
    ) -> None:
        normalized_ip = str(
            robot_ip or ""
        ).strip()

        if not normalized_ip:
            raise ValueError(
                "robot_ip cannot be empty."
            )

        if frequency <= 0:
            raise ValueError(
                "frequency must be greater than zero."
            )

        self.robot_ip = normalized_ip
        self.frequency = float(frequency)
        self.verbose = bool(verbose)

        self._receive: (
            rtde_receive.RTDEReceiveInterface | None
        ) = None

        self._control: (
            rtde_control.RTDEControlInterface | None
        ) = None

        self._control_lock = threading.Lock()
        self._connection_lock = threading.Lock()

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect_receive(
        self,
    ) -> None:
        """
        Opens only the RTDE receive connection.

        Use this first to test observation without creating a
        control connection or sending any script.
        """

        with self._connection_lock:
            if (
                self._receive is not None
                and self._receive.isConnected()
            ):
                return

            self._receive = (
                rtde_receive.RTDEReceiveInterface(
                    self.robot_ip,
                    self.frequency,
                    [],
                    self.verbose,
                )
            )

            if not self._receive.isConnected():
                self._receive = None

                raise ConnectionError(
                    "Could not connect RTDE receive "
                    f"interface to {self.robot_ip}."
                )

    def connect_control(
        self,
    ) -> None:
        """
        Opens the RTDE control connection.

        This is deliberately separate from connect_receive(), so
        state monitoring can be tested before control is enabled.
        """

        with self._connection_lock:
            if (
                self._control is not None
                and self._control.isConnected()
            ):
                return

            self._control = (
                rtde_control.RTDEControlInterface(
                    self.robot_ip,
                    self.frequency,
                )
            )

            if not self._control.isConnected():
                self._control = None

                raise ConnectionError(
                    "Could not connect RTDE control "
                    f"interface to {self.robot_ip}."
                )

    def connect(
        self,
        *,
        include_control: bool = False,
    ) -> None:
        self.connect_receive()

        if include_control:
            self.connect_control()

    def disconnect(
        self,
    ) -> None:
        with self._connection_lock:
            if self._control is not None:
                try:
                    with self._control_lock:
                        self._control.disconnect()
                finally:
                    self._control = None

            if self._receive is not None:
                try:
                    self._receive.disconnect()
                finally:
                    self._receive = None

    def is_receive_connected(
        self,
    ) -> bool:
        return bool(
            self._receive is not None
            and self._receive.isConnected()
        )

    def is_control_connected(
        self,
    ) -> bool:
        return bool(
            self._control is not None
            and self._control.isConnected()
        )

    # ============================================================
    # STATE
    # ============================================================

    def get_snapshot(
        self,
    ) -> RobotSnapshot:
        receive = self._require_receive()

        robot_mode = int(
            receive.getRobotMode()
        )

        runtime_state = int(
            receive.getRuntimeState()
        )

        return RobotSnapshot(
            connected=bool(
                receive.isConnected()
            ),
            robot_mode=robot_mode,
            robot_mode_name=(
                ROBOT_MODE_NAMES.get(
                    robot_mode,
                    f"unknown_{robot_mode}",
                )
            ),
            runtime_state=runtime_state,
            runtime_state_name=(
                RUNTIME_STATE_NAMES.get(
                    runtime_state,
                    f"unknown_{runtime_state}",
                )
            ),
            program_running=bool(
                receive.getRobotStatus() & 0b10
            ),
            protective_stopped=bool(
                receive.isProtectiveStopped()
            ),
            emergency_stopped=bool(
                receive.isEmergencyStopped()
            ),
            safety_mode=int(
                receive.getSafetyMode()
            ),
            safety_status_bits=int(
                receive.getSafetyStatusBits()
            ),
            actual_q=[
                float(value)
                for value in receive.getActualQ()
            ],
            actual_tcp_pose=[
                float(value)
                for value
                in receive.getActualTCPPose()
            ],
            speed_scaling=float(
                receive.getSpeedScaling()
            ),
            timestamp=time.time(),
        )

    def validate_ready_for_script(
        self,
    ) -> tuple[bool, str, RobotSnapshot]:
        snapshot = self.get_snapshot()

        if not snapshot.connected:
            return (
                False,
                "RTDE receive connection is not active.",
                snapshot,
            )

        if snapshot.emergency_stopped:
            return (
                False,
                "Robot is emergency-stopped.",
                snapshot,
            )

        if snapshot.protective_stopped:
            return (
                False,
                "Robot is protective-stopped.",
                snapshot,
            )

        if snapshot.robot_mode != 7:
            return (
                False,
                "Robot mode must be RUNNING. "
                f"Current mode is "
                f"{snapshot.robot_mode_name}.",
                snapshot,
            )

        if snapshot.program_running:
            return (
                False,
                "A robot program is already running.",
                snapshot,
            )

        return (
            True,
            "Robot is ready to receive a script.",
            snapshot,
        )

    # ============================================================
    # SCRIPT
    # ============================================================

    def load_script(
        self,
        script_path: str | Path,
    ) -> str:
        path = Path(script_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"URScript file does not exist: {path}"
            )

        script = path.read_text(
            encoding="utf-8-sig"
        )

        script = script.replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        if not script.strip():
            raise ValueError(
                "URScript file is empty."
            )

        # sendCustomScript expects lines to be newline terminated.
        if not script.endswith("\n"):
            script += "\n"

        return script

    def send_script(
        self,
        script: str,
    ) -> bool:
        control = self._require_control()

        normalized_script = str(
            script or ""
        ).replace(
            "\r\n",
            "\n",
        ).replace(
            "\r",
            "\n",
        )

        if not normalized_script.strip():
            raise ValueError(
                "script cannot be empty."
            )

        if not normalized_script.endswith("\n"):
            normalized_script += "\n"

        with self._control_lock:
            return bool(
                control.sendCustomScript(
                    normalized_script
                )
            )

    def execute_script_file(
        self,
        script_path: str | Path,
        *,
        start_timeout_seconds: float = 5.0,
        execution_timeout_seconds: float = 600.0,
        poll_interval_seconds: float = 0.1,
    ) -> ScriptExecutionResult:
        if start_timeout_seconds <= 0:
            raise ValueError(
                "start_timeout_seconds must be positive."
            )

        if execution_timeout_seconds <= 0:
            raise ValueError(
                "execution_timeout_seconds must be positive."
            )

        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be positive."
            )

        script = self.load_script(
            script_path
        )

        ready, ready_message, snapshot = (
            self.validate_ready_for_script()
        )

        if not ready:
            return ScriptExecutionResult(
                accepted=False,
                started=False,
                finished=False,
                message=ready_message,
                elapsed_seconds=0.0,
                final_snapshot=snapshot,
            )

        started_at = time.monotonic()

        accepted = self.send_script(
            script
        )

        if not accepted:
            return ScriptExecutionResult(
                accepted=False,
                started=False,
                finished=False,
                message=(
                    "sendCustomScript() did not "
                    "accept the script."
                ),
                elapsed_seconds=(
                    time.monotonic() - started_at
                ),
                final_snapshot=self.get_snapshot(),
            )

        # --------------------------------------------------------
        # Wait for the controller to report program running.
        # --------------------------------------------------------

        start_deadline = (
            time.monotonic()
            + start_timeout_seconds
        )

        program_started = False

        while time.monotonic() < start_deadline:
            current = self.get_snapshot()

            self._raise_for_stop(
                current
            )

            if current.program_running:
                program_started = True
                break

            time.sleep(
                poll_interval_seconds
            )

        if not program_started:
            return ScriptExecutionResult(
                accepted=True,
                started=False,
                finished=False,
                message=(
                    "Script was accepted, but program "
                    "execution was not observed before "
                    "the start timeout."
                ),
                elapsed_seconds=(
                    time.monotonic() - started_at
                ),
                final_snapshot=self.get_snapshot(),
            )

        # --------------------------------------------------------
        # Wait for the running program to stop.
        # --------------------------------------------------------

        execution_deadline = (
            time.monotonic()
            + execution_timeout_seconds
        )

        while time.monotonic() < execution_deadline:
            current = self.get_snapshot()

            self._raise_for_stop(
                current
            )

            if not current.program_running:
                return ScriptExecutionResult(
                    accepted=True,
                    started=True,
                    finished=True,
                    message=(
                        "URScript execution finished."
                    ),
                    elapsed_seconds=(
                        time.monotonic()
                        - started_at
                    ),
                    final_snapshot=current,
                )

            time.sleep(
                poll_interval_seconds
            )

        return ScriptExecutionResult(
            accepted=True,
            started=True,
            finished=False,
            message=(
                "Robot program exceeded the execution "
                "timeout."
            ),
            elapsed_seconds=(
                time.monotonic() - started_at
            ),
            final_snapshot=self.get_snapshot(),
        )

    # ============================================================
    # INTERNAL
    # ============================================================

    def _raise_for_stop(
        self,
        snapshot: RobotSnapshot,
    ) -> None:
        if snapshot.emergency_stopped:
            raise RuntimeError(
                "Robot entered emergency stop."
            )

        if snapshot.protective_stopped:
            raise RuntimeError(
                "Robot entered protective stop."
            )

    def _require_receive(
        self,
    ) -> rtde_receive.RTDEReceiveInterface:
        if (
            self._receive is None
            or not self._receive.isConnected()
        ):
            raise RuntimeError(
                "RTDE receive interface is not connected."
            )

        return self._receive

    def _require_control(
        self,
    ) -> rtde_control.RTDEControlInterface:
        if (
            self._control is None
            or not self._control.isConnected()
        ):
            raise RuntimeError(
                "RTDE control interface is not connected."
            )

        return self._control

    def __enter__(
        self,
    ) -> "URRobotClient":
        self.connect(
            include_control=False
        )
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.disconnect()