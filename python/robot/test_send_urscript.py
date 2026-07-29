from __future__ import annotations

import json
import sys
from pathlib import Path


PYTHON_ROOT = Path(
    __file__
).resolve().parents[1]

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PYTHON_ROOT),
    )


from robot.ur_robot_client import (
    URRobotClient,
)


ROBOT_IP = "192.168.56.101"

SCRIPT_PATH = Path(
    r"C:\\Users\\Usuario\\Downloads\\test02.script"
)


def main() -> None:
    client = URRobotClient(
        ROBOT_IP,
        frequency=10.0,
    )

    try:
        client.connect_receive()

        ready, message, snapshot = (
            client.validate_ready_for_script()
        )

        print(
            json.dumps(
                {
                    "ready": ready,
                    "message": message,
                    "snapshot": snapshot.to_dict(),
                },
                indent=2,
                default=str,
            )
        )

        if not ready:
            return

        # Create the control connection only after the state and
        # safety checks have passed.
        client.connect_control()

        result = client.execute_script_file(
            SCRIPT_PATH,
            start_timeout_seconds=5.0,
            execution_timeout_seconds=600.0,
            poll_interval_seconds=0.1,
        )

        print()
        print(
            json.dumps(
                result.to_dict(),
                indent=2,
                default=str,
            )
        )

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()