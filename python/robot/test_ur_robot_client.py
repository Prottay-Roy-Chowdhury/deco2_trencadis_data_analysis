from __future__ import annotations

import json
import sys
import time
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


def main() -> None:
    client = URRobotClient(
        ROBOT_IP,
        frequency=10.0,
    )

    try:
        # Receive connection only. No script can be sent.
        client.connect(
            include_control=False
        )

        print(
            "RTDE receive connected:",
            client.is_receive_connected(),
        )

        for sample_index in range(10):
            snapshot = client.get_snapshot()

            print()
            print(
                json.dumps(
                    snapshot.to_dict(),
                    indent=2,
                    default=str,
                )
            )

            time.sleep(0.5)

    finally:
        client.disconnect()


if __name__ == "__main__":
    main()