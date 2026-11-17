import math
import rtde_control
import rtde_receive

ROBOT_IP = "192.168.56.101"  # Change to your UR10e IP

JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3"
]


def print_joints(title, joints):
    print(f"\n{title}")
    print("-" * 48)

    for name, value in zip(JOINT_NAMES, joints):
        print(
            f"{name:12s}: "
            f"{value:9.5f} rad  "
            f"{math.degrees(value):9.2f} deg"
        )


def main():
    rtde_c = None
    rtde_r = None

    try:
        print("Connecting to UR10e...")

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        print("Connected.")

        # 1. Read current robot state
        actual_q = rtde_r.getActualQ()
        actual_tcp = rtde_r.getActualTCPPose()

        print_joints(
            "Actual robot joint configuration",
            actual_q
        )

        print("\nActual TCP pose (robot base frame):")
        print([round(v, 6) for v in actual_tcp])

        # 2. Ask controller for IK
        print("\nRequesting inverse kinematics...")

        has_solution = rtde_c.getInverseKinematicsHasSolution(
            actual_tcp,
            actual_q
        )

        print("IK solution available:", has_solution)

        if not has_solution:
            print("IK test failed: no solution.")
            return

        ik_q = rtde_c.getInverseKinematics(
            actual_tcp,
            actual_q
        )

        print_joints(
            "Calculated IK joint configuration",
            ik_q
        )

        # 3. Compare actual and calculated joints
        print("\nJoint differences:")
        print("-" * 48)

        for name, actual, calculated in zip(
            JOINT_NAMES, actual_q, ik_q
        ):
            difference = math.degrees(
                calculated - actual
            )

            print(
                f"{name:12s}: "
                f"{difference:+.6f} deg"
            )

        # 4. Check joint safety limits
        valid = rtde_c.isJointsWithinSafetyLimits(
            ik_q
        )

        print("\nJoint safety check:", valid)

        # 5. Inspect the controller's TCP configuration
        tcp_offset = rtde_c.getTCPOffset()

        print("\nActive TCP offset:")
        print([round(v, 6) for v in tcp_offset])

        print("\nIK TEST COMPLETED")
        print("No motion commands were sent.")

    except Exception as e:
        print("\nERROR:", str(e))

    finally:
        if rtde_c is not None:
            rtde_c.stopScript()

        if rtde_r is not None:
            rtde_r.disconnect()

        if rtde_c is not None:
            rtde_c.disconnect()


if __name__ == "__main__":
    main()