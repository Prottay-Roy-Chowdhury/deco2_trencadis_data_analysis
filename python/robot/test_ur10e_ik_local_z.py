import math
import rtde_control
import rtde_receive


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.56.101"  # Change to your UR10e IP

ROTATION_ANGLES_DEG = [0, 15, -15, 30, -30]

JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3"
]


# ============================================================
# ROTATION VECTOR / MATRIX UTILITIES
# ============================================================

def rotvec_to_matrix(rotvec):
    """
    Convert UR rotation vector [rx, ry, rz] to 3x3 matrix
    using Rodrigues' rotation formula.
    """
    theta = math.sqrt(sum(v * v for v in rotvec))

    if theta < 1e-12:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

    x, y, z = [v / theta for v in rotvec]

    c = math.cos(theta)
    s = math.sin(theta)
    t = 1.0 - c

    return [
        [
            t*x*x + c,
            t*x*y - s*z,
            t*x*z + s*y
        ],
        [
            t*x*y + s*z,
            t*y*y + c,
            t*y*z - s*x
        ],
        [
            t*x*z - s*y,
            t*y*z + s*x,
            t*z*z + c
        ]
    ]


def matrix_multiply(A, B):
    """Multiply two 3x3 matrices."""
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]


def matrix_to_rotvec(R):
    """
    Convert a 3x3 rotation matrix to a UR rotation vector.

    Suitable for the small-rotation test.
    Raises an error near 180 degrees.
    """
    trace = R[0][0] + R[1][1] + R[2][2]

    cos_theta = max(
        -1.0,
        min(1.0, (trace - 1.0) / 2.0)
    )

    theta = math.acos(cos_theta)

    if theta < 1e-10:
        return [0.0, 0.0, 0.0]

    if abs(math.pi - theta) < 1e-6:
        raise ValueError(
            "Rotation near 180 degrees: "
            "use a more robust conversion."
        )

    scale = theta / (2.0 * math.sin(theta))

    return [
        scale * (R[2][1] - R[1][2]),
        scale * (R[0][2] - R[2][0]),
        scale * (R[1][0] - R[0][1])
    ]


def rotate_tcp_local_z(pose, angle_deg):
    """
    Rotate TCP orientation around its LOCAL Z axis.

    R_target = R_current @ Rz(angle)

    TCP position remains unchanged.
    """
    angle = math.radians(angle_deg)

    R_current = rotvec_to_matrix(pose[3:6])

    c = math.cos(angle)
    s = math.sin(angle)

    Rz = [
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ]

    R_target = matrix_multiply(R_current, Rz)

    new_rotvec = matrix_to_rotvec(R_target)

    return list(pose[:3]) + new_rotvec


# ============================================================
# DISPLAY UTILITIES
# ============================================================

def print_joints(title, joints):
    print(f"\n{title}")
    print("-" * 62)

    for name, q in zip(JOINT_NAMES, joints):
        print(
            f"{name:12s}: "
            f"{q:9.5f} rad  "
            f"{math.degrees(q):9.2f} deg"
        )


def print_pose(title, pose):
    print(f"\n{title}")
    print("-" * 62)

    labels = ["X", "Y", "Z", "RX", "RY", "RZ"]

    for label, value in zip(labels, pose):
        print(f"{label:4s}: {value:12.6f}")


def joint_differences_deg(reference, solution):
    return [
        math.degrees(sol - ref)
        for ref, sol in zip(reference, solution)
    ]


def position_error_mm(pose_a, pose_b):
    """Euclidean TCP position error in millimeters."""
    return 1000.0 * math.sqrt(
        sum(
            (pose_a[i] - pose_b[i]) ** 2
            for i in range(3)
        )
    )


def orientation_error_deg(pose_a, pose_b):
    """
    Relative orientation error between two TCP poses.

    Uses rotation matrices rather than directly
    subtracting rotation vectors.
    """
    Ra = rotvec_to_matrix(pose_a[3:6])
    Rb = rotvec_to_matrix(pose_b[3:6])

    relative = [
        [
            sum(Ra[k][i] * Rb[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]

    trace = (
        relative[0][0] +
        relative[1][1] +
        relative[2][2]
    )

    cosine = max(
        -1.0,
        min(1.0, (trace - 1.0) / 2.0)
    )

    return math.degrees(math.acos(cosine))


# ============================================================
# IK TEST
# ============================================================

def test_ik(
    rtde_c,
    target_pose,
    actual_q,
    angle_deg,
    tcp_offset
):
    print("\n" + "=" * 62)
    print(f"LOCAL Z ROTATION TEST: {angle_deg:+.1f} deg")
    print("=" * 62)

    print_pose("Target TCP pose", target_pose)

    try:
        # ----------------------------------------------------
        # 1. Check whether IK solution exists
        # ----------------------------------------------------

        has_solution = (
            rtde_c.getInverseKinematicsHasSolution(
                target_pose,
                actual_q
            )
        )

        print("\nIK solution available:", has_solution)

        if not has_solution:
            return {
                "angle": angle_deg,
                "success": False
            }

        # ----------------------------------------------------
        # 2. Calculate IK using actual joints as seed
        # ----------------------------------------------------

        ik_q = list(
            rtde_c.getInverseKinematics(
                target_pose,
                actual_q
            )
        )

        print_joints(
            "Calculated IK joints",
            ik_q
        )

        differences = joint_differences_deg(
            actual_q,
            ik_q
        )

        print("\nJoint differences from actual:")
        print("-" * 62)

        for name, diff in zip(JOINT_NAMES, differences):
            print(f"{name:12s}: {diff:+10.4f} deg")

        # ----------------------------------------------------
        # 3. Check controller joint safety limits
        # ----------------------------------------------------

        safe = rtde_c.isJointsWithinSafetyLimits(
            ik_q
        )

        print("\nJoint safety check:", safe)

        # ----------------------------------------------------
        # 4. Verify IK with forward kinematics
        #
        # IMPORTANT:
        # Explicitly supply the active TCP offset.
        # This fixes the FK discrepancy found previously.
        # ----------------------------------------------------

        fk_pose = list(
            rtde_c.getForwardKinematics(
                ik_q,
                tcp_offset
            )
        )

        print_pose(
            "Forward kinematics TCP pose",
            fk_pose
        )

        pos_error = position_error_mm(
            target_pose,
            fk_pose
        )

        rot_error = orientation_error_deg(
            target_pose,
            fk_pose
        )

        print("\nForward kinematics verification:")
        print(f"Position error:    {pos_error:.6f} mm")
        print(f"Orientation error: {rot_error:.6f} deg")

        return {
            "angle": angle_deg,
            "success": True,
            "joints": ik_q,
            "differences": differences,
            "safe": safe,
            "pos_error": pos_error,
            "rot_error": rot_error
        }

    except Exception as e:
        print("\nIK ERROR:", str(e))

        return {
            "angle": angle_deg,
            "success": False
        }


# ============================================================
# MAIN
# ============================================================

def main():

    rtde_c = None
    rtde_r = None

    try:
        print("=" * 62)
        print("UR10e LOCAL TCP Z ROTATION IK TEST")
        print("CONTROLLER-BASED IK / NO ROBOT MOVEMENT")
        print("=" * 62)

        # ----------------------------------------------------
        # 1. Connect to robot
        # ----------------------------------------------------

        print("\nConnecting to UR10e...")

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        print("Connected successfully.")

        # ----------------------------------------------------
        # 2. Read actual robot state
        # ----------------------------------------------------

        actual_q = list(
            rtde_r.getActualQ()
        )

        actual_tcp = list(
            rtde_r.getActualTCPPose()
        )

        # Read active TCP offset from controller
        tcp_offset = list(
            rtde_c.getTCPOffset()
        )

        print_joints(
            "Actual joint configuration",
            actual_q
        )

        print_pose(
            "Actual TCP pose (robot base)",
            actual_tcp
        )

        print_pose(
            "Active TCP offset",
            tcp_offset
        )

        # ----------------------------------------------------
        # 3. Generate and test rotated TCP poses
        # ----------------------------------------------------

        results = []

        for angle in ROTATION_ANGLES_DEG:

            target_pose = rotate_tcp_local_z(
                actual_tcp,
                angle
            )

            result = test_ik(
                rtde_c,
                target_pose,
                actual_q,
                angle,
                tcp_offset
            )

            results.append(result)

        # ----------------------------------------------------
        # 4. Print summary
        # ----------------------------------------------------

        print("\n" + "=" * 90)
        print("IK TEST SUMMARY")
        print("=" * 90)

        print(
            f"{'Rotation':>10s} "
            f"{'IK':>8s} "
            f"{'J1 delta':>12s} "
            f"{'J6 delta':>12s} "
            f"{'Safe':>8s} "
            f"{'FK pos mm':>12s} "
            f"{'FK rot deg':>12s}"
        )

        print("-" * 90)

        for result in results:

            if not result["success"]:
                print(
                    f"{result['angle']:+10.1f} "
                    f"{'FAIL':>8s}"
                )
                continue

            print(
                f"{result['angle']:+10.1f} "
                f"{'OK':>8s} "
                f"{result['differences'][0]:+12.4f} "
                f"{result['differences'][5]:+12.4f} "
                f"{str(result['safe']):>8s} "
                f"{result['pos_error']:12.6f} "
                f"{result['rot_error']:12.6f}"
            )

        print("\n" + "=" * 62)
        print("TEST COMPLETED")
        print("No robot motion commands were sent.")
        print("=" * 62)

    except Exception as e:
        print("\nMAIN ERROR:", str(e))

    finally:

        # ----------------------------------------------------
        # 5. Clean shutdown
        # ----------------------------------------------------

        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception as e:
                print("stopScript warning:", e)

        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass

        if rtde_c is not None:
            try:
                rtde_c.disconnect()
            except Exception:
                pass

        print("\nDisconnected from UR10e.")


if __name__ == "__main__":
    main()