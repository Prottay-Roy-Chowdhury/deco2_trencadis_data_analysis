
import math
import rtde_control
import rtde_receive


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.56.101"  # Change to your UR10e IP

HOME_Q_DEG = [
    135.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PICK_REFERENCE_Q_DEG = [
    90.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PLACE_REFERENCE_Q_DEG = [
    180.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3"
]

CONFIGURATIONS_DEG = {
    "HOME": HOME_Q_DEG,
    "PICK_REFERENCE": PICK_REFERENCE_Q_DEG,
    "PLACE_REFERENCE": PLACE_REFERENCE_Q_DEG
}


# ============================================================
# BASIC UTILITIES
# ============================================================

def deg_to_rad(values):
    return [math.radians(v) for v in values]


def rad_to_deg(values):
    return [math.degrees(v) for v in values]


def print_joints(title, joints_rad):
    print(f"\n{title}")
    print("-" * 65)

    for name, value in zip(JOINT_NAMES, joints_rad):
        print(
            f"{name:12s}: "
            f"{value:10.5f} rad  "
            f"{math.degrees(value):9.2f} deg"
        )


def print_pose(title, pose):
    print(f"\n{title}")
    print("-" * 65)

    labels = ["X", "Y", "Z", "RX", "RY", "RZ"]

    for label, value in zip(labels, pose):
        print(f"{label:4s}: {value:12.6f}")


def position_distance_mm(pose_a, pose_b):
    return 1000.0 * math.sqrt(
        sum(
            (pose_b[i] - pose_a[i]) ** 2
            for i in range(3)
        )
    )


def radial_distance(pose):
    return math.sqrt(
        pose[0] ** 2 + pose[1] ** 2
    )


def joint_difference_deg(q_from, q_to):
    return [
        math.degrees(b - a)
        for a, b in zip(q_from, q_to)
    ]


def total_joint_travel_deg(q_from, q_to):
    differences = joint_difference_deg(q_from, q_to)
    return sum(abs(v) for v in differences)


# ============================================================
# ROTATION UTILITIES
# ============================================================

def rotvec_to_matrix(rotvec):
    """
    Convert UR rotation vector to rotation matrix.
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
        [t*x*x+c, t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c]
    ]


def orientation_difference_deg(pose_a, pose_b):
    """
    Shortest angular difference between two TCP orientations.
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

    trace = sum(relative[i][i] for i in range(3))

    cosine = max(
        -1.0,
        min(1.0, (trace - 1.0) / 2.0)
    )

    return math.degrees(math.acos(cosine))


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration(
    rtde_c,
    name,
    joints_deg,
    tcp_offset
):
    print("\n" + "=" * 70)
    print(f"VALIDATING CONFIGURATION: {name}")
    print("=" * 70)

    joints_rad = deg_to_rad(joints_deg)

    print_joints(
        "Input joint configuration",
        joints_rad
    )

    # --------------------------------------------------------
    # 1. Joint safety limits
    # --------------------------------------------------------

    safe = rtde_c.isJointsWithinSafetyLimits(
        joints_rad
    )

    print("\nJoint safety limits:", safe)

    # --------------------------------------------------------
    # 2. Forward kinematics with active TCP offset
    # --------------------------------------------------------

    tcp_pose = list(
        rtde_c.getForwardKinematics(
            joints_rad,
            tcp_offset
        )
    )

    if len(tcp_pose) != 6:
        raise ValueError(
            f"Invalid FK result for {name}: {tcp_pose}"
        )

    print_pose(
        "Calculated TCP pose (robot base)",
        tcp_pose
    )

    # --------------------------------------------------------
    # 3. Spatial measurements
    # --------------------------------------------------------

    radius = radial_distance(tcp_pose)
    height = tcp_pose[2]

    print("\nTCP spatial information")
    print("-" * 65)

    print(f"Radial distance: {radius:.6f} m")
    print(f"TCP height:      {height:.6f} m")

    # --------------------------------------------------------
    # 4. FK / IK consistency check
    # --------------------------------------------------------

    ik_available = (
        rtde_c.getInverseKinematicsHasSolution(
            tcp_pose,
            joints_rad
        )
    )

    print("\nIK solution available:", ik_available)

    ik_q = None
    ik_joint_error = None
    fk_position_error = None
    fk_orientation_error = None

    if ik_available:

        ik_q = list(
            rtde_c.getInverseKinematics(
                tcp_pose,
                joints_rad
            )
        )

        print_joints(
            "IK reconstructed joints",
            ik_q
        )

        ik_joint_error = joint_difference_deg(
            joints_rad,
            ik_q
        )

        print("\nIK joint differences from input")
        print("-" * 65)

        for joint_name, error in zip(
            JOINT_NAMES,
            ik_joint_error
        ):
            print(
                f"{joint_name:12s}: {error:+.6f} deg"
            )

        # Verify reconstructed IK through FK
        reconstructed_tcp = list(
            rtde_c.getForwardKinematics(
                ik_q,
                tcp_offset
            )
        )

        fk_position_error = position_distance_mm(
            tcp_pose,
            reconstructed_tcp
        )

        fk_orientation_error = (
            orientation_difference_deg(
                tcp_pose,
                reconstructed_tcp
            )
        )

        print("\nFK / IK consistency")
        print("-" * 65)

        print(
            f"Position error:    "
            f"{fk_position_error:.6f} mm"
        )

        print(
            f"Orientation error: "
            f"{fk_orientation_error:.6f} deg"
        )

    else:
        print(
            "\nNo IK solution returned for this "
            "reference configuration."
        )

    return {
        "name": name,
        "joints_deg": joints_deg,
        "joints_rad": joints_rad,
        "safe": safe,
        "tcp_pose": tcp_pose,
        "radius": radius,
        "height": height,
        "ik_available": ik_available,
        "ik_q": ik_q,
        "ik_joint_error": ik_joint_error,
        "fk_position_error": fk_position_error,
        "fk_orientation_error": fk_orientation_error
    }


# ============================================================
# TRANSITION COMPARISON
# ============================================================

def compare_configurations(
    result_a,
    result_b
):
    name_a = result_a["name"]
    name_b = result_b["name"]

    q_a = result_a["joints_rad"]
    q_b = result_b["joints_rad"]

    pose_a = result_a["tcp_pose"]
    pose_b = result_b["tcp_pose"]

    print("\n" + "=" * 70)
    print(f"TRANSITION: {name_a} -> {name_b}")
    print("=" * 70)

    differences = joint_difference_deg(
        q_a,
        q_b
    )

    print("\nJoint differences")
    print("-" * 65)

    for name, diff in zip(
        JOINT_NAMES,
        differences
    ):
        print(f"{name:12s}: {diff:+10.4f} deg")

    total_travel = total_joint_travel_deg(
        q_a,
        q_b
    )

    maximum_joint_travel = max(
        abs(v) for v in differences
    )

    print("\nJoint travel")
    print("-" * 65)

    print(
        f"Total absolute joint travel: "
        f"{total_travel:.4f} deg"
    )

    print(
        f"Maximum individual joint travel: "
        f"{maximum_joint_travel:.4f} deg"
    )

    # --------------------------------------------------------
    # TCP translation in robot-base coordinates
    # --------------------------------------------------------

    dx = pose_b[0] - pose_a[0]
    dy = pose_b[1] - pose_a[1]
    dz = pose_b[2] - pose_a[2]

    tcp_distance = position_distance_mm(
        pose_a,
        pose_b
    )

    rotation_difference = (
        orientation_difference_deg(
            pose_a,
            pose_b
        )
    )

    print("\nTCP transformation comparison")
    print("-" * 65)

    print(f"Delta X: {dx:+.6f} m")
    print(f"Delta Y: {dy:+.6f} m")
    print(f"Delta Z: {dz:+.6f} m")

    print(
        f"TCP straight-line separation: "
        f"{tcp_distance:.3f} mm"
    )

    print(
        f"TCP orientation difference: "
        f"{rotation_difference:.4f} deg"
    )

    print(
        "\nNote: TCP separation is not "
        "the actual robot motion path."
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(results):

    print("\n" + "=" * 105)
    print("REFERENCE CONFIGURATION SUMMARY")
    print("=" * 105)

    print(
        f"{'Configuration':<20s}"
        f"{'Safe':>8s}"
        f"{'IK':>8s}"
        f"{'X (m)':>12s}"
        f"{'Y (m)':>12s}"
        f"{'Z (m)':>12s}"
        f"{'Radius (m)':>14s}"
        f"{'FK err (mm)':>14s}"
    )

    print("-" * 105)

    for result in results.values():

        pose = result["tcp_pose"]

        fk_error = result["fk_position_error"]

        fk_error_text = (
            f"{fk_error:.6f}"
            if fk_error is not None
            else "N/A"
        )

        print(
            f"{result['name']:<20s}"
            f"{str(result['safe']):>8s}"
            f"{str(result['ik_available']):>8s}"
            f"{pose[0]:12.5f}"
            f"{pose[1]:12.5f}"
            f"{pose[2]:12.5f}"
            f"{result['radius']:14.5f}"
            f"{fk_error_text:>14s}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rtde_c = None
    rtde_r = None

    try:
        print("=" * 70)
        print("UR10e REFERENCE CONFIGURATION FK / IK TEST")
        print("NO ROBOT MOVEMENT")
        print("=" * 70)

        # ----------------------------------------------------
        # 1. Connect to UR10e
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
        # 2. Read active robot state
        # ----------------------------------------------------

        actual_q = list(
            rtde_r.getActualQ()
        )

        actual_tcp = list(
            rtde_r.getActualTCPPose()
        )

        tcp_offset = list(
            rtde_c.getTCPOffset()
        )

        print_joints(
            "Actual robot joints",
            actual_q
        )

        print_pose(
            "Actual TCP pose",
            actual_tcp
        )

        print_pose(
            "Active TCP offset",
            tcp_offset
        )

        # ----------------------------------------------------
        # 3. Validate reference configurations
        # ----------------------------------------------------

        results = {}

        for name, joints_deg in CONFIGURATIONS_DEG.items():

            results[name] = validate_configuration(
                rtde_c,
                name,
                joints_deg,
                tcp_offset
            )

        # ----------------------------------------------------
        # 4. Compare reference configurations
        # ----------------------------------------------------

        compare_configurations(
            results["HOME"],
            results["PICK_REFERENCE"]
        )

        compare_configurations(
            results["HOME"],
            results["PLACE_REFERENCE"]
        )

        compare_configurations(
            results["PICK_REFERENCE"],
            results["PLACE_REFERENCE"]
        )

        # ----------------------------------------------------
        # 5. Summary
        # ----------------------------------------------------

        print_summary(results)

        print("\n" + "=" * 70)
        print("TEST COMPLETED")
        print("No robot motion commands were sent.")
        print("=" * 70)

    except Exception as e:
        print("\nMAIN ERROR:", str(e))

    finally:

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