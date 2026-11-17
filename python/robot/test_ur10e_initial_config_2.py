import math
import rtde_control
import rtde_receive


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.56.101"  # Change to your robot IP

# Joint order:
# [Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3]

HOME_Q_DEG = [
    135.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PICK_REFERENCE_Q_DEG = [
    90.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PLACE_REFERENCE_Q_DEG = [
    180.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

# ------------------------------------------------------------
# TILE-SPECIFIC TARGETS
# ------------------------------------------------------------
# Position in robot-base coordinates, metres.
# Replace with your actual tile positions for future tests.

PICK_POSITION = [0.175, -0.518, 0.250]
PLACE_POSITION = [0.518, 0.175, 0.250]

# Rotation around the TCP's LOCAL Z axis, relative to
# the corresponding reference orientation.

PICK_ROTATION_DEG = 20.0
PLACE_ROTATION_DEG = -30.0

# Approach height measured along robot-base GLOBAL +Z.
APPROACH_HEIGHT = 0.080

JOINT_NAMES = [
    "Base", "Shoulder", "Elbow",
    "Wrist 1", "Wrist 2", "Wrist 3"
]


# ============================================================
# UTILITIES
# ============================================================

def deg_to_rad(values):
    return [math.radians(v) for v in values]


def rad_to_deg(values):
    return [math.degrees(v) for v in values]


def print_joints(title, q):
    print(f"\n{title}")
    print("-" * 65)

    for name, value in zip(JOINT_NAMES, q):
        print(
            f"{name:12s}: "
            f"{value:10.5f} rad "
            f"{math.degrees(value):9.2f} deg"
        )


def print_pose(title, pose):
    print(f"\n{title}")
    print("-" * 65)

    labels = ["X", "Y", "Z", "RX", "RY", "RZ"]

    for label, value in zip(labels, pose):
        print(f"{label:4s}: {value:12.6f}")


def joint_difference_deg(q_from, q_to):
    return [
        math.degrees(b - a)
        for a, b in zip(q_from, q_to)
    ]


def total_joint_travel_deg(q_from, q_to):
    return sum(
        abs(v)
        for v in joint_difference_deg(q_from, q_to)
    )


def position_error_mm(pose_a, pose_b):
    return 1000.0 * math.sqrt(
        sum(
            (pose_a[i] - pose_b[i]) ** 2
            for i in range(3)
        )
    )


# ============================================================
# ROTATION MATRIX UTILITIES
# ============================================================

def matrix_multiply(A, B):
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]


def matrix_transpose(A):
    return [
        [A[j][i] for j in range(3)]
        for i in range(3)
    ]


def rotvec_to_matrix(rotvec):

    theta = math.sqrt(
        sum(v * v for v in rotvec)
    )

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


def matrix_to_rotvec(R):
    """
    Convert rotation matrix to rotation vector.

    Handles near-zero and near-180-degree rotations.
    """

    trace = R[0][0] + R[1][1] + R[2][2]

    cosine = max(
        -1.0,
        min(1.0, (trace - 1.0) / 2.0)
    )

    theta = math.acos(cosine)

    if theta < 1e-8:
        return [0.0, 0.0, 0.0]

    # Near 180 degrees: extract axis using the
    # diagonal entries of R + I.

    if math.pi - theta < 1e-5:

        xx = max(0.0, (R[0][0] + 1.0) / 2.0)
        yy = max(0.0, (R[1][1] + 1.0) / 2.0)
        zz = max(0.0, (R[2][2] + 1.0) / 2.0)

        if xx >= yy and xx >= zz:
            x = math.sqrt(xx)
            y = (R[0][1] + R[1][0]) / (4.0 * x)
            z = (R[0][2] + R[2][0]) / (4.0 * x)

        elif yy >= zz:
            y = math.sqrt(yy)
            x = (R[0][1] + R[1][0]) / (4.0 * y)
            z = (R[1][2] + R[2][1]) / (4.0 * y)

        else:
            z = math.sqrt(zz)
            x = (R[0][2] + R[2][0]) / (4.0 * z)
            y = (R[1][2] + R[2][1]) / (4.0 * z)

        axis = [x, y, z]

        norm = math.sqrt(sum(v*v for v in axis))

        axis = [v / norm for v in axis]

        return [theta * v for v in axis]

    factor = theta / (2.0 * math.sin(theta))

    return [
        factor * (R[2][1] - R[1][2]),
        factor * (R[0][2] - R[2][0]),
        factor * (R[1][0] - R[0][1])
    ]


def orientation_error_deg(pose_a, pose_b):

    Ra = rotvec_to_matrix(pose_a[3:6])
    Rb = rotvec_to_matrix(pose_b[3:6])

    Rrel = matrix_multiply(
        matrix_transpose(Ra),
        Rb
    )

    trace = sum(Rrel[i][i] for i in range(3))

    cosine = max(
        -1.0,
        min(1.0, (trace - 1.0) / 2.0)
    )

    return math.degrees(math.acos(cosine))


# ============================================================
# TARGET POSE GENERATION
# ============================================================

def rotate_tcp_local_z(pose, angle_deg):
    """
    Rotate TCP orientation about its LOCAL Z axis.

    The TCP position remains unchanged.

    R_target = R_reference @ Rz(angle)
    """

    angle = math.radians(angle_deg)

    R_reference = rotvec_to_matrix(pose[3:6])

    c = math.cos(angle)
    s = math.sin(angle)

    Rz = [
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ]

    R_target = matrix_multiply(
        R_reference,
        Rz
    )

    target_rotvec = matrix_to_rotvec(R_target)

    return list(pose[:3]) + target_rotvec


def generate_target_pose(
    reference_pose,
    target_position,
    rotation_deg
):
    """
    Generate tile-specific TCP target.

    Translation:
        Actual tile position in robot-base coordinates.

    Orientation:
        Reference TCP orientation rotated about local Z.
    """

    rotated = rotate_tcp_local_z(
        reference_pose,
        rotation_deg
    )

    return list(target_position) + rotated[3:6]


def generate_approach_pose(target_pose, height):
    """
    Offset TCP position along robot-base GLOBAL +Z.

    Orientation remains unchanged.
    """

    approach_pose = list(target_pose)

    approach_pose[2] += height

    return approach_pose


# ============================================================
# IK VALIDATION
# ============================================================

def solve_and_validate_ik(
    rtde_c,
    name,
    target_pose,
    seed_q,
    tcp_offset
):

    print("\n" + "=" * 70)
    print(f"IK TEST: {name}")
    print("=" * 70)

    print_pose("Target TCP pose", target_pose)

    available = (
        rtde_c.getInverseKinematicsHasSolution(
            target_pose,
            seed_q
        )
    )

    print("\nIK solution available:", available)

    if not available:
        return {
            "name": name,
            "available": False,
            "safe": False,
            "q": None
        }

    q_solution = list(
        rtde_c.getInverseKinematics(
            target_pose,
            seed_q
        )
    )

    safe = rtde_c.isJointsWithinSafetyLimits(
        q_solution
    )

    print("\nJoint safety limits:", safe)

    print_joints(
        "Calculated IK joints",
        q_solution
    )

    differences = joint_difference_deg(
        seed_q,
        q_solution
    )

    print("\nJoint differences from IK seed")
    print("-" * 65)

    for name_joint, difference in zip(
        JOINT_NAMES,
        differences
    ):
        print(
            f"{name_joint:12s}: "
            f"{difference:+.4f} deg"
        )

    print(
        "\nTotal absolute joint difference: "
        f"{total_joint_travel_deg(seed_q, q_solution):.4f} deg"
    )

    # --------------------------------------------------------
    # FK reconstruction
    # --------------------------------------------------------

    reconstructed_pose = list(
        rtde_c.getForwardKinematics(
            q_solution,
            tcp_offset
        )
    )

    pos_error = position_error_mm(
        target_pose,
        reconstructed_pose
    )

    rot_error = orientation_error_deg(
        target_pose,
        reconstructed_pose
    )

    print("\nFK reconstruction")
    print("-" * 65)

    print(f"Position error:    {pos_error:.6f} mm")
    print(f"Orientation error: {rot_error:.6f} deg")

    return {
        "name": name,
        "available": True,
        "safe": safe,
        "q": q_solution,
        "position_error_mm": pos_error,
        "orientation_error_deg": rot_error
    }


# ============================================================
# JOINT TRANSITION COMPARISON
# ============================================================

def compare_joint_transition(name, q_from, q_to):

    print("\n" + "=" * 70)
    print(f"JOINT TRANSITION: {name}")
    print("=" * 70)

    differences = joint_difference_deg(
        q_from,
        q_to
    )

    for joint_name, difference in zip(
        JOINT_NAMES,
        differences
    ):
        print(
            f"{joint_name:12s}: "
            f"{difference:+.4f} deg"
        )

    print(
        "\nTotal absolute joint travel: "
        f"{total_joint_travel_deg(q_from, q_to):.4f} deg"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    rtde_c = None
    rtde_r = None

    try:

        print("=" * 70)
        print("UR10e TILE-SPECIFIC PICK AND PLACE IK TEST")
        print("NO ROBOT MOVEMENT")
        print("=" * 70)

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        print("\nConnected to UR10e.")

        tcp_offset = list(
            rtde_c.getTCPOffset()
        )

        print_pose(
            "Active TCP offset",
            tcp_offset
        )

        home_q = deg_to_rad(HOME_Q_DEG)

        pick_reference_q = deg_to_rad(
            PICK_REFERENCE_Q_DEG
        )

        place_reference_q = deg_to_rad(
            PLACE_REFERENCE_Q_DEG
        )

        # ----------------------------------------------------
        # 1. Reference TCP orientations from FK
        # ----------------------------------------------------

        pick_reference_pose = list(
            rtde_c.getForwardKinematics(
                pick_reference_q,
                tcp_offset
            )
        )

        place_reference_pose = list(
            rtde_c.getForwardKinematics(
                place_reference_q,
                tcp_offset
            )
        )

        print_pose(
            "Pick reference TCP pose",
            pick_reference_pose
        )

        print_pose(
            "Place reference TCP pose",
            place_reference_pose
        )

        # ----------------------------------------------------
        # 2. Generate actual tile target poses
        # ----------------------------------------------------

        pick_pose = generate_target_pose(
            pick_reference_pose,
            PICK_POSITION,
            PICK_ROTATION_DEG
        )

        place_pose = generate_target_pose(
            place_reference_pose,
            PLACE_POSITION,
            PLACE_ROTATION_DEG
        )

        pre_pick_pose = generate_approach_pose(
            pick_pose,
            APPROACH_HEIGHT
        )

        pre_place_pose = generate_approach_pose(
            place_pose,
            APPROACH_HEIGHT
        )

        # ----------------------------------------------------
        # 3. IK calculations
        # ----------------------------------------------------

        results = {}

        # results["PRE_PICK"] = solve_and_validate_ik(
        #     rtde_c,
        #     "PRE_PICK",
        #     pre_pick_pose,
        #     pick_reference_q,
        #     tcp_offset
        # )
        results["PRE_PICK"] = solve_and_validate_ik(
            rtde_c,
            "PRE_PICK",
            pre_pick_pose,
            home_q,
            tcp_offset
        )

        # Seed pick from pre-pick if available.
        # pick_seed = (
        #     results["PRE_PICK"]["q"]
        #     if results["PRE_PICK"]["available"]
        #     else pick_reference_q
        # )
        pick_seed = (
            results["PRE_PICK"]["q"]
            if results["PRE_PICK"]["available"]
            else home_q
        )

        results["PICK"] = solve_and_validate_ik(
            rtde_c,
            "PICK",
            pick_pose,
            pick_seed,
            tcp_offset
        )

        # results["PRE_PLACE"] = solve_and_validate_ik(
        #     rtde_c,
        #     "PRE_PLACE",
        #     pre_place_pose,
        #     place_reference_q,
        #     tcp_offset
        # )

        results["PRE_PLACE"] = solve_and_validate_ik(
            rtde_c,
            "PRE_PLACE",
            pre_place_pose,
            home_q,
            tcp_offset
        )

        # Seed place from pre-place if available.
        # place_seed = (
        #     results["PRE_PLACE"]["q"]
        #     if results["PRE_PLACE"]["available"]
        #     else place_reference_q
        # )

        place_seed = (
            results["PRE_PLACE"]["q"]
            if results["PRE_PLACE"]["available"]
            else home_q
        )

        results["PLACE"] = solve_and_validate_ik(
            rtde_c,
            "PLACE",
            place_pose,
            place_seed,
            tcp_offset
        )

        # ----------------------------------------------------
        # 4. Compare the joint-space transitions
        # ----------------------------------------------------

        if results["PRE_PICK"]["available"]:

            compare_joint_transition(
                "HOME -> PRE_PICK",
                home_q,
                results["PRE_PICK"]["q"]
            )

        if (
            results["PRE_PICK"]["available"]
            and results["PICK"]["available"]
        ):

            compare_joint_transition(
                "PRE_PICK -> PICK",
                results["PRE_PICK"]["q"],
                results["PICK"]["q"]
            )

        if results["PRE_PLACE"]["available"]:

            compare_joint_transition(
                "HOME -> PRE_PLACE",
                home_q,
                results["PRE_PLACE"]["q"]
            )

        if (
            results["PRE_PLACE"]["available"]
            and results["PLACE"]["available"]
        ):

            compare_joint_transition(
                "PRE_PLACE -> PLACE",
                results["PRE_PLACE"]["q"],
                results["PLACE"]["q"]
            )

        # ----------------------------------------------------
        # 5. Final summary
        # ----------------------------------------------------

        print("\n" + "=" * 85)
        print("FINAL IK SUMMARY")
        print("=" * 85)

        print(
            f"{'Target':<15s}"
            f"{'IK':>10s}"
            f"{'Safe':>10s}"
            f"{'Position err (mm)':>22s}"
            f"{'Orientation err (deg)':>25s}"
        )

        print("-" * 85)

        for result in results.values():

            pos_err = result.get("position_error_mm")
            rot_err = result.get("orientation_error_deg")

            pos_text = (
                f"{pos_err:.6f}"
                if pos_err is not None else "N/A"
            )

            rot_text = (
                f"{rot_err:.6f}"
                if rot_err is not None else "N/A"
            )

            print(
                f"{result['name']:<15s}"
                f"{str(result['available']):>10s}"
                f"{str(result['safe']):>10s}"
                f"{pos_text:>22s}"
                f"{rot_text:>25s}"
            )

        print("\nNo robot movement commands were sent.")

    except Exception as e:
        print("\nERROR:", str(e))

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