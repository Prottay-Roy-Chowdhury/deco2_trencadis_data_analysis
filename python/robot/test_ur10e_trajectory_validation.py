import math
import rtde_control
import rtde_receive


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.56.101"

HOME_Q_DEG = [
    135.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PICK_REFERENCE_Q_DEG = [
    90.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

PLACE_REFERENCE_Q_DEG = [
    180.00, -99.45, 126.61, -114.71, -90.00, 0.00
]

# Same example targets as previous successful test.
PICK_POSITION = [0.175, -0.518, 0.250]
PLACE_POSITION = [0.518, 0.175, 0.250]

PICK_ROTATION_DEG = 20.0
PLACE_ROTATION_DEG = -30.0

APPROACH_HEIGHT = 0.080

# Maximum sampling increments.
JOINT_SAMPLE_STEP_DEG = 2.0
LINEAR_SAMPLE_STEP_M = 0.005

# Detect unexpectedly large changes between IK samples.
MAX_IK_JUMP_DEG = 10.0

# FK reconstruction tolerances.
MAX_POSITION_ERROR_MM = 0.1
MAX_ORIENTATION_ERROR_DEG = 0.1

# Optional TCP workspace limits.
# These are NOT configured safety planes or collision boundaries.
# Leave disabled until actual limits are measured.
ENABLE_TCP_WORKSPACE_CHECK = False

TCP_WORKSPACE = {
    "x": (-1.0, 1.0),
    "y": (-1.0, 1.0),
    "z": (0.0, 1.5)
}

JOINT_NAMES = [
    "Base", "Shoulder", "Elbow",
    "Wrist 1", "Wrist 2", "Wrist 3"
]


# ============================================================
# MATHEMATICAL UTILITIES
# ============================================================

def radians(values):
    return [math.radians(v) for v in values]


def norm(values):
    return math.sqrt(sum(v * v for v in values))


def clamp(value, low, high):
    return max(low, min(high, value))


def matmul(A, B):
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]


def transpose(A):
    return [
        [A[j][i] for j in range(3)]
        for i in range(3)
    ]


def rotvec_to_matrix(v):
    theta = norm(v)

    if theta < 1e-12:
        return [
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ]

    x, y, z = [a / theta for a in v]
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
    Robust conversion, including rotations near 180 degrees.
    Uses the largest diagonal component to recover the axis.
    """
    trace = sum(R[i][i] for i in range(3))
    theta = math.acos(
        clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    )

    if theta < 1e-7:
        return [
            (R[2][1] - R[1][2]) / 2.0,
            (R[0][2] - R[2][0]) / 2.0,
            (R[1][0] - R[0][1]) / 2.0
        ]

    if math.pi - theta < 1e-5:
        diagonal = [
            max(0., (R[i][i] + 1.) / 2.)
            for i in range(3)
        ]

        i = max(range(3), key=lambda j: diagonal[j])
        axis = [0., 0., 0.]
        axis[i] = math.sqrt(diagonal[i])

        if axis[i] < 1e-8:
            raise ValueError("Cannot recover rotation axis")

        others = [j for j in range(3) if j != i]

        for j in others:
            axis[j] = (
                R[i][j] + R[j][i]
            ) / (4.0 * axis[i])

        magnitude = norm(axis)
        axis = [v / magnitude for v in axis]

        return [theta * v for v in axis]

    scale = theta / (2.0 * math.sin(theta))

    return [
        scale * (R[2][1] - R[1][2]),
        scale * (R[0][2] - R[2][0]),
        scale * (R[1][0] - R[0][1])
    ]


def rotation_difference_deg(pose_a, pose_b):
    A = rotvec_to_matrix(pose_a[3:])
    B = rotvec_to_matrix(pose_b[3:])

    R = matmul(transpose(A), B)
    trace = sum(R[i][i] for i in range(3))

    angle = math.acos(
        clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    )

    return math.degrees(angle)


def position_difference_mm(a, b):
    return 1000.0 * norm([
        a[i] - b[i] for i in range(3)
    ])


def joint_difference_deg(a, b):
    return [
        math.degrees(b[i] - a[i])
        for i in range(6)
    ]


def interpolate(a, b, t):
    return [
        a[i] + t * (b[i] - a[i])
        for i in range(len(a))
    ]


# ============================================================
# TARGET POSE GENERATION
# ============================================================

def rotate_local_z(pose, angle_deg):
    angle = math.radians(angle_deg)

    c = math.cos(angle)
    s = math.sin(angle)

    Rz = [
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.]
    ]

    R = matmul(
        rotvec_to_matrix(pose[3:]),
        Rz
    )

    return list(pose[:3]) + matrix_to_rotvec(R)


def make_target(reference_pose, position, angle_deg):
    rotated = rotate_local_z(
        reference_pose, angle_deg
    )
    return list(position) + rotated[3:]


def make_approach(target, height):
    pose = list(target)
    pose[2] += height
    return pose


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def check_workspace(pose):
    if not ENABLE_TCP_WORKSPACE_CHECK:
        return True

    for i, axis in enumerate(["x", "y", "z"]):
        low, high = TCP_WORKSPACE[axis]

        if not (low <= pose[i] <= high):
            return False

    return True


def validate_fk(rtde_c, q, target, tcp_offset):
    fk = list(
        rtde_c.getForwardKinematics(q, tcp_offset)
    )

    pos_error = position_difference_mm(fk, target)
    rot_error = rotation_difference_deg(fk, target)

    valid = (
        pos_error <= MAX_POSITION_ERROR_MM
        and rot_error <= MAX_ORIENTATION_ERROR_DEG
    )

    return valid, pos_error, rot_error


def solve_ik(rtde_c, pose, seed, tcp_offset):
    if not rtde_c.getInverseKinematicsHasSolution(
        pose, seed
    ):
        return None, "IK unavailable"

    q = list(
        rtde_c.getInverseKinematics(pose, seed)
    )

    if len(q) != 6 or not all(math.isfinite(v) for v in q):
        return None, "Invalid IK result"

    if not rtde_c.isJointsWithinSafetyLimits(q):
        return None, "Joint safety limits"

    valid, pe, re = validate_fk(
        rtde_c, q, pose, tcp_offset
    )

    if not valid:
        return None, (
            f"FK mismatch: {pe:.4f} mm, {re:.4f} deg"
        )

    if not check_workspace(pose):
        return None, "Outside TCP workspace"

    return q, None


# ============================================================
# MOVEJ PATH VALIDATION
# ============================================================

def validate_movej(rtde_c, name, q_start, q_end,
                   tcp_offset):

    differences = joint_difference_deg(
        q_start, q_end
    )

    max_travel = max(abs(v) for v in differences)

    steps = max(
        1,
        math.ceil(
            max_travel / JOINT_SAMPLE_STEP_DEG
        )
    )

    min_tcp_z = float("inf")
    max_tcp_z = float("-inf")

    for i in range(steps + 1):
        t = i / steps

        q = interpolate(q_start, q_end, t)

        if not rtde_c.isJointsWithinSafetyLimits(q):
            return {
                "name": name,
                "type": "moveJ",
                "passed": False,
                "reason": f"Joint limits at sample {i}/{steps}"
            }

        tcp = list(
            rtde_c.getForwardKinematics(q, tcp_offset)
        )

        if len(tcp) != 6 or not all(
            math.isfinite(v) for v in tcp
        ):
            return {
                "name": name,
                "type": "moveJ",
                "passed": False,
                "reason": f"Invalid FK at {i}/{steps}"
            }

        if not check_workspace(tcp):
            return {
                "name": name,
                "type": "moveJ",
                "passed": False,
                "reason": f"Workspace at {i}/{steps}"
            }

        min_tcp_z = min(min_tcp_z, tcp[2])
        max_tcp_z = max(max_tcp_z, tcp[2])

    return {
        "name": name,
        "type": "moveJ",
        "passed": True,
        "samples": steps + 1,
        "min_z": min_tcp_z,
        "max_z": max_tcp_z,
        "max_joint_travel": max_travel
    }


# ============================================================
# MOVEL PATH VALIDATION
# ============================================================

def validate_movel(rtde_c, name,
                   pose_start, pose_end,
                   q_start, tcp_offset):

    distance = norm([
        pose_end[i] - pose_start[i]
        for i in range(3)
    ])

    steps = max(
        1,
        math.ceil(
            distance / LINEAR_SAMPLE_STEP_M
        )
    )

    R_start = rotvec_to_matrix(pose_start[3:])
    R_end = rotvec_to_matrix(pose_end[3:])

    orientation_change = rotation_difference_deg(
        pose_start, pose_end
    )

    # This test is designed for fixed-orientation
    # vertical approach/retreat segments.
    if orientation_change > 0.001:
        return {
            "name": name,
            "type": "moveL",
            "passed": False,
            "reason": (
                "Changing orientation is not supported "
                "by this fixed-orientation test"
            )
        }

    q_previous = list(q_start)

    max_joint_step = 0.0
    min_tcp_z = float("inf")
    max_tcp_z = float("-inf")

    for i in range(steps + 1):
        t = i / steps

        xyz = interpolate(
            pose_start[:3],
            pose_end[:3],
            t
        )

        # Preserve the starting orientation.
        pose = xyz + list(pose_start[3:])

        if not check_workspace(pose):
            return {
                "name": name,
                "type": "moveL",
                "passed": False,
                "reason": f"Workspace at {i}/{steps}"
            }

        q, error = solve_ik(
            rtde_c, pose, q_previous, tcp_offset
        )

        if error is not None:
            return {
                "name": name,
                "type": "moveL",
                "passed": False,
                "reason": f"{error} at {i}/{steps}"
            }

        joint_step = max(
            abs(v)
            for v in joint_difference_deg(
                q_previous, q
            )
        )

        if joint_step > MAX_IK_JUMP_DEG:
            return {
                "name": name,
                "type": "moveL",
                "passed": False,
                "reason": (
                    f"IK jump {joint_step:.3f} deg "
                    f"at {i}/{steps}"
                )
            }

        max_joint_step = max(
            max_joint_step,
            joint_step
        )

        min_tcp_z = min(min_tcp_z, pose[2])
        max_tcp_z = max(max_tcp_z, pose[2])

        q_previous = q

    return {
        "name": name,
        "type": "moveL",
        "passed": True,
        "samples": steps + 1,
        "min_z": min_tcp_z,
        "max_z": max_tcp_z,
        "max_joint_step": max_joint_step,
        "q_end": q_previous
    }


# ============================================================
# REPORT
# ============================================================

def print_result(result):
    status = "PASS" if result["passed"] else "FAIL"

    print(
        f"\n{result['name']:<30s} "
        f"{result['type']:<6s} {status}"
    )

    if result["passed"]:
        print(
            f"  Samples: {result['samples']}"
        )
        print(
            f"  TCP Z range: "
            f"{result['min_z']:.4f} to "
            f"{result['max_z']:.4f} m"
        )

        if result["type"] == "moveL":
            print(
                f"  Max sampled IK joint step: "
                f"{result['max_joint_step']:.4f} deg"
            )

    else:
        print(f"  Reason: {result['reason']}")


# ============================================================
# MAIN
# ============================================================

def main():
    rtde_c = None
    rtde_r = None

    try:
        print("=" * 70)
        print("UR10e MOVEJ / MOVEL TRAJECTORY VALIDATION")
        print("NO ROBOT MOVEMENT")
        print("=" * 70)

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        tcp_offset = list(
            rtde_c.getTCPOffset()
        )

        print("\nActive TCP offset:", tcp_offset)

        home_q = radians(HOME_Q_DEG)
        pick_ref_q = radians(PICK_REFERENCE_Q_DEG)
        place_ref_q = radians(PLACE_REFERENCE_Q_DEG)

        # Reference FK
        pick_ref_pose = list(
            rtde_c.getForwardKinematics(
                pick_ref_q, tcp_offset
            )
        )

        place_ref_pose = list(
            rtde_c.getForwardKinematics(
                place_ref_q, tcp_offset
            )
        )

        # Tile-specific poses
        pick_pose = make_target(
            pick_ref_pose,
            PICK_POSITION,
            PICK_ROTATION_DEG
        )

        place_pose = make_target(
            place_ref_pose,
            PLACE_POSITION,
            PLACE_ROTATION_DEG
        )

        pre_pick_pose = make_approach(
            pick_pose, APPROACH_HEIGHT
        )

        pre_place_pose = make_approach(
            place_pose, APPROACH_HEIGHT
        )

        # Solve endpoint IK.
        pre_pick_q, err1 = solve_ik(
            rtde_c, pre_pick_pose,
            pick_ref_q, tcp_offset
        )

        pick_q = None
        err2 = "Pre-pick IK unavailable"

        if pre_pick_q is not None:
            pick_q, err2 = solve_ik(
                rtde_c, pick_pose,
                pre_pick_q, tcp_offset
            )

        pre_place_q, err3 = solve_ik(
            rtde_c, pre_place_pose,
            place_ref_q, tcp_offset
        )

        place_q = None
        err4 = "Pre-place IK unavailable"

        if pre_place_q is not None:
            place_q, err4 = solve_ik(
                rtde_c, place_pose,
                pre_place_q, tcp_offset
            )

        endpoints = {
            "PRE_PICK": (pre_pick_q, err1),
            "PICK": (pick_q, err2),
            "PRE_PLACE": (pre_place_q, err3),
            "PLACE": (place_q, err4)
        }

        print("\nENDPOINT IK")
        print("-" * 70)

        for name, (q, error) in endpoints.items():
            print(
                f"{name:<12s}: "
                f"{'PASS' if q is not None else 'FAIL'}"
                + (f" ({error})" if error else "")
            )

        if any(q is None for q, _ in endpoints.values()):
            print(
                "\nTrajectory test aborted: "
                "one or more endpoint IK solutions failed."
            )
            return

        # ----------------------------------------------------
        # Validate the eight intended motion segments.
        # ----------------------------------------------------

        results = []

        results.append(validate_movej(
            rtde_c, "HOME -> PRE_PICK",
            home_q, pre_pick_q, tcp_offset
        ))

        results.append(validate_movel(
            rtde_c, "PRE_PICK -> PICK",
            pre_pick_pose, pick_pose,
            pre_pick_q, tcp_offset
        ))

        results.append(validate_movel(
            rtde_c, "PICK -> PRE_PICK",
            pick_pose, pre_pick_pose,
            pick_q, tcp_offset
        ))

        results.append(validate_movej(
            rtde_c, "PRE_PICK -> HOME",
            pre_pick_q, home_q, tcp_offset
        ))

        results.append(validate_movej(
            rtde_c, "HOME -> PRE_PLACE",
            home_q, pre_place_q, tcp_offset
        ))

        results.append(validate_movel(
            rtde_c, "PRE_PLACE -> PLACE",
            pre_place_pose, place_pose,
            pre_place_q, tcp_offset
        ))

        results.append(validate_movel(
            rtde_c, "PLACE -> PRE_PLACE",
            place_pose, pre_place_pose,
            place_q, tcp_offset
        ))

        results.append(validate_movej(
            rtde_c, "PRE_PLACE -> HOME",
            pre_place_q, home_q, tcp_offset
        ))

        print("\n" + "=" * 70)
        print("TRAJECTORY VALIDATION RESULTS")
        print("=" * 70)

        for result in results:
            print_result(result)

        passed = sum(r["passed"] for r in results)

        print("\n" + "=" * 70)
        print(f"KINEMATIC SEGMENTS PASSED: {passed}/{len(results)}")
        print("=" * 70)

        if passed == len(results):
            print(
                "\nAll sampled kinematic checks passed."
            )
        else:
            print(
                "\nOne or more sampled checks failed."
            )

        print(
            "\nIMPORTANT: Collision clearance is NOT verified."
        )
        print(
            "This test does not authorize robot movement."
        )
        print(
            "No robot motion commands were sent."
        )

    except Exception as e:
        print("\nERROR:", str(e))

    finally:
        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception:
                pass

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