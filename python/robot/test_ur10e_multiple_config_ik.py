
import math
import csv
import random

import rtde_control
from scipy.spatial.transform import Rotation as R


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

# Target TCP positions in robot base frame (metres).
# Same positions as the previous successful test.
PICK_POSITION = [0.175, -0.518, 0.250]
PLACE_POSITION = [0.518, 0.175, 0.250]

# Rotation about the reference TCP's local Z axis.
PICK_ROTATION_DEG = 20.0
PLACE_ROTATION_DEG = -30.0


# ============================================================
# IK DISCOVERY SETTINGS
# ============================================================

RANDOM_SEED_COUNT = 300
RANDOM_SEED = 42

# Exploration ranges only.
# These are NOT the controller's joint safety limits.
JOINT_SEED_RANGES_DEG = [
    (-180.0, 180.0),  # J1
    (-180.0, 0.0),    # J2
    (-180.0, 180.0),  # J3
    (-180.0, 180.0),  # J4
    (-180.0, 180.0),  # J5
    (-180.0, 180.0),  # J6
]

# Joint-vector duplicate tolerance.
DUPLICATE_TOLERANCE_DEG = 0.05

# Geometric grouping tolerance, modulo 360 degrees.
GEOMETRIC_TOLERANCE_DEG = 0.05

# FK verification tolerances.
MAX_POSITION_ERROR_MM = 0.1
MAX_ORIENTATION_ERROR_DEG = 0.1

SAVE_CSV = True


# ============================================================
# BASIC UTILITIES
# ============================================================

def to_rad(values):
    return [math.radians(v) for v in values]


def to_deg(values):
    return [math.degrees(v) for v in values]


def vector_norm(values):
    return math.sqrt(sum(v * v for v in values))


def position_error_mm(pose_a, pose_b):
    return 1000.0 * vector_norm([
        pose_a[i] - pose_b[i]
        for i in range(3)
    ])


def rotation_error_deg(pose_a, pose_b):
    """
    Relative orientation difference between two TCP poses.
    Robust near 180-degree rotations.
    """
    rotation_a = R.from_rotvec(pose_a[3:6])
    rotation_b = R.from_rotvec(pose_b[3:6])

    relative = rotation_a.inv() * rotation_b

    return math.degrees(relative.magnitude())


# ============================================================
# TARGET TCP POSE GENERATION
# ============================================================

def rotate_local_z(pose, angle_deg):
    """
    Rotate a TCP orientation around its LOCAL Z axis.

    R_target = R_reference @ Rz(angle)

    Position is unchanged.
    """

    reference_rotation = R.from_rotvec(
        pose[3:6]
    )

    local_z_rotation = R.from_euler(
        "z",
        angle_deg,
        degrees=True
    )

    target_rotation = (
        reference_rotation * local_z_rotation
    )

    return (
        list(pose[:3])
        + target_rotation.as_rotvec().tolist()
    )


def make_target(reference_pose, position, angle_deg):
    """
    Generate a target TCP pose using:
      - specified robot-base position
      - reference TCP orientation
      - additional local-Z rotation
    """

    rotated_pose = rotate_local_z(
        reference_pose,
        angle_deg
    )

    return (
        list(position)
        + rotated_pose[3:6]
    )


# ============================================================
# JOINT CONFIGURATION COMPARISON
# ============================================================

def same_joint_configuration(q_a, q_b):
    """
    Compare actual joint angles.

    A 360-degree difference is NOT treated as identical.
    """

    return all(
        abs(math.degrees(a - b))
        < DUPLICATE_TOLERANCE_DEG
        for a, b in zip(q_a, q_b)
    )


def wrapped_difference_deg(a, b):
    """
    Shortest angular difference modulo 360 degrees.
    """

    difference = math.degrees(a - b)

    return (
        (difference + 180.0) % 360.0
    ) - 180.0


def same_geometric_configuration(q_a, q_b):
    """
    Treat joint angles differing by full revolutions
    as geometrically equivalent.
    """

    return all(
        abs(wrapped_difference_deg(a, b))
        < GEOMETRIC_TOLERANCE_DEG
        for a, b in zip(q_a, q_b)
    )


def joint_distance_deg(q_a, q_b):
    """
    Sum of absolute joint differences in degrees.

    This is a simple reference-distance metric,
    not a prediction of actual motion time.
    """

    return sum(
        abs(math.degrees(a - b))
        for a, b in zip(q_a, q_b)
    )


def max_joint_difference_deg(q_a, q_b):
    return max(
        abs(math.degrees(a - b))
        for a, b in zip(q_a, q_b)
    )


# ============================================================
# IK SEED GENERATION
# ============================================================

def generate_seeds(reference_q_deg):

    rng = random.Random(RANDOM_SEED)

    seeds_deg = []

    # Original station reference.
    seeds_deg.append(list(reference_q_deg))

    # Deterministic variations.
    variations = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, -180, 0, 0, 0],
        [0, 0, 180, 0, 0, 0],
        [0, 0, 0, 0, 180, 0],
        [0, 0, 0, 0, -180, 0],
        [180, 0, 0, 0, 0, 0],
        [-180, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 360],
        [0, 0, 0, 0, 0, -360],
    ]

    for variation in variations:
        seeds_deg.append([
            reference_q_deg[i] + variation[i]
            for i in range(6)
        ])

    # Random exploration.
    for _ in range(RANDOM_SEED_COUNT):

        seed = [
            rng.uniform(low, high)
            for low, high in JOINT_SEED_RANGES_DEG
        ]

        seeds_deg.append(seed)

    return [
        to_rad(seed)
        for seed in seeds_deg
    ]


# ============================================================
# IK SOLUTION DISCOVERY
# ============================================================

def discover_ik_solutions(
    rtde_c,
    name,
    target_pose,
    reference_q,
    tcp_offset
):

    seeds = generate_seeds(
        to_deg(reference_q)
    )

    unique_solutions = []

    attempts = 0
    successful_queries = 0
    invalid_results = 0
    no_solution_count = 0
    query_errors = 0

    print("\n" + "=" * 95)
    print(f"DISCOVERING IK SOLUTIONS: {name}")
    print("=" * 95)

    print(
        "Target TCP pose:",
        [round(v, 6) for v in target_pose]
    )

    print("Number of seeds:", len(seeds))

    # --------------------------------------------------------
    # QUERY CONTROLLER IK
    # --------------------------------------------------------

    for seed in seeds:

        attempts += 1

        try:

            has_solution = (
                rtde_c.getInverseKinematicsHasSolution(
                    target_pose,
                    seed
                )
            )

            if not has_solution:
                no_solution_count += 1
                continue

            q = list(
                rtde_c.getInverseKinematics(
                    target_pose,
                    seed
                )
            )

            if len(q) != 6:
                invalid_results += 1
                continue

            if not all(math.isfinite(v) for v in q):
                invalid_results += 1
                continue

            successful_queries += 1

            # ------------------------------------------------
            # FK VALIDATION
            # ------------------------------------------------

            fk_pose = list(
                rtde_c.getForwardKinematics(
                    q,
                    tcp_offset
                )
            )

            pos_err = position_error_mm(
                target_pose,
                fk_pose
            )

            rot_err = rotation_error_deg(
                target_pose,
                fk_pose
            )

            if (
                pos_err > MAX_POSITION_ERROR_MM
                or rot_err > MAX_ORIENTATION_ERROR_DEG
            ):
                invalid_results += 1
                continue

            # ------------------------------------------------
            # JOINT SAFETY
            # ------------------------------------------------

            safe = bool(
                rtde_c.isJointsWithinSafetyLimits(q)
            )

            # ------------------------------------------------
            # REMOVE DUPLICATE JOINT VECTORS
            # ------------------------------------------------

            duplicate = any(
                same_joint_configuration(
                    q,
                    item["q"]
                )
                for item in unique_solutions
            )

            if duplicate:
                continue

            unique_solutions.append({
                "q": q,
                "safe": safe,
                "pos_err": pos_err,
                "rot_err": rot_err,
                "reference_distance": joint_distance_deg(
                    reference_q,
                    q
                ),
                "max_joint_difference": max_joint_difference_deg(
                    reference_q,
                    q
                ),
            })

        except Exception as exc:

            query_errors += 1

            print(
                f"Seed {attempts} query error: {exc}"
            )

    # --------------------------------------------------------
    # GROUP GEOMETRICALLY EQUIVALENT SOLUTIONS
    # --------------------------------------------------------

    geometric_groups = []

    for solution in unique_solutions:

        matching_group = None

        for group in geometric_groups:

            if same_geometric_configuration(
                solution["q"],
                group[0]["q"]
            ):
                matching_group = group
                break

        if matching_group is None:
            geometric_groups.append(
                [solution]
            )
        else:
            matching_group.append(
                solution
            )

    # --------------------------------------------------------
    # ASSIGN GEOMETRIC BRANCH IDs
    # --------------------------------------------------------

    for branch_id, group in enumerate(
        geometric_groups,
        start=1
    ):

        for solution in group:
            solution["branch"] = branch_id

    # --------------------------------------------------------
    # SORT BY REFERENCE DISTANCE
    # --------------------------------------------------------

    unique_solutions.sort(
        key=lambda item: (
            not item["safe"],
            item["reference_distance"]
        )
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\nDiscovery summary")
    print("-" * 95)

    print("Seeds tested:", attempts)
    print("Successful IK queries:", successful_queries)
    print("No-solution queries:", no_solution_count)
    print("Invalid FK/results:", invalid_results)
    print("Query errors:", query_errors)

    print(
        "Distinct joint vectors:",
        len(unique_solutions)
    )

    print(
        "Distinct geometric groups:",
        len(geometric_groups)
    )

    print(
        "Safety-valid joint vectors:",
        sum(
            solution["safe"]
            for solution in unique_solutions
        )
    )

    # --------------------------------------------------------
    # PRINT ALL DISCOVERED JOINT CONFIGURATIONS
    # --------------------------------------------------------

    print("\nDiscovered configurations (degrees)")
    print("-" * 135)

    header = (
        f"{'ID':>3s} "
        f"{'Branch':>7s} "
        f"{'Safe':>6s} "
        f"{'J1':>9s} "
        f"{'J2':>9s} "
        f"{'J3':>9s} "
        f"{'J4':>9s} "
        f"{'J5':>9s} "
        f"{'J6':>9s} "
        f"{'RefDist':>10s} "
        f"{'MaxDiff':>9s}"
    )

    print(header)
    print("-" * 135)

    for index, solution in enumerate(
        unique_solutions,
        start=1
    ):

        q_deg = to_deg(
            solution["q"]
        )

        print(
            f"{index:3d} "
            f"{solution['branch']:7d} "
            f"{str(solution['safe']):>6s} "
            + " ".join(
                f"{v:9.2f}"
                for v in q_deg
            )
            + f" {solution['reference_distance']:10.2f}"
            + f" {solution['max_joint_difference']:9.2f}"
        )

    # --------------------------------------------------------
    # PRINT ONE PREFERRED SOLUTION PER GEOMETRIC GROUP
    # --------------------------------------------------------

    print("\nPreferred configuration per geometric group")
    print("-" * 135)

    preferred_by_group = []

    for group in geometric_groups:

        safe_solutions = [
            solution
            for solution in group
            if solution["safe"]
        ]

        candidates = (
            safe_solutions
            if safe_solutions
            else group
        )

        preferred = min(
            candidates,
            key=lambda item: item["reference_distance"]
        )

        preferred_by_group.append(preferred)

    preferred_by_group.sort(
        key=lambda item: (
            not item["safe"],
            item["reference_distance"]
        )
    )

    for solution in preferred_by_group:

        q_deg = to_deg(
            solution["q"]
        )

        print(
            f"Branch {solution['branch']:2d} | "
            f"Safe: {str(solution['safe']):5s} | "
            f"Joints: "
            + " ".join(
                f"{v:8.2f}"
                for v in q_deg
            )
            + f" | RefDist: "
            f"{solution['reference_distance']:.2f} deg"
        )

    return unique_solutions


# ============================================================
# CSV EXPORT
# ============================================================

def save_solutions_csv(filename, solutions):

    if not SAVE_CSV:
        return

    columns = [
        "id",
        "geometric_group",
        "safe",
        "J1_deg",
        "J2_deg",
        "J3_deg",
        "J4_deg",
        "J5_deg",
        "J6_deg",
        "reference_distance_deg",
        "max_joint_difference_deg",
        "fk_position_error_mm",
        "fk_orientation_error_deg",
    ]

    with open(
        filename,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(columns)

        for index, solution in enumerate(
            solutions,
            start=1
        ):

            writer.writerow([
                index,
                solution["branch"],
                solution["safe"],
                *to_deg(solution["q"]),
                solution["reference_distance"],
                solution["max_joint_difference"],
                solution["pos_err"],
                solution["rot_err"],
            ])

    print(f"\nSaved: {filename}")


# ============================================================
# MAIN
# ============================================================

def main():

    rtde_c = None

    try:

        print("=" * 95)
        print("UR10e MULTIPLE IK SOLUTION DISCOVERY")
        print("NO ROBOT MOVEMENT")
        print("=" * 95)

        # ----------------------------------------------------
        # CONNECT TO CONTROLLER
        # ----------------------------------------------------

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        # ----------------------------------------------------
        # READ ACTIVE TCP OFFSET
        # ----------------------------------------------------

        tcp_offset = list(
            rtde_c.getTCPOffset()
        )

        print(
            "\nActive TCP offset:",
            tcp_offset
        )

        # ----------------------------------------------------
        # REFERENCE JOINT CONFIGURATIONS
        # ----------------------------------------------------

        pick_reference_q = to_rad(
            PICK_REFERENCE_Q_DEG
        )

        place_reference_q = to_rad(
            PLACE_REFERENCE_Q_DEG
        )

        # ----------------------------------------------------
        # REFERENCE TCP POSES FROM CONTROLLER FK
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

        print("\nREFERENCE TCP POSES")
        print("-" * 95)

        print(
            "Pick reference:",
            [
                round(v, 6)
                for v in pick_reference_pose
            ]
        )

        print(
            "Place reference:",
            [
                round(v, 6)
                for v in place_reference_pose
            ]
        )

        # ----------------------------------------------------
        # GENERATE TARGET TCP POSES
        # ----------------------------------------------------

        pick_pose = make_target(
            pick_reference_pose,
            PICK_POSITION,
            PICK_ROTATION_DEG
        )

        place_pose = make_target(
            place_reference_pose,
            PLACE_POSITION,
            PLACE_ROTATION_DEG
        )

        print("\nGENERATED TARGET TCP POSES")
        print("-" * 95)

        print(
            "Pick:",
            [
                round(v, 6)
                for v in pick_pose
            ]
        )

        print(
            "Place:",
            [
                round(v, 6)
                for v in place_pose
            ]
        )

        print(
            "\nPick rotation-vector magnitude:",
            round(
                math.degrees(
                    vector_norm(pick_pose[3:6])
                ),
                6
            ),
            "deg"
        )

        print(
            "Place rotation-vector magnitude:",
            round(
                math.degrees(
                    vector_norm(place_pose[3:6])
                ),
                6
            ),
            "deg"
        )

        # ----------------------------------------------------
        # DISCOVER PICK IK SOLUTIONS
        # ----------------------------------------------------

        pick_solutions = discover_ik_solutions(
            rtde_c,
            "PICK",
            pick_pose,
            pick_reference_q,
            tcp_offset
        )

        # ----------------------------------------------------
        # DISCOVER PLACE IK SOLUTIONS
        # ----------------------------------------------------

        place_solutions = discover_ik_solutions(
            rtde_c,
            "PLACE",
            place_pose,
            place_reference_q,
            tcp_offset
        )

        # ----------------------------------------------------
        # EXPORT CSV
        # ----------------------------------------------------

        save_solutions_csv(
            "pick_ik_solutions.csv",
            pick_solutions
        )

        save_solutions_csv(
            "place_ik_solutions.csv",
            place_solutions
        )

        # ----------------------------------------------------
        # FINAL COMPARISON
        # ----------------------------------------------------

        print("\n" + "=" * 95)
        print("FINAL COMPARISON")
        print("=" * 95)

        for name, solutions in [
            ("PICK", pick_solutions),
            ("PLACE", place_solutions)
        ]:

            geometric_group_count = len(
                set(
                    solution["branch"]
                    for solution in solutions
                )
            )

            safety_valid_count = sum(
                solution["safe"]
                for solution in solutions
            )

            print(
                f"{name:<10s}: "
                f"{len(solutions)} distinct joint vectors, "
                f"{geometric_group_count} geometric groups, "
                f"{safety_valid_count} safety-valid configurations"
            )

        print("\nNo robot movement commands were sent.")

    except Exception as exc:

        print("\nERROR:", exc)

        raise

    finally:

        if rtde_c is not None:

            try:
                rtde_c.stopScript()
            except Exception:
                pass

            try:
                rtde_c.disconnect()
            except Exception:
                pass

        print("\nDisconnected from UR10e.")


if __name__ == "__main__":
    main()