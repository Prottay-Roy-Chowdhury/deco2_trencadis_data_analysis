"""
UR10e REUSABLE CLASSES INTEGRATION TEST
======================================

Tests
-----
1. Robot model loading
2. External configuration validation
3. Forward kinematics
4. Collision checking
5. Direct MoveJ planning
6. Direct MoveL planning
7. RRT-Connect interface
8. MoveJ obstacle avoidance
9. MoveL collision rejection
10. Swift visualization

Architecture
------------
URRobotModel
    Robot description and kinematics.

CollisionChecker
    Joint limits, self-collision and obstacles.

MotionPlanner
    MoveJ and MoveL planning.

RRTConnectPlanner
    Collision-avoiding joint-space planning.

MotionVisualizer
    Swift visualization.

Important
---------
- No configuration file required.
- No connection to the physical robot.
- No RTDE commands.
- No physical robot movement.
- All trajectories are geometric and offline.
"""

# ============================================================
# WINDOWS UTF-8 BOOTSTRAP
# ============================================================

import os
import sys
import subprocess

if sys.flags.utf8_mode == 0:

    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            *sys.argv,
        ],
        env=environment,
        check=False,
    )

    sys.exit(result.returncode)


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import spatialgeometry as sg

from spatialmath import SE3

from robot_model import URRobotModel
from collision_checker import CollisionChecker
from motion_planner import MotionPlanner
from motion_visualizer import MotionVisualizer
from rrt_connect import RRTConnectPlanner


# ============================================================
# TEST PARAMETERS
# ============================================================

# Temporary starting configuration.
# This is NOT defined inside URRobotModel.

TEST_START_DEG = [
    135.0,
    -99.45,
    126.61,
    -114.71,
    -90.0,
    0.0,
]

# Joint-space target for MoveJ.

TEST_TARGET_DEG = [
    180.0,
    -94.45,
    121.61,
    -180.71,
    -90.0,
    0.0,
]

# Cartesian MoveL translation relative to the robot base.
# Move the tool0 frame vertically upward by 300 mm.

LINEAR_TRANSLATION = [
    0.0,
    0.0,
    0.30,
]

# Trajectory resolution.

MOVEJ_SAMPLES = 150
MOVEL_SAMPLES = 120

# Collision checking resolution.

MAX_JOINT_STEP = 0.025

# RRT parameters.

RRT_SEED = 42
RRT_STEP_SIZE = 0.25
RRT_ITERATIONS = 3000

# Visualization.

ENABLE_VISUALIZATION = True
VISUALIZATION_DT = 0.04


# ============================================================
# PRINT UTILITIES
# ============================================================

def section(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_configuration(name, q):

    print(f"\n{name}")

    q_deg = np.rad2deg(q)

    for index, angle in enumerate(q_deg, start=1):

        print(
            f"Joint {index}: {angle:10.4f} deg"
        )


def print_motion_result(name, result):

    print(f"\n{name}")

    print(f"Success : {result.success}")
    print(f"Method  : {result.method}")
    print(f"Reason  : {result.reason}")

    if result.success:

        print(
            f"Samples : {len(result.path)}"
        )

        print(
            f"Path shape: {result.path.shape}"
        )

        if result.details:

            print(
                f"Details : {result.details}"
            )


def assert_valid(result, name):

    if not result.valid:

        raise RuntimeError(
            f"{name} validation failed: "
            f"{result.reason}"
        )


# ============================================================
# OBSTACLE SEARCH
# ============================================================

def find_intermediate_obstacle(
    robot,
    q_path,
    candidate_sizes=None,
    max_joint_step=MAX_JOINT_STEP,
):
    """
    Find an obstacle that collides with an intermediate
    robot configuration while leaving both endpoints clear.

    Designed for both MoveJ and MoveL test trajectories.

    Returns
    -------
    obstacle
        SpatialGeometry Cuboid.

    first_collision_index
        Index of first colliding sampled configuration.
    """

    path = np.asarray(q_path, dtype=float)

    if path.ndim != 2 or len(path) < 3:
        raise ValueError(
            "Trajectory must contain at least 3 samples."
        )

    # Include links closer to the end effector.

    candidate_links = [
        "wrist_3_link",
        "wrist_2_link",
        "wrist_1_link",
        "forearm_link",
        "upper_arm_link",
    ]

    # Smaller obstacles are important for MoveL.

    if candidate_sizes is None:
        candidate_sizes = [
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.08,
            0.10,
            0.12,
        ]

    candidate_fractions = [
        0.50,
        0.45,
        0.55,
        0.40,
        0.60,
        0.35,
        0.65,
        0.30,
        0.70,
    ]

    candidate_offsets = [
        [0.00, 0.00, 0.00],
        [0.02, 0.00, 0.00],
        [-0.02, 0.00, 0.00],
        [0.00, 0.02, 0.00],
        [0.00, -0.02, 0.00],
        [0.00, 0.00, 0.02],
        [0.00, 0.00, -0.02],
        [0.04, 0.00, 0.00],
        [-0.04, 0.00, 0.00],
        [0.00, 0.04, 0.00],
        [0.00, -0.04, 0.00],
        [0.00, 0.00, 0.04],
        [0.00, 0.00, -0.04],
    ]

    print(
        "\nSearching for an intermediate obstacle..."
    )

    candidates_tested = 0

    for fraction in candidate_fractions:

        sample_index = int(
            fraction * (len(path) - 1)
        )

        for link_name in candidate_links:

            T_link = robot.fkine(
                path[sample_index],
                end=link_name,
            )

            link_position = np.asarray(
                T_link.t,
                dtype=float,
            )

            for offset in candidate_offsets:

                center = (
                    link_position
                    + np.asarray(offset, dtype=float)
                )

                for size in candidate_sizes:

                    candidates_tested += 1

                    obstacle = sg.Cuboid(
                        scale=[size, size, size],
                        pose=SE3.Trans(*center),
                        color=[1.0, 0.1, 0.1, 0.5],
                    )

                    checker = CollisionChecker(
                        robot,
                        max_joint_step=max_joint_step,
                    )

                    checker.add_obstacle(obstacle)

                    # ----------------------------------
                    # 1. Test intermediate configuration
                    # ----------------------------------

                    middle_result = (
                        checker.check_configuration(
                            path[sample_index]
                        )
                    )

                    if (
                        middle_result.reason
                        != "OBSTACLE_COLLISION"
                    ):
                        continue

                    # ----------------------------------
                    # 2. Check start configuration
                    # ----------------------------------

                    start_result = (
                        checker.check_configuration(
                            path[0]
                        )
                    )

                    if not start_result.valid:
                        continue

                    # ----------------------------------
                    # 3. Check end configuration
                    # ----------------------------------

                    end_result = (
                        checker.check_configuration(
                            path[-1]
                        )
                    )

                    if not end_result.valid:
                        continue

                    # ----------------------------------
                    # 4. Validate complete trajectory
                    # ----------------------------------

                    path_result = checker.check_path(
                        path
                    )

                    if (
                        path_result.reason
                        != "OBSTACLE_COLLISION"
                    ):
                        continue

                    # ----------------------------------
                    # 5. Find first sampled collision
                    # ----------------------------------

                    first_collision = None

                    for index in range(
                        1,
                        len(path) - 1,
                    ):

                        result = (
                            checker.check_configuration(
                                path[index]
                            )
                        )

                        if (
                            result.reason
                            == "OBSTACLE_COLLISION"
                        ):

                            first_collision = index
                            break

                    if first_collision is None:
                        continue

                    # ----------------------------------
                    # SUCCESS
                    # ----------------------------------

                    print(
                        "[OK] Suitable obstacle found."
                    )

                    print(
                        f"Reference link: {link_name}"
                    )

                    print(
                        "Obstacle center:",
                        np.round(center, 5),
                    )

                    print(
                        "Obstacle size:",
                        size,
                    )

                    print(
                        "Reference sample:",
                        sample_index,
                    )

                    print(
                        "First sampled collision:",
                        first_collision,
                    )

                    print(
                        "Candidates tested:",
                        candidates_tested,
                    )

                    return (
                        obstacle,
                        first_collision,
                    )

    raise RuntimeError(
        "Could not find a suitable intermediate obstacle. "
        f"Candidates tested: {candidates_tested}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    section(
        "UR10e REUSABLE CLASSES INTEGRATION TEST"
    )

    print("NO PHYSICAL ROBOT MOVEMENT")

    # ========================================================
    # 1. LOAD ROBOT MODEL
    # ========================================================

    section("[1] ROBOT MODEL")

    model = URRobotModel(
        robot_type="ur10e",
        tool_frame="tool0",
        verbose=True,
    )

    robot = model.load()

    assert model.is_loaded
    assert model.n_joints == 6

    model.summary()

    print(
        "\n[PASS] Robot model loaded."
    )

    # ========================================================
    # 2. EXTERNAL CONFIGURATIONS
    # ========================================================

    section("[2] EXTERNAL CONFIGURATIONS")

    q_start = np.deg2rad(
        TEST_START_DEG
    )

    q_target = np.deg2rad(
        TEST_TARGET_DEG
    )

    q_start = model.validate_configuration(
        q_start,
        check_limits=True,
    )

    q_target = model.validate_configuration(
        q_target,
        check_limits=True,
    )

    print_configuration(
        "START",
        q_start,
    )

    print_configuration(
        "TARGET",
        q_target,
    )

    print(
        "\n[PASS] External configurations validated."
    )

    # ========================================================
    # 3. FORWARD KINEMATICS
    # ========================================================

    section("[3] FORWARD KINEMATICS")

    T_start = model.fk(
        q_start
    )

    T_target = model.fk(
        q_target
    )

    print("\nSTART tool0 pose:")
    print(T_start)

    print("\nTARGET tool0 pose:")
    print(T_target)

    print(
        "\n[PASS] Forward kinematics computed."
    )

    # ========================================================
    # 4. COLLISION CHECKER
    # ========================================================

    section("[4] COLLISION CHECKER")

    collision = CollisionChecker(
        robot,
        max_joint_step=MAX_JOINT_STEP,
    )

    start_check = collision.check_configuration(
        q_start
    )

    target_check = collision.check_configuration(
        q_target
    )

    print(
        "START valid:",
        start_check.valid
    )

    print(
        "TARGET valid:",
        target_check.valid
    )

    assert_valid(
        start_check,
        "START"
    )

    assert_valid(
        target_check,
        "TARGET"
    )

    print(
        "\n[PASS] Endpoint collision validation."
    )

    # ========================================================
    # 5. MOTION PLANNER
    # ========================================================

    section("[5] MOTION PLANNER")

    planner = MotionPlanner(
        robot,
        collision,
        tool_frame=model.tool_frame,
        rrt_seed=RRT_SEED,
        rrt_step_size=RRT_STEP_SIZE,
        rrt_iterations=RRT_ITERATIONS,
    )

    print(
        "[PASS] MotionPlanner initialized."
    )

    # ========================================================
    # 6. DIRECT MOVEJ
    # ========================================================

    section("[6] DIRECT MOVEJ PLANNING")

    result_j = planner.move_j(
        q_start=q_start,
        q_goal=q_target,
        samples=MOVEJ_SAMPLES,
        allow_rrt=True,
    )

    print_motion_result(
        "START -> TARGET",
        result_j,
    )

    if not result_j.success:

        raise RuntimeError(
            f"Direct MoveJ failed: {result_j.reason}"
        )

    if result_j.method != "direct":

        raise RuntimeError(
            "Expected direct MoveJ, "
            f"received: {result_j.method}"
        )

    joint_validation = collision.check_path(
        result_j.path
    )

    assert_valid(
        joint_validation,
        "Direct MoveJ"
    )

    print(
        "\n[PASS] Direct MoveJ validated."
    )

    # ========================================================
    # 7. DIRECT MOVEL
    # ========================================================

    section("[7] DIRECT MOVEL PLANNING")

    T_linear_target = (
        SE3.Trans(*LINEAR_TRANSLATION)
        * T_target
    )

    print("\nMoveL starting pose:")
    print(T_target)

    print("\nMoveL target pose:")
    print(T_linear_target)

    result_l = planner.move_l(
        q_start=q_target,
        target_pose=T_linear_target,
        samples=MOVEL_SAMPLES,
    )

    print_motion_result(
        "TARGET -> LINEAR TARGET",
        result_l,
    )

    if not result_l.success:

        raise RuntimeError(
            f"Direct MoveL failed: {result_l.reason}"
        )

    linear_validation = collision.check_path(
        result_l.path
    )

    assert_valid(
        linear_validation,
        "Direct MoveL"
    )

    print(
        "\n[PASS] Direct MoveL validated."
    )

    # ========================================================
    # 8. RRT-CONNECT INTERFACE
    # ========================================================

    section("[8] RRT-CONNECT INTERFACE")

    rrt = RRTConnectPlanner(
        joint_limits=model.joint_limits,
        collision_fn=lambda q: (
            not collision.check_configuration(q).valid
        ),
        step_size=RRT_STEP_SIZE,
        collision_step=MAX_JOINT_STEP,
        max_iterations=RRT_ITERATIONS,
        seed=RRT_SEED,
    )

    rrt_result = rrt.plan(
        q_start,
        q_target,
    )

    print(
        "Method:",
        rrt_result["method"]
    )

    print(
        "Iterations:",
        rrt_result["iterations"]
    )

    print(
        "Collision checks:",
        rrt_result["collision_checks"]
    )

    print(
        "Dense samples:",
        len(rrt_result["path"])
    )

    assert_valid(
        collision.check_path(
            rrt_result["path"]
        ),
        "RRT interface trajectory"
    )

    print(
        "\n[PASS] RRT-Connect interface validated."
    )

    # ========================================================
    # 9. MOVEJ OBSTACLE AVOIDANCE
    # ========================================================

    section("[9] MOVEJ OBSTACLE AVOIDANCE")

    # Generate an obstacle along the original MoveJ path.

    joint_obstacle, first_joint_collision = (
        find_intermediate_obstacle(
            robot,
            result_j.path,
            candidate_sizes=[
                0.10,
                0.12,
                0.15,
            ],
        )
    )

    joint_obstacle_checker = CollisionChecker(
        robot,
        max_joint_step=MAX_JOINT_STEP,
    )

    joint_obstacle_checker.add_obstacle(
        joint_obstacle
    )

    # Verify the direct trajectory is obstructed.

    direct_check = joint_obstacle_checker.check_path(
        result_j.path
    )

    print(
        "\nOriginal MoveJ valid:",
        direct_check.valid
    )

    print(
        "Collision reason:",
        direct_check.reason
    )

    print(
        "First sampled collision:",
        first_joint_collision
    )

    if direct_check.valid:

        raise RuntimeError(
            "Obstacle did not block direct MoveJ."
        )

    if (
        direct_check.reason
        != "OBSTACLE_COLLISION"
    ):

        raise RuntimeError(
            "Unexpected direct MoveJ failure: "
            f"{direct_check.reason}"
        )

    # Verify start and goal remain clear.

    assert_valid(
        joint_obstacle_checker.check_configuration(
            q_start
        ),
        "MoveJ start"
    )

    assert_valid(
        joint_obstacle_checker.check_configuration(
            q_target
        ),
        "MoveJ goal"
    )

    print(
        "\n[OK] Direct MoveJ path is obstructed."
    )

    # Create an obstacle-aware motion planner.

    obstacle_motion_planner = MotionPlanner(
        robot,
        joint_obstacle_checker,
        tool_frame=model.tool_frame,
        rrt_seed=RRT_SEED,
        rrt_step_size=RRT_STEP_SIZE,
        rrt_iterations=RRT_ITERATIONS,
    )

    print(
        "\nRunning MoveJ with RRT-Connect fallback..."
    )

    result_rrt = obstacle_motion_planner.move_j(
        q_start=q_start,
        q_goal=q_target,
        samples=MOVEJ_SAMPLES,
        allow_rrt=True,
    )

    print_motion_result(
        "MoveJ obstacle avoidance",
        result_rrt,
    )

    if not result_rrt.success:

        raise RuntimeError(
            f"RRT-Connect failed: {result_rrt.reason}"
        )

    if result_rrt.method != "rrt_connect":

        raise RuntimeError(
            "Expected RRT-Connect, "
            f"received: {result_rrt.method}"
        )

    # Validate the alternative path.

    rrt_validation = (
        joint_obstacle_checker.check_path(
            result_rrt.path
        )
    )

    assert_valid(
        rrt_validation,
        "RRT alternative trajectory"
    )

    print(
        "\n[PASS] Collision-free alternative MoveJ generated."
    )

    # ========================================================
    # 10. MOVEL COLLISION REJECTION
    # ========================================================

    section("[10] MOVEL COLLISION REJECTION")

    # Generate an obstacle along the original MoveL path.

    linear_obstacle, first_linear_collision = (
        find_intermediate_obstacle(
            robot,
            result_l.path,
            candidate_sizes=[
                0.04,
                0.05,
                0.06,
            ],
        )
    )

    linear_obstacle_checker = CollisionChecker(
        robot,
        max_joint_step=MAX_JOINT_STEP,
    )

    linear_obstacle_checker.add_obstacle(
        linear_obstacle
    )

    # Verify the Cartesian trajectory is obstructed.

    linear_path_check = (
        linear_obstacle_checker.check_path(
            result_l.path
        )
    )

    print(
        "\nOriginal MoveL valid:",
        linear_path_check.valid
    )

    print(
        "Collision reason:",
        linear_path_check.reason
    )

    print(
        "First sampled collision:",
        first_linear_collision
    )

    if linear_path_check.valid:

        raise RuntimeError(
            "Obstacle did not block MoveL."
        )

    if (
        linear_path_check.reason
        != "OBSTACLE_COLLISION"
    ):

        raise RuntimeError(
            "Unexpected MoveL validation failure: "
            f"{linear_path_check.reason}"
        )

    # Both Cartesian trajectory endpoints must remain clear.

    assert_valid(
        linear_obstacle_checker.check_configuration(
            result_l.path[0]
        ),
        "MoveL start"
    )

    assert_valid(
        linear_obstacle_checker.check_configuration(
            result_l.path[-1]
        ),
        "MoveL end"
    )

    print(
        "\n[OK] MoveL endpoints are clear."
    )

    print(
        "[OK] Intermediate MoveL path is obstructed."
    )

    # Create obstacle-aware planner.

    blocked_linear_planner = MotionPlanner(
        robot,
        linear_obstacle_checker,
        tool_frame=model.tool_frame,
    )

    # Request the same linear motion.

    blocked_linear_result = (
        blocked_linear_planner.move_l(
            q_start=q_target,
            target_pose=T_linear_target,
            samples=MOVEL_SAMPLES,
        )
    )

    print_motion_result(
        "MoveL with obstacle",
        blocked_linear_result,
    )

    # Must reject instead of rerouting.

    if blocked_linear_result.success:

        raise RuntimeError(
            "MoveL incorrectly accepted "
            "the obstructed Cartesian trajectory."
        )

    if (
        blocked_linear_result.reason
        != "OBSTACLE_COLLISION"
    ):

        raise RuntimeError(
            "MoveL rejected for an unexpected reason: "
            f"{blocked_linear_result.reason}"
        )

    print(
        "\n[PASS] MoveL correctly rejected."
    )

    print(
        "[OK] No alternative MoveL trajectory generated."
    )

    # ========================================================
    # 11. SWIFT VISUALIZATION
    # ========================================================

    if ENABLE_VISUALIZATION:

        section("[11] SWIFT VISUALIZATION")

        visualizer = MotionVisualizer(
            robot,
            tool_frame=model.tool_frame,
            dt=VISUALIZATION_DT,
        )

        # Initialize offline model.

        model.set_configuration(
            q_start,
            check_limits=True,
        )

        # Launch Swift.

        visualizer.launch(
            camera_position=[-1.8, 2.0, 1.6],
            camera_look_at=[0.0, 0.0, 0.8],
            lighting="soft",
        )

        # ----------------------------------------------------
        # A. DIRECT MOVEJ
        # ----------------------------------------------------

        print(
            "\n[Visualization 1]"
        )

        print(
            "Animating direct MoveJ..."
        )

        visualizer.animate(
            result_j,
            show_path=True,
            show_frames=True,
        )

        # ----------------------------------------------------
        # B. DIRECT MOVEL
        # ----------------------------------------------------

        print(
            "\n[Visualization 2]"
        )

        print(
            "Animating direct MoveL..."
        )

        visualizer.animate(
            result_l,
            show_path=True,
            show_frames=True,
        )

        # ----------------------------------------------------
        # C. RRT-CONNECT ALTERNATIVE
        # ----------------------------------------------------

        print(
            "\n[Visualization 3]"
        )

        print(
            "Adding MoveJ collision obstacle..."
        )

        visualizer.add_obstacles(
            [joint_obstacle]
        )

        print(
            "Animating RRT-Connect alternative..."
        )

        model.set_configuration(
            q_start
        )

        visualizer.animate(
            result_rrt,
            show_path=True,
            show_frames=True,
        )

        print(
            "\n[PASS] RRT-Connect alternative visualized."
        )

        # ----------------------------------------------------
        # D. BLOCKED MOVEL REFERENCE
        # ----------------------------------------------------

        print(
            "\n[Visualization 4]"
        )

        print(
            "Adding MoveL collision obstacle..."
        )

        visualizer.add_obstacles(
            [linear_obstacle]
        )

        # Show the original Cartesian trajectory as
        # a static reference line.
        #
        # The rejected motion is NOT animated.

        visualizer.add_path(
            result_l.path,
            color=[1.0, 0.2, 0.2, 1.0],
            radius=0.004,
        )

        # Display robot at MoveL start.

        model.set_configuration(
            q_target
        )

        visualizer.env.step(0.05)

        print(
            "\nBlocked MoveL reference displayed."
        )

        print(
            "Red trajectory represents the rejected MoveL."
        )

        print(
            "Rejected MoveL will not be animated."
        )

        print(
            "\nClose Swift to finish."
        )

        visualizer.hold()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    section("FINAL INTEGRATION TEST SUMMARY")

    print("[PASS] URRobotModel")
    print("[PASS] External configuration validation")
    print("[PASS] Forward kinematics")
    print("[PASS] CollisionChecker")
    print("[PASS] Direct MoveJ")
    print("[PASS] Direct MoveL")
    print("[PASS] RRTConnectPlanner interface")
    print("[PASS] MoveJ obstacle detection")
    print("[PASS] RRT-Connect alternative planning")
    print("[PASS] RRT trajectory collision validation")
    print("[PASS] MoveL intermediate collision detection")
    print("[PASS] MoveL collision rejection")

    if ENABLE_VISUALIZATION:

        print(
            "[PASS] MotionVisualizer"
        )

    print()

    print(
        "All integration checks passed."
    )

    print(
        "No robot controller connection was made."
    )

    print(
        "No physical robot movement commands were sent."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()