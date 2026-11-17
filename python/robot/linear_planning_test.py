
"""
UR10e collision-validated Cartesian linear motion test.

Tests:
    1. Generate a linear tool0 trajectory with IK.
    2. Validate joint limits and self-collision.
    3. Test collision-free movement.
    4. Create an obstacle blocking the linear trajectory.
    5. Reject the blocked trajectory before movement.
    6. Visualize both outcomes in Swift.

MoveL behavior:
    - Preserve the specified Cartesian trajectory.
    - Reject collision, IK failure, or joint-limit violation.
    - Never automatically replan around obstacles.

OFFLINE SIMULATION ONLY.
No physical robot connection or movement.
"""

# --------------------------------------------------
# Windows UTF-8 mode
# --------------------------------------------------

import os
import sys
import subprocess

if sys.platform == "win32" and not sys.flags.utf8_mode:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"

    sys.exit(
        subprocess.call(
            [sys.executable, *sys.argv],
            env=env,
        )
    )

# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
import spatialgeometry as sg
import swift

from spatialmath import SE3

from model_loader import load_robot
from motion_test import define_poses, plan_linear_motion


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SHOW_VISUALIZATION = True

TOOL_FRAME = "tool0"

SIMULATION_DT = 0.04

# Maximum change of any joint between collision samples.
# Radians (~1.43 degrees).
COLLISION_STEP = 0.025

SHOW_TARGET_FRAMES = True
SHOW_TRAJECTORY_LINE = True
SHOW_MOVING_TOOL_FRAME = True

FRAME_LENGTH = 0.12
PATH_RADIUS = 0.003

CLEAR_PATH_COLOR = [0.10, 0.85, 0.35, 1.0]
BLOCKED_PATH_COLOR = [1.00, 0.25, 0.20, 0.65]
OBSTACLE_COLOR = [1.00, 0.20, 0.20, 0.60]

CAMERA_POSITION = [-0.6, 1.2, 1.4]
CAMERA_LOOK_AT = [-0.3, 0.1, 0.65]


# --------------------------------------------------
# Collision checking
# --------------------------------------------------

def configuration_status(robot, q, obstacles):
    """Check one joint configuration."""

    q = np.asarray(q, dtype=float)

    if q.shape != (robot.n,) or not np.all(np.isfinite(q)):
        return "INVALID_JOINT_CONFIGURATION"

    limits = np.asarray(robot.qlim, dtype=float)

    if np.any(q < limits[0] - 1e-9):
        return "JOINT_LIMIT"

    if np.any(q > limits[1] + 1e-9):
        return "JOINT_LIMIT"

    if robot.isselfcollided(q):
        return "SELF_COLLISION"

    for i, obstacle in enumerate(obstacles):
        if robot.iscollided(q, obstacle):
            return f"OBSTACLE_COLLISION_{i}"

    return None


def validate_linear_trajectory(robot, q_path, obstacles):
    """
    Validate each waypoint and additional intermediate
    joint configurations between successive waypoints.

    Returns a result dictionary. Does not animate.
    """

    q_path = np.asarray(q_path, dtype=float)

    if q_path.ndim != 2 or q_path.shape[1] != robot.n:
        return {
            "valid": False,
            "reason": "INVALID_TRAJECTORY_SHAPE",
            "sample": None,
        }

    if len(q_path) == 0:
        return {
            "valid": False,
            "reason": "EMPTY_TRAJECTORY",
            "sample": None,
        }

    total_checks = 0

    # Check first configuration.
    reason = configuration_status(
        robot, q_path[0], obstacles
    )
    total_checks += 1

    if reason is not None:
        return {
            "valid": False,
            "reason": reason,
            "sample": 0,
            "checks": total_checks,
        }

    # Validate every segment, including its endpoint.
    for i in range(len(q_path) - 1):

        q1 = q_path[i]
        q2 = q_path[i + 1]

        steps = max(
            1,
            int(np.ceil(
                np.max(np.abs(q2 - q1))
                / COLLISION_STEP
            )),
        )

        for j in range(1, steps + 1):

            alpha = j / steps
            q = (1.0 - alpha) * q1 + alpha * q2

            reason = configuration_status(
                robot, q, obstacles
            )

            total_checks += 1

            if reason is not None:
                return {
                    "valid": False,
                    "reason": reason,
                    "sample": i + 1,
                    "segment": i,
                    "fraction": alpha,
                    "checks": total_checks,
                }

    return {
        "valid": True,
        "reason": None,
        "sample": None,
        "checks": total_checks,
    }


# --------------------------------------------------
# Obstacle generation for blocked-path test
# --------------------------------------------------

def create_blocking_obstacle(robot, q_path):
    """
    Find a synthetic obstacle intersecting the linear
    motion but not its start or goal configurations.

    The obstacle is selected only for this demonstration.
    """

    print("\n[3] Creating obstructed MoveL scenario...")

    fractions = [0.5, 0.4, 0.6, 0.3, 0.7]

    reference_links = [
        "wrist_3_link",
        "wrist_2_link",
        "wrist_1_link",
    ]

    sizes = [0.08, 0.12, 0.16, 0.20]

    offsets = [
        [0.0, 0.0, 0.0],
        [0.04, 0.0, 0.0],
        [-0.04, 0.0, 0.0],
        [0.0, 0.04, 0.0],
        [0.0, -0.04, 0.0],
        [0.0, 0.0, 0.04],
    ]

    for fraction in fractions:

        index = int(
            round(fraction * (len(q_path) - 1))
        )

        q_middle = q_path[index]

        for link_name in reference_links:

            T_link = robot.fkine(
                q_middle,
                end=link_name,
            )

            for size in sizes:
                for offset in offsets:

                    center = (
                        T_link.t
                        + np.asarray(offset, dtype=float)
                    )

                    obstacle = sg.Cuboid(
                        scale=[size, size, size],
                        pose=SE3(*center),
                        color=OBSTACLE_COLOR,
                    )

                    # Both endpoints must remain collision-free.
                    if configuration_status(
                        robot, q_path[0], [obstacle]
                    ) is not None:
                        continue

                    if configuration_status(
                        robot, q_path[-1], [obstacle]
                    ) is not None:
                        continue

                    result = validate_linear_trajectory(
                        robot,
                        q_path,
                        [obstacle],
                    )

                    if (
                        not result["valid"]
                        and result["reason"].startswith(
                            "OBSTACLE_COLLISION"
                        )
                    ):
                        print("[OK] Blocking obstacle found.")
                        print("Reference link:", link_name)
                        print(
                            "Obstacle center:",
                            np.round(center, 5),
                        )
                        print("Obstacle size:", size)
                        print(
                            "First invalid sample:",
                            result["sample"],
                        )

                        return obstacle

    raise RuntimeError(
        "Could not find an obstacle that blocks the "
        "linear path while preserving both endpoints."
    )


# --------------------------------------------------
# Trajectory visualization helpers
# --------------------------------------------------

def create_tool_path(robot, q_path, color):

    points = np.array([
        robot.fkine(q, end=TOOL_FRAME).t
        for q in q_path
    ])

    return sg.Polyline(
        points.T,
        radius=PATH_RADIUS,
        color=color,
    )


def animate_linear_path(
    env,
    robot,
    q_path,
    moving_tool_frame,
):

    print("\nAnimating collision-free MoveL...")

    for q in q_path:

        robot.q = q.copy()

        if moving_tool_frame is not None:
            moving_tool_frame.T = robot.fkine(
                q,
                end=TOOL_FRAME,
            )

        env.step(SIMULATION_DT)

    print("[OK] Collision-free MoveL completed.")


# --------------------------------------------------
# Swift visualization
# --------------------------------------------------

def visualize(
    robot,
    q_path,
    obstacle,
    clear_result,
    blocked_result,
):

    print("\n[5] Launching Swift visualization...")

    q_start = q_path[0].copy()
    q_goal = q_path[-1].copy()

    T_start = robot.fkine(
        q_start,
        end=TOOL_FRAME,
    )

    T_goal = robot.fkine(
        q_goal,
        end=TOOL_FRAME,
    )

    robot.q = q_start.copy()

    env = swift.Swift()

    env.launch(
        realtime=True,
        axes=True,
        ground_opacity=0.25,
        ground_pattern="@grid",
        ground_pattern_width=0.25,
    )

    env.set_camera_pose(
        position=CAMERA_POSITION,
        look_at=CAMERA_LOOK_AT,
    )

    env.add(robot)

    if SHOW_TARGET_FRAMES:
        env.add_shape(
            sg.Axes(
                length=FRAME_LENGTH,
                pose=T_start,
            )
        )

        env.add_shape(
            sg.Axes(
                length=FRAME_LENGTH,
                pose=T_goal,
            )
        )

    moving_tool_frame = None

    if SHOW_MOVING_TOOL_FRAME:
        moving_tool_frame = sg.Axes(
            length=0.08,
            pose=T_start,
        )

        env.add_shape(moving_tool_frame)

    # --------------------------------------------------
    # Scenario 1: Valid linear motion
    # --------------------------------------------------

    print("\nSCENARIO 1: COLLISION-FREE MoveL")

    if clear_result["valid"]:

        if SHOW_TRAJECTORY_LINE:
            env.add_shape(
                create_tool_path(
                    robot,
                    q_path,
                    CLEAR_PATH_COLOR,
                )
            )

        env.step(SIMULATION_DT)

        animate_linear_path(
            env,
            robot,
            q_path,
            moving_tool_frame,
        )

    else:
        print("[REJECTED] Clear-scene motion invalid.")
        print(clear_result["reason"])

    # --------------------------------------------------
    # Scenario 2: Blocked linear motion
    # --------------------------------------------------

    print("\nSCENARIO 2: BLOCKED MoveL")

    # Return to start before showing the blocked scene.
    # This is an instantaneous visualization reset,
    # not a planned or collision-validated return motion.
    robot.q = q_start.copy()

    if moving_tool_frame is not None:
        moving_tool_frame.T = T_start

    # Add the obstacle only for the blocked scenario.
    env.add_shape(obstacle)

    # Show the requested but rejected tool0 path.
    if SHOW_TRAJECTORY_LINE:
        env.add_shape(
            create_tool_path(
                robot,
                q_path,
                BLOCKED_PATH_COLOR,
            )
        )

    env.step(SIMULATION_DT)

    if not blocked_result["valid"]:

        print("[REJECTED] Linear motion blocked.")
        print("Reason:", blocked_result["reason"])
        print("Invalid sample:", blocked_result["sample"])
        print(
            "[STOP] Robot remains at the starting pose."
        )

    else:
        print(
            "[WARNING] Expected blocked scenario "
            "was not detected."
        )

    print("\nClose Swift to finish.")
    env.hold()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n========================================")
    print("UR10e LINEAR COLLISION PLANNING TEST")
    print("========================================")

    robot = load_robot()

    # Reuse the Cartesian targets from motion_test.py.
    q_home, q_a, T_a, T_b = define_poses(robot)

    # --------------------------------------------------
    # Generate the requested linear trajectory
    # --------------------------------------------------

    print("\n[1] Generating Cartesian linear trajectory...")

    try:
        q_path = plan_linear_motion(
            robot,
            q_a,
            T_a,
            T_b,
        )

    except RuntimeError as exc:
        print("\n[REJECTED] Linear planning failed.")
        print("Reason: IK or trajectory generation failure.")
        print("Details:", exc)
        return

    # --------------------------------------------------
    # Scenario 1: No workspace obstacle
    # --------------------------------------------------

    print("\n[2] Testing collision-free MoveL...")

    clear_result = validate_linear_trajectory(
        robot,
        q_path,
        obstacles=[],
    )

    print("Valid:", clear_result["valid"])
    print("Reason:", clear_result["reason"])
    print("Collision checks:", clear_result["checks"])

    if not clear_result["valid"]:
        print(
            "[REJECTED] Baseline linear path is invalid. "
            "No animation or blocked-scene test will run."
        )
        return

    # --------------------------------------------------
    # Scenario 2: Obstacle along the linear trajectory
    # --------------------------------------------------

    obstacle = create_blocking_obstacle(
        robot,
        q_path,
    )

    print("\n[4] Validating obstructed MoveL...")

    blocked_result = validate_linear_trajectory(
        robot,
        q_path,
        obstacles=[obstacle],
    )

    print("Valid:", blocked_result["valid"])
    print("Reason:", blocked_result["reason"])
    print("First invalid sample:", blocked_result["sample"])
    print("Collision checks:", blocked_result["checks"])

    if blocked_result["valid"]:
        raise RuntimeError(
            "Blocked-scene test failed: no collision detected."
        )

    # --------------------------------------------------
    # Visualize
    # --------------------------------------------------

    if SHOW_VISUALIZATION:
        visualize(
            robot,
            q_path,
            obstacle,
            clear_result,
            blocked_result,
        )

    print("\nLinear planning test completed.")


if __name__ == "__main__":
    main()