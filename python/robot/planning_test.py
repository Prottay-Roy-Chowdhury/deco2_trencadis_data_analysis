"""
UR10e collision-aware joint-space planning test.

Sequence:
1. Load the existing project-local UR10e model
2. Define HOME and Pose A
3. Find a test obstacle that blocks the direct path
   without colliding with either endpoint
4. Run RRT-Connect if the direct path is blocked
5. Validate the alternative path
6. Visualize the planned path in Swift

OFFLINE SIMULATION ONLY.
No physical robot commands.
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
import roboticstoolbox as rtb
import spatialgeometry as sg
import swift

from spatialmath import SE3

from model_loader import load_robot, HOME_Q_DEG
from motion_test import POSE_A_Q_DEG
from rrt_connect import RRTConnectPlanner


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SHOW_VISUALIZATION = True
TOOL_FRAME = "tool0"

DIRECT_STEPS = 80
SIMULATION_DT = 0.04

RRT_STEP_SIZE = 0.25       # Joint-space Euclidean step, rad
COLLISION_STEP = 0.05      # Max change per joint, rad
RRT_MAX_ITERATIONS = 1500
RRT_RANDOM_SEED = 42

OBSTACLE_COLOR = [1.0, 0.2, 0.2, 0.6]
DIRECT_PATH_COLOR = [0.2, 0.4, 1.0, 0.45]
PLANNED_PATH_COLOR = [0.1, 0.85, 0.35, 1.0]

PATH_RADIUS = 0.003

CAMERA_POSITION = [-0.6, 1.2, 1.4]
CAMERA_LOOK_AT = [-0.3, 0.1, 0.5]


# --------------------------------------------------
# Collision callback
# --------------------------------------------------

def configuration_in_collision(robot, q, obstacle):
    return (
        robot.iscollided(q, obstacle)
        or robot.isselfcollided(q)
    )


# --------------------------------------------------
# Create a valid planning scene
# --------------------------------------------------

def find_test_obstacle(robot, q_start, q_goal, direct_path):
    """
    Find an obstacle that collides with an intermediate
    direct-path configuration, but not either endpoint.

    This is a synthetic test scene, not real workspace geometry.
    """

    print("\n[1] Searching for a suitable test obstacle...")

    # Different positions along the direct joint path.
    fractions = [0.50, 0.40, 0.60, 0.30, 0.70]

    # Use robot link frames to place obstacles near its path.
    reference_links = [
        "wrist_2_link",
        "wrist_1_link",
        "forearm_link",
    ]

    sizes = [0.12, 0.16, 0.20]

    offsets = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.06],
        [0.0, 0.06, 0.0],
        [0.06, 0.0, 0.0],
        [0.0, -0.06, 0.0],
    ]

    for fraction in fractions:
        index = round(
            fraction * (len(direct_path) - 1)
        )

        q_middle = direct_path[index]

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

                    # Both endpoints must be valid.
                    if configuration_in_collision(
                        robot, q_start, obstacle
                    ):
                        continue

                    if configuration_in_collision(
                        robot, q_goal, obstacle
                    ):
                        continue

                    # The obstacle must block the direct path.
                    hits = [
                        i
                        for i, q in enumerate(direct_path)
                        if robot.iscollided(q, obstacle)
                    ]

                    # Require interior collisions rather than
                    # a scene that only touches the endpoints.
                    if (
                        len(hits) >= 2
                        and hits[0] > 0
                        and hits[-1] < len(direct_path) - 1
                    ):
                        print("[OK] Valid test scene found.")
                        print("Reference link:", link_name)
                        print("Obstacle center:", np.round(center, 5))
                        print("Obstacle size:", size)
                        print("Direct collision samples:", len(hits))
                        print("First collision sample:", hits[0])

                        return obstacle

    raise RuntimeError(
        "No suitable obstacle found automatically. "
        "Adjust candidate sizes, offsets, or target joints."
    )


# --------------------------------------------------
# Validate planned trajectory
# --------------------------------------------------

def validate_planned_path(
    robot,
    q_path,
    q_start,
    q_goal,
    obstacle,
    planner,
):

    print("\n[3] Validating planned path...")

    q_path = np.asarray(q_path, dtype=float)

    if q_path.ndim != 2 or q_path.shape[1] != robot.n:
        raise RuntimeError("Invalid planned trajectory dimensions.")

    if not np.all(np.isfinite(q_path)):
        raise RuntimeError("Trajectory contains non-finite values.")

    if not np.allclose(q_path[0], q_start, atol=1e-7):
        raise RuntimeError("Trajectory start does not match HOME.")

    if not np.allclose(q_path[-1], q_goal, atol=1e-7):
        raise RuntimeError("Trajectory goal does not match Pose A.")

    for i, q in enumerate(q_path):
        if not planner.state_valid(q):
            raise RuntimeError(
                f"Collision or joint-limit violation at sample {i}."
            )

    # Recheck the connections at a finer resolution.
    original_step = planner.collision_step

    try:
        planner.collision_step = min(original_step, 0.025)

        for i, (q1, q2) in enumerate(
            zip(q_path[:-1], q_path[1:])
        ):
            if not planner.edge_valid(q1, q2):
                raise RuntimeError(
                    f"Collision between samples {i} and {i + 1}."
                )

    finally:
        planner.collision_step = original_step

    print("[OK] Joint limits and sampled collisions validated.")
    print("Trajectory samples:", len(q_path))


# --------------------------------------------------
# Visualization helpers
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


def visualize(
    robot,
    q_start,
    q_goal,
    direct_path,
    planned_path,
    obstacle,
):

    print("\n[4] Launching Swift...")

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
    env.add_shape(obstacle)

    # Static start and goal coordinate frames.
    env.add_shape(
        sg.Axes(
            length=0.12,
            pose=robot.fkine(q_start, end=TOOL_FRAME),
        )
    )

    env.add_shape(
        sg.Axes(
            length=0.12,
            pose=robot.fkine(q_goal, end=TOOL_FRAME),
        )
    )

    # Show the rejected direct path for comparison.
    env.add_shape(
        create_tool_path(
            robot,
            direct_path,
            DIRECT_PATH_COLOR,
        )
    )

    # Show the alternative path.
    env.add_shape(
        create_tool_path(
            robot,
            planned_path,
            PLANNED_PATH_COLOR,
        )
    )

    # Moving tool0 coordinate frame.
    moving_tool_frame = sg.Axes(
        length=0.08,
        pose=robot.fkine(q_start, end=TOOL_FRAME),
    )

    env.add_shape(moving_tool_frame)

    env.step(SIMULATION_DT)

    print("\nAnimating collision-checked alternative path...")

    for q in planned_path:
        # Defensive check before each displayed configuration.
        if configuration_in_collision(robot, q, obstacle):
            print("[STOP] Collision detected during playback.")
            break

        robot.q = q.copy()

        moving_tool_frame.T = robot.fkine(
            q,
            end=TOOL_FRAME,
        )

        env.step(SIMULATION_DT)

    print("\nFinal displayed joints (degrees):")
    print(np.round(np.rad2deg(robot.q), 4))

    print("\nClose Swift to finish.")
    env.hold()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n========================================")
    print("UR10e RRT-CONNECT PLANNING TEST")
    print("========================================")

    robot = load_robot()

    q_start = np.deg2rad(HOME_Q_DEG)
    q_goal = np.deg2rad(POSE_A_Q_DEG)

    if robot.n != 6:
        raise RuntimeError("Expected a six-axis UR10e model.")

    # Check self-collision before creating the scene.
    if robot.isselfcollided(q_start):
        raise RuntimeError("HOME is in self-collision.")

    if robot.isselfcollided(q_goal):
        raise RuntimeError("Pose A is in self-collision.")

    # Direct path for comparison.
    direct_path = np.asarray(
        rtb.jtraj(
            q_start,
            q_goal,
            DIRECT_STEPS,
        ).q
    )

    obstacle = find_test_obstacle(
        robot,
        q_start,
        q_goal,
        direct_path,
    )

    # Separate collision callback from the planning algorithm.
    def collision_fn(q):
        return configuration_in_collision(robot, q, obstacle)

    planner = RRTConnectPlanner(
        joint_limits=robot.qlim,
        collision_fn=collision_fn,
        step_size=RRT_STEP_SIZE,
        collision_step=COLLISION_STEP,
        max_iterations=RRT_MAX_ITERATIONS,
        seed=RRT_RANDOM_SEED,
    )

    # Verify that a direct connection is actually obstructed.
    if planner.edge_valid(q_start, q_goal):
        raise RuntimeError(
            "Direct path is collision-free. "
            "The obstacle does not create a useful RRT test."
        )

    print("\n[2] Running RRT-Connect...")

    result = planner.plan(
        q_start,
        q_goal,
    )

    planned_path = result["path"]

    print("\n[OK] Planning completed.")
    print("Method:", result["method"])
    print("Iterations:", result["iterations"])
    print("Waypoints:", len(result["waypoints"]))
    print("Dense samples:", len(planned_path))
    print("Collision checks:", result["collision_checks"])

    validate_planned_path(
        robot,
        planned_path,
        q_start,
        q_goal,
        obstacle,
        planner,
    )

    if SHOW_VISUALIZATION:
        visualize(
            robot,
            q_start,
            q_goal,
            direct_path,
            planned_path,
            obstacle,
        )

    print("\nRRT-Connect test completed successfully.")


if __name__ == "__main__":
    main()