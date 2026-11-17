"""
Example of using reusable planning classes.

Offline UR10e planning and Swift visualization.
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

import numpy as np
import spatialgeometry as sg

from spatialmath import SE3

from model_loader import load_robot, HOME_Q_DEG
from collision_checker import CollisionChecker
from motion_planner import MotionPlanner
from motion_visualizer import MotionVisualizer


def main():
    # Load project-local UR10e.
    robot = load_robot()

    # Create collision environment.
    collision = CollisionChecker(robot)

    obstacle = sg.Cuboid(
        scale=[0.12, 0.12, 0.12],
        pose=SE3(-0.56, 0.04, 0.65),
        color=[1.0, 0.2, 0.2, 0.6],
    )

    collision.add_obstacle(obstacle)

    # Create planner.
    planner = MotionPlanner(
        robot,
        collision,
        tool_frame="tool0",
        rrt_seed=42,
    )

    # Start and goal joint configurations.
    q_start = np.deg2rad(HOME_Q_DEG)

    q_goal = np.deg2rad([
        180.0,
        -94.45,
        121.61,
        -180.71,
        -90.0,
        0.0,
    ])

    # ----------------------------------------------
    # MoveJ: direct or RRT-Connect
    # ----------------------------------------------

    result = planner.move_j(q_start, q_goal)

    print("\nMoveJ successful:", result.success)
    print("Method:", result.method)
    print("Reason:", result.reason)

    if not result.success:
        return

    # ----------------------------------------------
    # MoveL: straight-line tool0 motion
    # ----------------------------------------------

    T_goal = robot.fkine(
        q_goal, end="tool0"
    )

    T_linear_goal = SE3.Rt(
        T_goal.R,
        T_goal.t + np.array([0, 0, 0.5]),
    )

    linear_result = planner.move_l(
        q_goal,
        T_linear_goal,
    )

    print("\nMoveL successful:", linear_result.success)
    print("Reason:", linear_result.reason)

    # ----------------------------------------------
    # Visualize successful MoveJ
    # ----------------------------------------------

    visualizer = MotionVisualizer(robot)

    visualizer.launch(
        camera_position=[-0.6, 1.2, 1.4],
        camera_look_at=[-0.3, 0.1, 0.5],
    )

    visualizer.add_obstacles(
        collision.obstacles
    )

    visualizer.animate(result)

    if linear_result.success:
        visualizer.animate(linear_result)
    else:
        print(
            "MoveL rejected:",
            linear_result.reason,
        )

    visualizer.hold()


if __name__ == "__main__":
    main()