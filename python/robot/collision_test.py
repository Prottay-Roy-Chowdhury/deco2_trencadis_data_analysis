"""
UR10e collision detection test.

Tests:
1. Collision geometry availability
2. Known collision and known separation
3. Self-collision
4. Joint trajectory versus obstacle
5. Linear trajectory versus obstacle
6. Optional Swift visualization

OFFLINE ONLY. No connection to the real robot.
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
    sys.exit(subprocess.call(
        [sys.executable, *sys.argv],
        env=env
    ))

# --------------------------------------------------
# Imports
# --------------------------------------------------

import numpy as np
import spatialgeometry as sg
import swift

from spatialmath import SE3

from model_loader import load_robot
from motion_test import (
    define_poses,
    plan_joint_motion,
    plan_linear_motion,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

SHOW_VISUALIZATION = True
SIMULATION_DT = 0.04

# Size of the example obstacle, in metres
OBSTACLE_SIZE = [0.18, 0.18, 0.18]

# --------------------------------------------------
# Collision geometry inspection
# --------------------------------------------------

def inspect_collision_geometry(robot):

    print("\n[1] Inspecting collision geometry...")

    total_shapes = 0

    for link in robot.links:

        count = len(link.collision)
        total_shapes += count

        print(
            f"{link.name:<22} "
            f"collision shapes: {count}"
        )

    print(f"\nTotal collision shapes: {total_shapes}")

    if total_shapes == 0:
        raise RuntimeError(
            "The robot has no collision geometry. "
            "Collision checking cannot be validated."
        )

    print("[OK] Robot collision geometry is present.")


# --------------------------------------------------
# Positive and negative control tests
# --------------------------------------------------

def test_collision_backend(robot, q_home):

    print("\n[2] Testing collision backend...")

    # Deliberately large box covering the robot.
    # This is ONLY a positive-control test.
    colliding_box = sg.Cuboid(
        scale=[4.0, 4.0, 4.0],
        pose=SE3(0.0, 0.0, 0.0),
    )

    # Clearly separated box.
    separated_box = sg.Cuboid(
        scale=[0.20, 0.20, 0.20],
        pose=SE3(10.0, 10.0, 10.0),
    )

    positive = robot.iscollided(
        q_home,
        colliding_box,
    )

    negative = robot.iscollided(
        q_home,
        separated_box,
    )

    print(f"Known overlap detected: {positive}")
    print(f"Known separation detected: {not negative}")

    if not positive:
        raise RuntimeError(
            "Collision backend failed the positive control."
        )

    if negative:
        raise RuntimeError(
            "Collision backend failed the negative control."
        )

    print("[OK] Positive and negative controls passed.")


# --------------------------------------------------
# Self-collision
# --------------------------------------------------

def test_self_collision(robot, q_home, q_a):

    print("\n[3] Testing self-collision...")

    home_collision = robot.isselfcollided(q_home)
    pose_a_collision = robot.isselfcollided(q_a)

    print(f"HOME self-collision: {home_collision}")
    print(f"Pose A self-collision: {pose_a_collision}")

    print("[OK] Self-collision queries completed.")

    if home_collision or pose_a_collision:
        print(
            "[WARNING] At least one configuration reports "
            "self-collision. This needs investigation."
        )


# --------------------------------------------------
# Test obstacle
# --------------------------------------------------

def create_test_obstacle(robot, joint_path):

    print("\n[4] Creating test obstacle...")

    # Position the obstacle around a wrist joint
    # at the midpoint of the joint trajectory.
    #
    # This is a deliberately approximate test scene,
    # not a calibrated workspace obstacle.

    midpoint_q = joint_path[len(joint_path) // 2]

    T_wrist = robot.fkine(
        midpoint_q,
        end="wrist_2_link",
    )

    obstacle = sg.Cuboid(
        scale=OBSTACLE_SIZE,
        pose=SE3(*T_wrist.t),
        color=[1.0, 0.2, 0.2, 0.55],
    )

    print("Obstacle center:", np.round(T_wrist.t, 5))
    print("Obstacle size:", OBSTACLE_SIZE)

    midpoint_collision = robot.iscollided(
        midpoint_q,
        obstacle,
    )

    print(
        "Collision at selected midpoint:",
        midpoint_collision
    )

    return obstacle


# --------------------------------------------------
# Trajectory collision checking
# --------------------------------------------------

def check_trajectory(robot, q_path, obstacle, label):

    print(f"\n[5] Checking {label}...")

    collision_indices = []
    self_collision_indices = []

    for i, q in enumerate(q_path):

        if robot.iscollided(q, obstacle):
            collision_indices.append(i)

        if robot.isselfcollided(q):
            self_collision_indices.append(i)

    print(f"Total samples: {len(q_path)}")
    print(
        "Obstacle-collision samples:",
        len(collision_indices)
    )
    print(
        "Self-collision samples:",
        len(self_collision_indices)
    )

    if collision_indices:
        print(
            "First obstacle collision at sample:",
            collision_indices[0]
        )
    else:
        print(
            "No obstacle collision detected at "
            "the sampled configurations."
        )

    if self_collision_indices:
        print(
            "First self-collision at sample:",
            self_collision_indices[0]
        )

    return collision_indices, self_collision_indices

# --------------------------------------------------
# Collision-aware animation
# --------------------------------------------------

def animate_until_collision(
    env,
    robot,
    q_path,
    obstacle,
    label,
):
    print(f"\nAnimating: {label}")

    for i, q in enumerate(q_path):

        # Check collision BEFORE updating robot pose
        obstacle_collision = robot.iscollided(q, obstacle)
        self_collision = robot.isselfcollided(q)

        if obstacle_collision or self_collision:

            print("\n[COLLISION DETECTED]")
            print(f"Trajectory: {label}")
            print(f"Sample: {i}")
            print(f"Obstacle collision: {obstacle_collision}")
            print(f"Self collision: {self_collision}")
            print("[STOP] Animation halted before colliding pose.")

            return False

        robot.q = q.copy()
        env.step(SIMULATION_DT)

    print(f"[OK] {label} completed without detected collision.")
    return True

# --------------------------------------------------
# Swift visualization
# --------------------------------------------------

def visualize_collision_scene(
    robot,
    joint_path,
    linear_path,
    obstacle,
):

    print("\n[6] Launching Swift collision visualization...")

    robot.q = joint_path[0].copy()

    env = swift.Swift()

    env.launch(
        realtime=True,
        axes=True,
        ground_opacity=0.25,
        ground_pattern="@grid",
    )

    env.add(robot)
    env.add_shape(obstacle)

    env.step(SIMULATION_DT)

    # --------------------------------------------------
    # Joint motion
    # --------------------------------------------------

    joint_success = animate_until_collision(
        env,
        robot,
        joint_path,
        obstacle,
        "HOME -> Pose A (Joint)",
    )

    # --------------------------------------------------
    # Linear motion
    # --------------------------------------------------

    if joint_success:

        linear_success = animate_until_collision(
            env,
            robot,
            linear_path,
            obstacle,
            "Pose A -> Pose B (Linear)",
        )

    else:
        print(
            "\n[STOP] Linear motion skipped because "
            "the preceding joint motion encountered a collision."
        )

    print("\n[OK] Collision trajectory visualization completed.")
    print("Close Swift to finish.")

    env.hold()


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n========================================")
    print("UR10e COLLISION DETECTION TEST")
    print("========================================")

    robot = load_robot()

    inspect_collision_geometry(robot)

    q_home, q_a, T_a, T_b = define_poses(robot)

    # Reuse the existing, working trajectory generators.
    joint_path = plan_joint_motion(
        robot,
        q_home,
        q_a,
    )

    linear_path = plan_linear_motion(
        robot,
        q_a,
        T_a,
        T_b,
    )

    test_collision_backend(robot, q_home)

    test_self_collision(robot, q_home, q_a)

    obstacle = create_test_obstacle(
        robot,
        joint_path,
    )

    check_trajectory(
        robot,
        joint_path,
        obstacle,
        "HOME -> Pose A (Joint)",
    )

    check_trajectory(
        robot,
        linear_path,
        obstacle,
        "Pose A -> Pose B (Linear)",
    )

    if SHOW_VISUALIZATION:
        visualize_collision_scene(
            robot,
            joint_path,
            linear_path,
            obstacle,
        )

    print("\nCollision detection test completed.")


if __name__ == "__main__":
    main()