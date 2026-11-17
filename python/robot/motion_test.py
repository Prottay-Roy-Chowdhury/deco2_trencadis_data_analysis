"""
UR10e joint and Cartesian motion test in Swift.

Sequence:
    HOME -> POSE A : Joint-space trajectory (MoveJ-style)
    POSE A -> POSE B : Cartesian linear trajectory (MoveL-style)

All poses are defined for the tool0 frame.

OFFLINE SIMULATION ONLY.
No physical robot connection or motion commands.
"""

# --------------------------------------------------
# UTF-8 mode for Windows
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
import swift
import spatialgeometry as sg

from spatialmath import SE3
from scipy.spatial.transform import Rotation

from model_loader import load_robot, HOME_Q_DEG


# --------------------------------------------------
# Runtime configuration
# --------------------------------------------------

SHOW_VISUALIZATION = True

TOOL_FRAME = "tool0"

JOINT_STEPS = 80
LINEAR_STEPS = 60

SIMULATION_DT = 0.04

# Example target joint configuration for Pose A.
# Kept close to HOME to make this a small first test.
POSE_A_Q_DEG = np.array([
    180.0,
    -94.45,
    121.61,
    -180.71,
    -90.0,
    0.0,
])

# Pose B is 500 mm above Pose A in the robot base frame.
LINEAR_OFFSET_M = np.array([
    0.0,
    0.0,
    0.5,
])

# Basic validation thresholds for this demonstration.
MAX_JOINT_STEP_DEG = 20.0
MAX_POSITION_ERROR_M = 0.002
MAX_ORIENTATION_ERROR_DEG = 1.0

# --------------------------------------------------
# Swift visualization settings
# --------------------------------------------------

SHOW_TARGET_FRAMES = True
SHOW_TRAJECTORY_LINES = True
SHOW_MOVING_TOOL_FRAME = True

TARGET_FRAME_LENGTH = 0.12
TOOL_FRAME_LENGTH = 0.08

JOINT_PATH_COLOR = [0.15, 0.40, 1.00, 1.0]   # Blue
LINEAR_PATH_COLOR = [0.10, 0.85, 0.35, 1.0]  # Green

PATH_RADIUS = 0.003  # 3 mm

GROUND_OPACITY = 0.25
GROUND_PATTERN = "@grid"
GROUND_GRID_SIZE = 0.25

CAMERA_POSITION = [1.6, -1.8, 1.4]
CAMERA_LOOK_AT = [-0.3, 0.1, 0.5]


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def print_pose(label, T):
    print(f"\n{label}")
    print(T)
    print("Position (m):", np.round(T.t, 6))


def validate_trajectory(robot, q_path, targets=None):
    """Basic joint continuity and tool0 FK checks.

    Note: this does not perform collision checking.
    """

    q_path = np.asarray(q_path, dtype=float)

    if q_path.ndim != 2 or q_path.shape[1] != robot.n:
        raise RuntimeError("Invalid trajectory dimensions.")

    if not np.all(np.isfinite(q_path)):
        raise RuntimeError("Trajectory contains non-finite joints.")

    jumps = np.abs(np.diff(q_path, axis=0))
    max_step_deg = (
        np.rad2deg(np.max(jumps)) if jumps.size else 0.0
    )

    print(f"Maximum joint step: {max_step_deg:.4f} deg")

    if max_step_deg > MAX_JOINT_STEP_DEG:
        raise RuntimeError(
            "Joint discontinuity detected: "
            f"{max_step_deg:.3f} deg"
        )

    if targets is not None:
        for i, (q, target) in enumerate(zip(q_path, targets)):
            actual = robot.fkine(q, end=TOOL_FRAME)

            position_error = np.linalg.norm(
                actual.t - target.t
            )

            rotation_error = Rotation.from_matrix(
                target.R.T @ actual.R
            ).magnitude()

            if (
                position_error > MAX_POSITION_ERROR_M
                or np.rad2deg(rotation_error)
                > MAX_ORIENTATION_ERROR_DEG
            ):
                raise RuntimeError(
                    f"FK mismatch at sample {i}: "
                    f"position={position_error:.6f} m, "
                    f"orientation="
                    f"{np.rad2deg(rotation_error):.4f} deg"
                )

    print("[OK] Basic trajectory validation passed.")


# --------------------------------------------------
# Define the two target poses
# --------------------------------------------------

def define_poses(robot):

    q_home = np.deg2rad(HOME_Q_DEG)
    q_a = np.deg2rad(POSE_A_Q_DEG)

    T_home = robot.fkine(
        q_home,
        end=TOOL_FRAME,
    )

    # Pose A: derived from the selected joint configuration.
    T_a = robot.fkine(
        q_a,
        end=TOOL_FRAME,
    )

    # Pose B: translate in the base coordinate frame.
    # Preserve the orientation of Pose A.
    T_b = SE3.Rt(
        T_a.R,
        T_a.t + LINEAR_OFFSET_M,
    )

    print_pose("HOME tool0 pose:", T_home)
    print_pose("POSE A tool0 pose:", T_a)
    print_pose("POSE B tool0 pose:", T_b)

    return q_home, q_a, T_a, T_b


# --------------------------------------------------
# Joint trajectory: HOME -> Pose A
# --------------------------------------------------

def plan_joint_motion(robot, q_home, q_a):

    print("\n[1] Planning joint-space trajectory...")

    trajectory = rtb.jtraj(
        q_home,
        q_a,
        JOINT_STEPS,
    )

    q_path = np.asarray(trajectory.q)

    validate_trajectory(robot, q_path)

    print(f"[OK] Joint trajectory: {len(q_path)} samples")

    return q_path


# --------------------------------------------------
# Linear trajectory: Pose A -> Pose B
# --------------------------------------------------

def plan_linear_motion(robot, q_a, T_a, T_b):

    print("\n[2] Planning Cartesian linear trajectory...")

    cartesian_path = rtb.ctraj(
        T_a,
        T_b,
        LINEAR_STEPS,
    )

    # Resolve IK at each Cartesian waypoint.
    # Seed each solve with the previous solution to
    # encourage continuity along the same IK branch.
    q_previous = q_a.copy()
    q_path = []

    for i, T_target in enumerate(cartesian_path):

        solution = robot.ikine_LM(
            T_target,
            q0=q_previous,
            end=TOOL_FRAME,
            joint_limits=True,
            ilimit=100,
            slimit=10,
            tol=1e-6,
        )

        if not solution.success:
            raise RuntimeError(
                f"IK failed at linear sample "
                f"{i + 1}/{LINEAR_STEPS}. "
                f"Reason: {solution.reason}"
            )

        q_previous = np.asarray(
            solution.q,
            dtype=float,
        ).copy()

        q_path.append(q_previous)

    q_path = np.asarray(q_path)

    validate_trajectory(
        robot,
        q_path,
        targets=cartesian_path,
    )

    print(f"[OK] Linear trajectory: {len(q_path)} samples")

    return q_path

# --------------------------------------------------
# Visualization geometry
# --------------------------------------------------

def add_visualization_geometry(
    env,
    robot,
    q_home,
    joint_path,
    linear_path,
):

    print("\nAdding trajectory visualization geometry...")

    # ----------------------------------------------
    # Calculate the tool0 poses
    # ----------------------------------------------

    T_home = robot.fkine(
        q_home,
        end=TOOL_FRAME,
    )

    T_a = robot.fkine(
        joint_path[-1],
        end=TOOL_FRAME,
    )

    T_b = robot.fkine(
        linear_path[-1],
        end=TOOL_FRAME,
    )

    # ----------------------------------------------
    # Static target frames
    # ----------------------------------------------

    if SHOW_TARGET_FRAMES:

        env.add_shape(
            sg.Axes(
                length=TARGET_FRAME_LENGTH,
                pose=T_home,
            )
        )

        env.add_shape(
            sg.Axes(
                length=TARGET_FRAME_LENGTH,
                pose=T_a,
            )
        )

        env.add_shape(
            sg.Axes(
                length=TARGET_FRAME_LENGTH,
                pose=T_b,
            )
        )

        print("[OK] HOME, Pose A and Pose B frames added.")

    # ----------------------------------------------
    # Joint and Cartesian paths
    # ----------------------------------------------

    if SHOW_TRAJECTORY_LINES:

        joint_points = np.array([
            robot.fkine(q, end=TOOL_FRAME).t
            for q in joint_path
        ])

        linear_points = np.array([
            robot.fkine(q, end=TOOL_FRAME).t
            for q in linear_path
        ])

        joint_line = sg.Polyline(
            joint_points.T,
            radius=PATH_RADIUS,
            color=JOINT_PATH_COLOR,
        )

        linear_line = sg.Polyline(
            linear_points.T,
            radius=PATH_RADIUS,
            color=LINEAR_PATH_COLOR,
        )

        env.add_shape(joint_line)
        env.add_shape(linear_line)

        print("[OK] Joint trajectory line added.")
        print("[OK] Linear trajectory line added.")

    # ----------------------------------------------
    # Moving tool frame
    # ----------------------------------------------

    moving_tool_frame = None

    if SHOW_MOVING_TOOL_FRAME:

        moving_tool_frame = sg.Axes(
            length=TOOL_FRAME_LENGTH,
            pose=T_home,
        )

        env.add_shape(moving_tool_frame)

        print("[OK] Moving tool0 frame added.")

    return moving_tool_frame

# --------------------------------------------------
# Animate robot and tool0 frame
# --------------------------------------------------

def animate_path(
    env,
    robot,
    q_path,
    label,
    moving_tool_frame=None,
):

    print(f"\nAnimating: {label}")

    for q in q_path:

        robot.q = q.copy()

        if moving_tool_frame is not None:

            T_tool = robot.fkine(
                q,
                end=TOOL_FRAME,
            )

            moving_tool_frame.T = T_tool

        env.step(SIMULATION_DT)

# --------------------------------------------------
# Swift visualization
# --------------------------------------------------

def visualize(robot, q_home, joint_path, linear_path):

    print("\n[3] Launching Swift...")

    robot.q = q_home.copy()

    env = swift.Swift()

    env.launch(
        realtime=True,
        axes=True,
        ground_opacity=GROUND_OPACITY,
        ground_pattern=GROUND_PATTERN,
        ground_pattern_width=GROUND_GRID_SIZE,
    )

    # Set camera view
    env.set_camera_pose(
        position=CAMERA_POSITION,
        look_at=CAMERA_LOOK_AT,
    )

    # Add robot
    env.add(robot)

    # Add targets, paths, and moving tool frame
    moving_tool_frame = add_visualization_geometry(
        env,
        robot,
        q_home,
        joint_path,
        linear_path,
    )

    env.step(SIMULATION_DT)

    # ----------------------------------------------
    # Joint motion
    # ----------------------------------------------

    animate_path(
        env,
        robot,
        joint_path,
        "HOME -> Pose A (Joint)",
        moving_tool_frame,
    )

    # ----------------------------------------------
    # Linear motion
    # ----------------------------------------------

    animate_path(
        env,
        robot,
        linear_path,
        "Pose A -> Pose B (Linear)",
        moving_tool_frame,
    )

    print("\n[OK] Both simulated motions completed.")

    print("Final configuration (degrees):")
    print(np.round(np.rad2deg(robot.q), 4))

    print("\nClose the Swift browser tab to finish.")

    env.hold()
# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("\n========================================")
    print("UR10e JOINT + LINEAR MOTION TEST")
    print("========================================")

    robot = load_robot()

    q_home, q_a, T_a, T_b = define_poses(robot)

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

    if SHOW_VISUALIZATION:
        visualize(
            robot,
            q_home,
            joint_path,
            linear_path,
        )

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()