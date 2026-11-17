"""Reusable Swift visualization for planned robot motions."""

import numpy as np
import spatialgeometry as sg
import swift


class MotionVisualizer:
    def __init__(
        self,
        robot,
        tool_frame="tool0",
        dt=0.04,
    ):
        self.robot = robot
        self.tool_frame = tool_frame
        self.dt = float(dt)

        self.env = None

    def launch(
        self,
        camera_position=None,
        camera_look_at=None,
    ):
        self.env = swift.Swift()

        self.env.launch(
            realtime=True,
            axes=True,
            ground_opacity=0.25,
            ground_pattern="@grid",
            ground_pattern_width=0.25,
        )

        self.env.add(self.robot)

        if (
            camera_position is not None
            and camera_look_at is not None
        ):
            self.env.set_camera_pose(
                position=camera_position,
                look_at=camera_look_at,
            )

        return self.env

    def add_obstacles(self, obstacles):
        if self.env is None:
            raise RuntimeError("Launch Swift first.")

        for obstacle in obstacles:
            self.env.add_shape(obstacle)

    def add_path(
        self,
        q_path,
        color=None,
        radius=0.003,
    ):
        if self.env is None:
            raise RuntimeError("Launch Swift first.")

        if color is None:
            color = [0.1, 0.85, 0.35, 1.0]

        points = np.array([
            self.robot.fkine(
                q, end=self.tool_frame
            ).t
            for q in q_path
        ])

        if len(points) < 2:
            return None

        line = sg.Polyline(
            points.T,
            radius=radius,
            color=color,
        )

        self.env.add_shape(line)
        return line

    def animate(
        self,
        motion_result,
        show_path=True,
        show_frames=True,
    ):
        if self.env is None:
            raise RuntimeError("Launch Swift first.")

        if not motion_result.success:
            print(
                "[REJECTED] Cannot animate invalid motion:",
                motion_result.reason,
            )
            return False

        q_path = np.asarray(motion_result.path)

        if len(q_path) == 0:
            raise ValueError("Motion path is empty.")

        self.robot.q = q_path[0].copy()

        moving_frame = None

        if show_frames:
            T_start = self.robot.fkine(
                q_path[0], end=self.tool_frame
            )

            T_goal = self.robot.fkine(
                q_path[-1], end=self.tool_frame
            )

            self.env.add_shape(
                sg.Axes(length=0.12, pose=T_start)
            )

            self.env.add_shape(
                sg.Axes(length=0.12, pose=T_goal)
            )

            moving_frame = sg.Axes(
                length=0.08, pose=T_start
            )

            self.env.add_shape(moving_frame)

        if show_path:
            color = (
                [0.10, 0.85, 0.35, 1.0]
                if motion_result.motion_type == "MoveJ"
                else [0.15, 0.40, 1.00, 1.0]
            )

            self.add_path(q_path, color=color)

        for q in q_path:
            self.robot.q = q.copy()

            if moving_frame is not None:
                moving_frame.T = self.robot.fkine(
                    q, end=self.tool_frame
                )

            self.env.step(self.dt)

        return True

    def hold(self):
        if self.env is not None:
            self.env.hold()