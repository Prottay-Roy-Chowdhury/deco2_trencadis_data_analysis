"""
Reusable Swift visualization for planned robot motions.

Features
--------
- Robot visualization
- MoveJ and MoveL animation
- Trajectory visualization
- Start and target frames
- Moving tool frame
- Obstacle visualization
- Camera configuration
- Configurable scene lighting

Lighting modes
--------------
"default":
    Swift's standard lighting.

"soft":
    Soft hemisphere and ambient lighting.
    No directional shadows.

"none":
    All scene lights disabled.

No physical robot movement.
"""

import numpy as np
import spatialgeometry as sg
import swift

from swift.Light import (
    HemisphereLight,
    AmbientLight,
)


class MotionVisualizer:

    # ========================================================
    # INITIALIZATION
    # ========================================================

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

    # ========================================================
    # LIGHTING
    # ========================================================

    @staticmethod
    def create_lighting(
        lighting="default",
    ):
        """
        Create a Swift lighting configuration.

        Parameters
        ----------
        lighting : str
            "default", "soft", or "none".

        Returns
        -------
        list or None
            Swift light objects.
        """

        if lighting == "default":

            # None preserves Swift's default lights.
            return None

        elif lighting == "soft":

            return [

                HemisphereLight(
                    sky_color=(1.0, 1.0, 1.0),
                    ground_color=(0.55, 0.55, 0.55),
                    intensity=2.0,
                ),

                AmbientLight(
                    color=(1.0, 1.0, 1.0),
                    intensity=1.50,
                ),

            ]

        elif lighting == "none":

            return []

        else:

            raise ValueError(
                "Unsupported lighting mode: "
                f"{lighting}. "
                "Use 'default', 'soft', or 'none'."
            )

    # ========================================================
    # LAUNCH SWIFT
    # ========================================================

    def launch(
        self,
        camera_position=None,
        camera_look_at=None,
        lighting="default",
    ):
        """
        Launch Swift visualization.

        Parameters
        ----------
        camera_position : array-like, optional
            Camera position in world coordinates.

        camera_look_at : array-like, optional
            Camera target in world coordinates.

        lighting : str
            Lighting mode:
                "default"
                "soft"
                "none"
        """

        scene_lights = self.create_lighting(
            lighting
        )

        self.env = swift.Swift()

        self.env.launch(
            realtime=True,
            axes=True,
            ground_opacity=0.25,
            ground_pattern="@grid",
            ground_pattern_width=0.25,
            lights=scene_lights,
        )

        self.env.add(self.robot)

        # ----------------------------------------------
        # Camera configuration
        # ----------------------------------------------

        if (
            camera_position is not None
            and camera_look_at is not None
        ):

            self.env.set_camera_pose(
                position=camera_position,
                look_at=camera_look_at,
            )

        return self.env

    # ========================================================
    # CHANGE LIGHTING
    # ========================================================

    def set_lighting(
        self,
        lighting="soft",
    ):
        """
        Change scene lighting after Swift has launched.

        Note
        ----
        The default Swift rig is initialized at launch.
        To restore that exact rig, relaunch with
        lighting="default".
        """

        if self.env is None:

            raise RuntimeError(
                "Launch Swift first."
            )

        if lighting == "default":

            raise ValueError(
                "To restore Swift's default lighting, "
                "relaunch with lighting='default'."
            )

        scene_lights = self.create_lighting(
            lighting
        )

        self.env.set_lights(
            scene_lights
        )

    # ========================================================
    # ADD OBSTACLES
    # ========================================================

    def add_obstacles(
        self,
        obstacles,
    ):

        if self.env is None:

            raise RuntimeError(
                "Launch Swift first."
            )

        for obstacle in obstacles:

            self.env.add_shape(
                obstacle
            )

    # ========================================================
    # ADD TRAJECTORY
    # ========================================================

    def add_path(
        self,
        q_path,
        color=None,
        radius=0.003,
    ):

        if self.env is None:

            raise RuntimeError(
                "Launch Swift first."
            )

        if color is None:

            color = [
                0.1,
                0.85,
                0.35,
                1.0,
            ]

        points = np.array([

            self.robot.fkine(
                q,
                end=self.tool_frame,
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

        self.env.add_shape(
            line
        )

        return line

    # ========================================================
    # ANIMATE MOTION
    # ========================================================

    def animate(
        self,
        motion_result,
        show_path=True,
        show_frames=True,
    ):

        if self.env is None:

            raise RuntimeError(
                "Launch Swift first."
            )

        if not motion_result.success:

            print(
                "[REJECTED] Cannot animate invalid motion:",
                motion_result.reason,
            )

            return False

        q_path = np.asarray(
            motion_result.path
        )

        if len(q_path) == 0:

            raise ValueError(
                "Motion path is empty."
            )

        self.robot.q = q_path[0].copy()

        moving_frame = None

        # ----------------------------------------------
        # Start and target frames
        # ----------------------------------------------

        if show_frames:

            T_start = self.robot.fkine(
                q_path[0],
                end=self.tool_frame,
            )

            T_goal = self.robot.fkine(
                q_path[-1],
                end=self.tool_frame,
            )

            self.env.add_shape(
                sg.Axes(
                    length=0.12,
                    pose=T_start,
                )
            )

            self.env.add_shape(
                sg.Axes(
                    length=0.12,
                    pose=T_goal,
                )
            )

            moving_frame = sg.Axes(
                length=0.08,
                pose=T_start,
            )

            self.env.add_shape(
                moving_frame
            )

        # ----------------------------------------------
        # Trajectory visualization
        # ----------------------------------------------

        if show_path:

            color = (

                [0.10, 0.85, 0.35, 1.0]

                if motion_result.motion_type == "MoveJ"

                else [0.15, 0.40, 1.00, 1.0]

            )

            self.add_path(
                q_path,
                color=color,
            )

        # ----------------------------------------------
        # Animation
        # ----------------------------------------------

        for q in q_path:

            self.robot.q = q.copy()

            if moving_frame is not None:

                moving_frame.T = (
                    self.robot.fkine(
                        q,
                        end=self.tool_frame,
                    )
                )

            self.env.step(
                self.dt
            )

        return True

    # ========================================================
    # HOLD VISUALIZATION
    # ========================================================

    def hold(self):

        if self.env is not None:

            self.env.hold()