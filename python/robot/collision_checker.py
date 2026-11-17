"""Reusable collision checking for Robotics Toolbox robots."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CollisionResult:
    valid: bool
    reason: str = ""
    obstacle_index: int | None = None
    segment: int | None = None
    fraction: float | None = None


class CollisionChecker:
    def __init__(self, robot, max_joint_step=0.025):
        self.robot = robot
        self.max_joint_step = float(max_joint_step)
        self.obstacles = []

        if self.max_joint_step <= 0:
            raise ValueError("max_joint_step must be positive.")

    def add_obstacle(self, obstacle):
        """Register a SpatialGeometry collision object."""
        self.obstacles.append(obstacle)
        return len(self.obstacles) - 1

    def clear_obstacles(self):
        self.obstacles.clear()

    def check_configuration(self, q):
        """Check joint limits, self-collision and obstacles."""
        q = np.asarray(q, dtype=float)

        if q.shape != (self.robot.n,):
            return CollisionResult(False, "INVALID_JOINT_SHAPE")

        if not np.all(np.isfinite(q)):
            return CollisionResult(False, "NONFINITE_JOINTS")

        limits = np.asarray(self.robot.qlim, dtype=float)

        if np.any(q < limits[0] - 1e-9) or np.any(
            q > limits[1] + 1e-9
        ):
            return CollisionResult(False, "JOINT_LIMIT")

        if self.robot.isselfcollided(q):
            return CollisionResult(False, "SELF_COLLISION")

        for i, obstacle in enumerate(self.obstacles):
            if self.robot.iscollided(q, obstacle):
                return CollisionResult(
                    False, "OBSTACLE_COLLISION",
                    obstacle_index=i,
                )

        return CollisionResult(True)

    def check_segment(self, q_start, q_goal, segment=None):
        """
        Check interpolated joint configurations, including
        both segment endpoints.
        """
        a = np.asarray(q_start, dtype=float)
        b = np.asarray(q_goal, dtype=float)

        if a.shape != (self.robot.n,) or b.shape != (self.robot.n,):
            return CollisionResult(False, "INVALID_JOINT_SHAPE")

        if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            return CollisionResult(False, "NONFINITE_JOINTS")

        steps = max(
            1,
            int(np.ceil(
                np.max(np.abs(b - a)) / self.max_joint_step
            )),
        )

        for j in range(steps + 1):
            alpha = j / steps
            q = (1.0 - alpha) * a + alpha * b

            result = self.check_configuration(q)

            if not result.valid:
                result.segment = segment
                result.fraction = alpha
                return result

        return CollisionResult(True)

    def check_path(self, q_path):
        """Check every joint-space segment of a path."""
        path = np.asarray(q_path, dtype=float)

        if path.ndim != 2 or path.shape[1] != self.robot.n:
            return CollisionResult(False, "INVALID_PATH_SHAPE")

        if len(path) == 0:
            return CollisionResult(False, "EMPTY_PATH")

        if len(path) == 1:
            return self.check_configuration(path[0])

        for i in range(len(path) - 1):
            result = self.check_segment(
                path[i], path[i + 1], segment=i
            )

            if not result.valid:
                return result

        return CollisionResult(True)