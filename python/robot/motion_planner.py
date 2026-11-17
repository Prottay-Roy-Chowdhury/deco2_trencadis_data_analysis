"""Reusable MoveJ and MoveL planning for Robotics Toolbox."""

from dataclasses import dataclass, field
import numpy as np
import roboticstoolbox as rtb

from scipy.spatial.transform import Rotation

from rrt_connect import RRTConnectPlanner


@dataclass
class MotionResult:
    success: bool
    motion_type: str
    method: str
    path: np.ndarray | None = None
    reason: str = ""
    details: dict = field(default_factory=dict)


class MotionPlanner:
    def __init__(
        self,
        robot,
        collision_checker,
        tool_frame="tool0",
        rrt_seed=42,
        rrt_step_size=0.25,
        rrt_iterations=1500,
    ):
        self.robot = robot
        self.collision = collision_checker
        self.tool_frame = tool_frame

        if self.collision.robot is not robot:
            raise ValueError(
                "CollisionChecker must use the same robot instance."
            )

        self.rrt_seed = rrt_seed
        self.rrt_step_size = rrt_step_size
        self.rrt_iterations = rrt_iterations

    def _failure(self, motion_type, reason, **details):
        return MotionResult(
            success=False,
            motion_type=motion_type,
            method="rejected",
            reason=reason,
            details=details,
        )

    def _check_endpoints(self, q_start, q_goal):
        start = self.collision.check_configuration(q_start)
        goal = self.collision.check_configuration(q_goal)

        if not start.valid:
            return f"START_{start.reason}"

        if not goal.valid:
            return f"GOAL_{goal.reason}"

        return None

    # ----------------------------------------------
    # MoveJ: direct path + RRT fallback
    # ----------------------------------------------

    def move_j(
        self,
        q_start,
        q_goal,
        samples=80,
        allow_rrt=True,
    ):
        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        if samples < 2:
            raise ValueError("MoveJ requires at least 2 samples.")

        reason = self._check_endpoints(q_start, q_goal)

        if reason:
            return self._failure("MoveJ", reason)

        direct = np.asarray(
            rtb.jtraj(q_start, q_goal, samples).q
        )

        validation = self.collision.check_path(direct)

        if validation.valid:
            return MotionResult(
                True,
                "MoveJ",
                "direct",
                path=direct,
            )

        if not allow_rrt:
            return self._failure(
                "MoveJ",
                validation.reason,
                blocked_segment=validation.segment,
            )

        # Search another joint-space route.
        planner = RRTConnectPlanner(
            joint_limits=self.robot.qlim,
            collision_fn=lambda q: not (
                self.collision.check_configuration(q).valid
            ),
            step_size=self.rrt_step_size,
            collision_step=self.collision.max_joint_step,
            max_iterations=self.rrt_iterations,
            seed=self.rrt_seed,
        )

        try:
            result = planner.plan(q_start, q_goal)

        except (ValueError, RuntimeError) as exc:
            return self._failure(
                "MoveJ",
                "RRT_PLANNING_FAILED",
                message=str(exc),
            )

        path = np.asarray(result["path"], dtype=float)

        final_check = self.collision.check_path(path)

        if not final_check.valid:
            return self._failure(
                "MoveJ",
                "RRT_PATH_VALIDATION_FAILED",
                collision_reason=final_check.reason,
            )

        return MotionResult(
            True,
            "MoveJ",
            result["method"],
            path=path,
            details={
                "iterations": result["iterations"],
                "collision_checks": result["collision_checks"],
                "waypoints": result["waypoints"],
            },
        )

    # ----------------------------------------------
    # MoveL: Cartesian path + IK + validation
    # ----------------------------------------------

    def move_l(
        self,
        q_start,
        target_pose,
        samples=60,
        max_ik_jump_rad=0.35,
        position_tolerance=0.002,
        orientation_tolerance_deg=1.0,
    ):
        """
        Generate a straight Cartesian path from the current
        tool_frame pose to target_pose.

        Reject if IK, limits, or collisions fail.
        Never invoke RRT or reroute the Cartesian path.
        """

        if samples < 2:
            raise ValueError("MoveL requires at least 2 samples.")

        q_start = np.asarray(q_start, dtype=float)

        start_check = self.collision.check_configuration(q_start)

        if not start_check.valid:
            return self._failure(
                "MoveL",
                f"START_{start_check.reason}",
            )

        T_start = self.robot.fkine(
            q_start, end=self.tool_frame
        )

        cartesian_path = rtb.ctraj(
            T_start, target_pose, samples
        )

        q_previous = q_start.copy()
        joint_path = [q_start.copy()]

        for i in range(1, len(cartesian_path)):
            target = cartesian_path[i]

            solution = self.robot.ikine_LM(
                target,
                q0=q_previous,
                end=self.tool_frame,
                joint_limits=True,
                ilimit=100,
                slimit=10,
                tol=1e-6,
            )

            if not solution.success:
                return self._failure(
                    "MoveL",
                    "IK_FAILED",
                    sample=i,
                    message=str(solution.reason),
                )

            q_new = np.asarray(
                solution.q, dtype=float
            ).copy()

            if q_new.shape != (self.robot.n,) or not np.all(
                np.isfinite(q_new)
            ):
                return self._failure(
                    "MoveL", "INVALID_IK_SOLUTION", sample=i
                )

            # Reject large IK branch changes.
            max_jump = np.max(np.abs(q_new - q_previous))

            if max_jump > max_ik_jump_rad:
                return self._failure(
                    "MoveL",
                    "IK_DISCONTINUITY",
                    sample=i,
                    jump_rad=float(max_jump),
                )

            # Verify FK agreement with the Cartesian target.
            actual = self.robot.fkine(
                q_new, end=self.tool_frame
            )

            position_error = np.linalg.norm(
                actual.t - target.t
            )

            angle_error = np.rad2deg(
                Rotation.from_matrix(
                    target.R.T @ actual.R
                ).magnitude()
            )

            if (
                position_error > position_tolerance
                or angle_error > orientation_tolerance_deg
            ):
                return self._failure(
                    "MoveL",
                    "CARTESIAN_ACCURACY_FAILED",
                    sample=i,
                    position_error_m=float(position_error),
                    orientation_error_deg=float(angle_error),
                )

            # Validate the entire joint segment before accepting it.
            segment_check = self.collision.check_segment(
                q_previous, q_new, segment=i - 1
            )

            if not segment_check.valid:
                return self._failure(
                    "MoveL",
                    segment_check.reason,
                    sample=i,
                    obstacle_index=segment_check.obstacle_index,
                    fraction=segment_check.fraction,
                )

            joint_path.append(q_new)
            q_previous = q_new

        path = np.asarray(joint_path)

        # Final complete-path validation.
        final_check = self.collision.check_path(path)

        if not final_check.valid:
            return self._failure(
                "MoveL",
                final_check.reason,
                segment=final_check.segment,
            )

        return MotionResult(
            True,
            "MoveL",
            "cartesian_ik",
            path=path,
            details={
                "samples": len(path),
                "target_pose": target_pose,
            },
        )