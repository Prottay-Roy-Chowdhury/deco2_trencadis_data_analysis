"""
Bidirectional RRT-Connect planner for a robot joint space.

- Independent of robot model and visualization
- Uses an external collision-checking callback
- Checks joint limits and intermediate configurations
- Uses a fixed random seed for repeatable tests
- Optional path shortcutting
- Returns joint-space waypoints and a densely sampled path

The output is geometric, not time-parameterized.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Node:
    q: np.ndarray
    parent: int | None


class Tree:
    def __init__(self, root, label):
        self.nodes = [Node(np.asarray(root, dtype=float).copy(), None)]
        self.label = label

    def nearest(self, q):
        distances = [
            np.linalg.norm(node.q - q)
            for node in self.nodes
        ]
        return int(np.argmin(distances))

    def add(self, q, parent):
        self.nodes.append(Node(np.asarray(q).copy(), parent))
        return len(self.nodes) - 1

    def trace(self, index):
        path = []

        while index is not None:
            node = self.nodes[index]
            path.append(node.q.copy())
            index = node.parent

        return path[::-1]


class RRTConnectPlanner:
    def __init__(
        self,
        joint_limits,
        collision_fn,
        step_size=0.25,
        collision_step=0.05,
        max_iterations=1500,
        seed=42,
    ):
        self.limits = np.asarray(joint_limits, dtype=float)
        self.collision_fn = collision_fn

        if self.limits.ndim != 2 or self.limits.shape[0] != 2:
            raise ValueError("Joint limits must be a 2xN array.")

        if not np.all(np.isfinite(self.limits)):
            raise ValueError("Joint limits must be finite.")

        if np.any(self.limits[1] <= self.limits[0]):
            raise ValueError("Invalid joint limits.")

        self.n = self.limits.shape[1]
        self.step_size = float(step_size)
        self.collision_step = float(collision_step)
        self.max_iterations = int(max_iterations)
        self.rng = np.random.default_rng(seed)
        self.collision_checks = 0

        if self.step_size <= 0 or self.collision_step <= 0:
            raise ValueError("Step sizes must be positive.")

    def state_valid(self, q):
        q = np.asarray(q, dtype=float)
        self.collision_checks += 1

        if q.shape != (self.n,) or not np.all(np.isfinite(q)):
            return False

        if np.any(q < self.limits[0] - 1e-9):
            return False

        if np.any(q > self.limits[1] + 1e-9):
            return False

        return not bool(self.collision_fn(q))

    def edge_valid(self, q1, q2):
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)

        # Limit the change of every joint between samples.
        steps = max(
            1,
            int(np.ceil(
                np.max(np.abs(q2 - q1)) / self.collision_step
            )),
        )

        for i in range(steps + 1):
            alpha = i / steps
            q = (1.0 - alpha) * q1 + alpha * q2

            if not self.state_valid(q):
                return False

        return True

    def steer(self, q_from, q_to):
        difference = q_to - q_from
        distance = np.linalg.norm(difference)

        if distance <= self.step_size:
            return q_to.copy()

        return q_from + difference / distance * self.step_size

    def extend(self, tree, target):
        nearest_index = tree.nearest(target)
        q_nearest = tree.nodes[nearest_index].q

        if np.linalg.norm(target - q_nearest) < 1e-9:
            return "reached", nearest_index

        q_new = self.steer(q_nearest, target)

        if not self.edge_valid(q_nearest, q_new):
            return "trapped", nearest_index

        new_index = tree.add(q_new, nearest_index)

        if np.linalg.norm(q_new - target) < 1e-9:
            return "reached", new_index

        return "advanced", new_index

    def connect(self, tree, target):
        for _ in range(200):
            status, index = self.extend(tree, target)

            if status != "advanced":
                return status, index

        return "trapped", index

    def sample(self, q_start, q_goal):
        r = self.rng.random()

        # Occasionally sample the endpoints.
        if r < 0.05:
            return q_start.copy()
        if r < 0.10:
            return q_goal.copy()

        # Mostly search near the direct start-goal region.
        if r < 0.75:
            alpha = self.rng.random()

            center = (
                (1.0 - alpha) * q_start
                + alpha * q_goal
            )

            sample = center + self.rng.normal(
                0.0, 0.65, size=self.n
            )

            return np.clip(
                sample,
                self.limits[0],
                self.limits[1],
            )

        # Global exploration when local detours are insufficient.
        return self.rng.uniform(
            self.limits[0],
            self.limits[1],
        )

    def densify(self, waypoints):
        result = [np.asarray(waypoints[0]).copy()]

        for q1, q2 in zip(waypoints[:-1], waypoints[1:]):
            steps = max(
                1,
                int(np.ceil(
                    np.max(np.abs(q2 - q1))
                    / self.collision_step
                )),
            )

            for i in range(1, steps + 1):
                alpha = i / steps
                result.append(
                    (1.0 - alpha) * q1 + alpha * q2
                )

        return np.asarray(result)

    def shortcut(self, waypoints, attempts=80):
        path = [np.asarray(q).copy() for q in waypoints]

        for _ in range(attempts):
            if len(path) < 3:
                break

            i, j = sorted(
                self.rng.choice(
                    len(path), size=2, replace=False
                )
            )

            if j <= i + 1:
                continue

            if self.edge_valid(path[i], path[j]):
                path = path[:i + 1] + path[j:]

        return path

    def plan(self, q_start, q_goal):
        self.collision_checks = 0

        q_start = np.asarray(q_start, dtype=float)
        q_goal = np.asarray(q_goal, dtype=float)

        if not self.state_valid(q_start):
            raise ValueError("Start configuration is invalid or colliding.")

        if not self.state_valid(q_goal):
            raise ValueError("Goal configuration is invalid or colliding.")

        # Avoid searching when a direct connection already works.
        if self.edge_valid(q_start, q_goal):
            waypoints = [q_start.copy(), q_goal.copy()]
            return {
                "waypoints": np.asarray(waypoints),
                "path": self.densify(waypoints),
                "iterations": 0,
                "collision_checks": self.collision_checks,
                "method": "direct",
            }

        start_tree = Tree(q_start, "start")
        goal_tree = Tree(q_goal, "goal")

        tree_a = start_tree
        tree_b = goal_tree

        for iteration in range(1, self.max_iterations + 1):
            target = self.sample(q_start, q_goal)

            status_a, index_a = self.extend(tree_a, target)

            if status_a != "trapped":
                q_new = tree_a.nodes[index_a].q

                status_b, index_b = self.connect(
                    tree_b, q_new
                )

                if status_b == "reached":
                    if tree_a.label == "start":
                        start_path = tree_a.trace(index_a)
                        goal_path = tree_b.trace(index_b)
                    else:
                        start_path = tree_b.trace(index_b)
                        goal_path = tree_a.trace(index_a)

                    waypoints = (
                        start_path + goal_path[::-1][1:]
                    )

                    waypoints = self.shortcut(waypoints)

                    # Final validation of every connection.
                    for q1, q2 in zip(
                        waypoints[:-1], waypoints[1:]
                    ):
                        if not self.edge_valid(q1, q2):
                            raise RuntimeError(
                                "Planned path failed final validation."
                            )

                    return {
                        "waypoints": np.asarray(waypoints),
                        "path": self.densify(waypoints),
                        "iterations": iteration,
                        "collision_checks": self.collision_checks,
                        "method": "rrt_connect",
                    }

            # Alternate which tree grows first.
            tree_a, tree_b = tree_b, tree_a

        raise RuntimeError(
            "RRT-Connect could not find a path within "
            f"{self.max_iterations} iterations."
        )