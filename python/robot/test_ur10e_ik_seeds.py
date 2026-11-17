import math
import rtde_control
import rtde_receive


# ============================================================
# CONFIGURATION
# ============================================================

ROBOT_IP = "192.168.56.101"  # Change to your UR10e IP

JOINT_NAMES = [
    "Base",
    "Shoulder",
    "Elbow",
    "Wrist 1",
    "Wrist 2",
    "Wrist 3"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_degrees(joints):
    return [math.degrees(q) for q in joints]


def print_joints(title, joints):
    print(f"\n{title}")
    print("-" * 60)

    for name, value in zip(JOINT_NAMES, joints):
        print(
            f"{name:12s}: "
            f"{value:9.5f} rad  "
            f"{math.degrees(value):9.2f} deg"
        )


def print_joint_difference(reference, solution):
    print("\nJoint differences from seed:")
    print("-" * 60)

    for name, ref, sol in zip(
        JOINT_NAMES, reference, solution
    ):
        diff = math.degrees(sol - ref)

        print(
            f"{name:12s}: "
            f"{diff:+10.4f} deg"
        )


def print_tcp_pose(pose):
    print("\nTCP pose (robot base frame)")
    print("-" * 60)

    labels = ["X", "Y", "Z", "RX", "RY", "RZ"]

    for label, value in zip(labels, pose):
        unit = "m" if label in ("X", "Y", "Z") else "rad"
        print(f"{label:4s}: {value:12.6f} {unit}")


def test_ik_seed(rtde_c, name, target_pose, seed):
    print("\n" + "=" * 60)
    print(f"IK TEST: {name}")
    print("=" * 60)

    print_joints("Seed configuration (qnear)", seed)

    try:
        has_solution = (
            rtde_c.getInverseKinematicsHasSolution(
                target_pose,
                seed
            )
        )

        print("\nIK solution available:", has_solution)

        if not has_solution:
            print("No IK solution for this seed.")
            return None

        solution = rtde_c.getInverseKinematics(
            target_pose,
            seed
        )

        print_joints(
            "Calculated IK configuration",
            solution
        )

        print_joint_difference(seed, solution)

        joint_safe = rtde_c.isJointsWithinSafetyLimits(
            solution
        )

        print("\nJoint safety check:", joint_safe)

        print(
            "J6 seed:",
            f"{math.degrees(seed[5]):.3f} deg"
        )

        print(
            "J6 IK:",
            f"{math.degrees(solution[5]):.3f} deg"
        )

        return {
            "name": name,
            "seed": seed,
            "solution": solution,
            "safe": joint_safe
        }

    except Exception as e:
        print(f"\nIK ERROR: {e}")
        return None


# ============================================================
# MAIN
# ============================================================

def main():
    rtde_c = None
    rtde_r = None

    try:
        print("=" * 60)
        print("UR10e IK SEED TEST")
        print("NO ROBOT MOTION")
        print("=" * 60)

        print("\nConnecting to UR10e...")

        rtde_r = rtde_receive.RTDEReceiveInterface(
            ROBOT_IP
        )

        rtde_c = rtde_control.RTDEControlInterface(
            ROBOT_IP
        )

        print("Connected successfully.")

        # ----------------------------------------------------
        # 1. Read actual robot state
        # ----------------------------------------------------

        actual_q = list(rtde_r.getActualQ())
        actual_tcp = list(rtde_r.getActualTCPPose())

        print_joints(
            "Actual robot joint configuration",
            actual_q
        )

        print_tcp_pose(actual_tcp)

        # ----------------------------------------------------
        # 2. Create IK seeds
        # ----------------------------------------------------

        seed_current = actual_q.copy()

        seed_j6_minus = actual_q.copy()
        seed_j6_minus[5] -= 2.0 * math.pi

        seed_j6_plus = actual_q.copy()
        seed_j6_plus[5] += 2.0 * math.pi

        seeds = [
            ("Current configuration", seed_current),
            ("J6 minus 360 degrees", seed_j6_minus),
            ("J6 plus 360 degrees", seed_j6_plus)
        ]

        # ----------------------------------------------------
        # 3. Run IK tests
        # ----------------------------------------------------

        results = []

        for name, seed in seeds:
            result = test_ik_seed(
                rtde_c,
                name,
                actual_tcp,
                seed
            )

            if result is not None:
                results.append(result)

        # ----------------------------------------------------
        # 4. Summary
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("IK TEST SUMMARY")
        print("=" * 60)

        print(
            f"{'Test':28s} "
            f"{'Seed J6':>10s} "
            f"{'IK J6':>10s} "
            f"{'Safe':>8s}"
        )

        print("-" * 60)

        for result in results:
            seed_j6 = math.degrees(
                result["seed"][5]
            )

            ik_j6 = math.degrees(
                result["solution"][5]
            )

            print(
                f"{result['name']:28s} "
                f"{seed_j6:10.2f} "
                f"{ik_j6:10.2f} "
                f"{str(result['safe']):>8s}"
            )

        # ----------------------------------------------------
        # 5. Compare IK solutions
        # ----------------------------------------------------

        print("\n" + "=" * 60)
        print("SOLUTION COMPARISON")
        print("=" * 60)

        if len(results) >= 2:
            reference_solution = results[0]["solution"]

            for result in results[1:]:
                print(
                    f"\nCompared with: "
                    f"{results[0]['name']}"
                )

                print(
                    f"Candidate: {result['name']}"
                )

                for joint_name, q_ref, q_sol in zip(
                    JOINT_NAMES,
                    reference_solution,
                    result["solution"]
                ):
                    difference = math.degrees(
                        q_sol - q_ref
                    )

                    print(
                        f"{joint_name:12s}: "
                        f"{difference:+10.4f} deg"
                    )

        print("\n" + "=" * 60)
        print("TEST COMPLETED")
        print("No robot motion commands were sent.")
        print("=" * 60)

    except Exception as e:
        print("\nMAIN ERROR:", str(e))

    finally:
        # ----------------------------------------------------
        # 6. Clean shutdown
        # ----------------------------------------------------

        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception as e:
                print("stopScript warning:", e)

        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass

        if rtde_c is not None:
            try:
                rtde_c.disconnect()
            except Exception:
                pass

        print("\nDisconnected from UR10e.")


if __name__ == "__main__":
    main()