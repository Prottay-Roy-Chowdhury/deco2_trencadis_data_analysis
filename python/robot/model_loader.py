# --------------------------------------------------
# Ensure Python UTF-8 mode on Windows
# --------------------------------------------------

import os
import sys
import subprocess

if sys.platform == "win32" and not sys.flags.utf8_mode:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    result = subprocess.call(
        [sys.executable, *sys.argv],
        env=env,
    )
    sys.exit(result)

# --------------------------------------------------
# Imports
# --------------------------------------------------

from pathlib import Path

import numpy as np
import swift
import xacrodoc as xd
import xacrodoc.packages as xpkg

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot

# --------------------------------------------------
# Runtime configuration
# --------------------------------------------------

SHOW_VISUALIZATION = True
SAVE_GENERATED_URDF = True

# --------------------------------------------------
# Paths
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

DESCRIPTION_DIR = SCRIPT_DIR / "description"
UR_DESCRIPTION_DIR = DESCRIPTION_DIR / "ur_description"

URDF_XACRO_PATH = UR_DESCRIPTION_DIR / "urdf" / "ur.urdf.xacro"

GENERATED_DIR = DESCRIPTION_DIR / "_generated"
GENERATED_URDF_PATH = GENERATED_DIR / "ur10e.urdf"

# --------------------------------------------------
# Robot configuration
# --------------------------------------------------

ROBOT_TYPE = "ur10e"

XACRO_ARGS = {
    "ur_type": ROBOT_TYPE,
    "name": ROBOT_TYPE,
}

# Your reference home configuration for offline testing only
HOME_Q_DEG = np.array([
    135.00,
    -99.45,
    126.61,
    -114.71,
    -90.00,
    0.00,
], dtype=float)

# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def print_header(title: str):
    print("\n" + "=" * 54)
    print(title)
    print("=" * 54)

def print_subheader(title: str):
    print("\n" + "-" * 54)
    print(title)
    print("-" * 54)

def print_vector6(title: str, values, unit=""):
    print_subheader(title)
    labels = ["X", "Y", "Z", "RX", "RY", "RZ"]
    for label, value in zip(labels, values):
        if unit:
            print(f"{label:<3} : {value:>12.6f} {unit}")
        else:
            print(f"{label:<3} : {value:>12.6f}")

# --------------------------------------------------
# Xacro / URDF generation
# --------------------------------------------------

def register_local_ur_description():
    print("\n[1] Registering local UR description package...")

    if not UR_DESCRIPTION_DIR.is_dir():
        raise FileNotFoundError(
            f"Local UR description directory not found:\n{UR_DESCRIPTION_DIR}\n"
            "Run setup_robot_description.py first."
        )

    # Register local ROS-style package name so $(find ur_description) resolves
    xpkg.update_package_cache({
        "ur_description": str(UR_DESCRIPTION_DIR)
    })

    print(f"[OK] Registered package: ur_description")
    print(f"      Path: {UR_DESCRIPTION_DIR}")

def expand_xacro_to_urdf():
    print("\n[2] Expanding UR10e Xacro...")

    if not URDF_XACRO_PATH.is_file():
        raise FileNotFoundError(
            f"URDF Xacro file not found:\n{URDF_XACRO_PATH}"
        )

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    document = xd.XacroDoc.from_file(
        str(URDF_XACRO_PATH),
        subargs=XACRO_ARGS,
    )

    urdf_xml = document.to_urdf_string()

    if SAVE_GENERATED_URDF:
        GENERATED_URDF_PATH.write_text(
            urdf_xml,
            encoding="utf-8",
        )
        print(f"[OK] Generated URDF saved to:\n      {GENERATED_URDF_PATH}")
    else:
        print("[OK] URDF generated in memory.")

    return urdf_xml

# --------------------------------------------------
# Robot loading
# --------------------------------------------------

def load_robot():
    register_local_ur_description()
    urdf_xml = expand_xacro_to_urdf()

    print("\n[3] Loading robot model into Robotics Toolbox...")

    if SAVE_GENERATED_URDF:
        robot = URDFRobot(str(GENERATED_URDF_PATH))
    else:
        # fallback: save temporary local file anyway for compatibility
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        GENERATED_URDF_PATH.write_text(urdf_xml, encoding="utf-8")
        robot = URDFRobot(str(GENERATED_URDF_PATH))

    if robot.n != 6:
        raise RuntimeError(
            f"Expected 6 robot joints, but found {robot.n}."
        )

    print("[OK] Robot model loaded successfully.")
    return robot

# --------------------------------------------------
# Inspection
# --------------------------------------------------

def inspect_robot(robot, q_rad):
    print_header("UR10e LOCAL MODEL INSPECTION")

    print(robot)

    print_subheader("Robot summary")
    print(f"Name               : {robot.name}")
    print(f"Number of joints   : {robot.n}")
    print(f"Xacro source       : {URDF_XACRO_PATH}")
    print(f"Generated URDF     : {GENERATED_URDF_PATH}")

    print_subheader("Joint configuration used for test")
    for i, q_deg in enumerate(np.rad2deg(q_rad), start=1):
        print(f"Joint {i:<2}         : {q_deg:>10.4f} deg")

    print("\n[4] Computing forward kinematics...")

    T = robot.fkine(q_rad, end="tool0")

    print("\nEnd-effector transformation:")
    print(T)

    print_vector6(
        "End-effector pose (translation + RPY approximation not used here)",
        [T.t[0], T.t[1], T.t[2], 0.0, 0.0, 0.0],
    )

    print_subheader("End-effector rotation matrix")
    print(np.round(T.R, 6))

    print("[OK] Forward kinematics computed successfully.")

# --------------------------------------------------
# Visualization
# --------------------------------------------------

def visualize_robot(robot, q_rad):
    print("\n[5] Launching Swift visualization...")

    robot.q = q_rad.copy()

    env = swift.Swift()
    env.launch(realtime=True)
    env.add(robot)

    # Step once so the first pose is shown
    env.step(0.01)

    print("[OK] Swift launched.")
    print("Close the Swift window/browser session to finish the script.")

    env.hold()

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print_header("UR10e LOCAL MODEL TEST")

    robot = load_robot()

    q_rad = np.deg2rad(HOME_Q_DEG)

    inspect_robot(robot, q_rad)

    if SHOW_VISUALIZATION:
        visualize_robot(robot, q_rad)

    print("\nTest completed successfully.")

if __name__ == "__main__":
    main()