"""
Download and install the official UR10e robot description locally.

Source:
    robot_descriptions -> Universal_Robots_ROS2_Description

Destination:
    python/robot/description/ur_description/

Behavior:
    - Uses the official robot_descriptions package.
    - Downloads the version pinned by that package.
    - Copies the complete description into the project.
    - Removes nested Git tracking from the copied files.
    - Records the upstream commit.
    - Does not overwrite an existing description.
    - Does not modify the system's robot_descriptions cache.

Requirements:
    pip install robot-descriptions
    Git must be available.
"""

import importlib
import os
import shutil
import tempfile
from pathlib import Path

from git import Repo


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

DESCRIPTION_DIR = SCRIPT_DIR / "description"

ROBOT_NAME = "ur10e"

SOURCE_REPOSITORY = (
    "https://github.com/UniversalRobots/"
    "Universal_Robots_ROS2_Description"
)

TARGET_DIR = DESCRIPTION_DIR / "ur_description"

VERSION_FILE = DESCRIPTION_DIR / "UPSTREAM_VERSION.txt"


# --------------------------------------------------
# Robot description setup
# --------------------------------------------------

def setup_robot_description():

    print("\n========================================")
    print("UR10e ROBOT DESCRIPTION SETUP")
    print("========================================\n")

    # Existing installation protection
    if TARGET_DIR.exists():

        if not TARGET_DIR.is_dir():
            raise RuntimeError(
                f"Target exists but is not a directory: {TARGET_DIR}"
            )

        print("[INFO] Robot description already exists.")
        print(f"[INFO] Location: {TARGET_DIR}")

        if VERSION_FILE.exists():
            print("\n[INFO] Installed version:")
            print(VERSION_FILE.read_text(encoding="utf-8"))

        print("[INFO] No files modified.")
        return

    DESCRIPTION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Use a temporary cache so that the system cache
    # and the project installation remain independent.
    with tempfile.TemporaryDirectory(
        prefix="ur10e_description_"
    ) as temp_dir:

        old_cache = os.environ.get(
            "ROBOT_DESCRIPTIONS_CACHE"
        )

        temporary_cache = Path(temp_dir) / "cache"

        try:
            os.environ["ROBOT_DESCRIPTIONS_CACHE"] = str(
                temporary_cache
            )

            print("[1] Downloading official UR10e description...")

            description = importlib.import_module(
                "robot_descriptions.ur10e_description"
            )

            source_dir = Path(
                description.REPOSITORY_PATH
            ).resolve()

        finally:
            if old_cache is None:
                os.environ.pop(
                    "ROBOT_DESCRIPTIONS_CACHE",
                    None,
                )
            else:
                os.environ[
                    "ROBOT_DESCRIPTIONS_CACHE"
                ] = old_cache

        if not source_dir.is_dir():
            raise RuntimeError(
                f"Description download failed: {source_dir}"
            )

        # Read the exact upstream commit before stripping Git.
        upstream_commit = Repo(
            str(source_dir)
        ).head.commit.hexsha

        print(f"[INFO] Commit: {upstream_commit}")

        # Prepare the copy before making it the final installation.
        staging_dir = DESCRIPTION_DIR / "_ur_description_setup"

        if staging_dir.exists():
            raise RuntimeError(
                f"Temporary setup directory already exists: {staging_dir}"
            )

        try:
            print("\n[2] Copying description into project...")

            shutil.copytree(
                source_dir,
                staging_dir,
                ignore=shutil.ignore_patterns(
                    ".git",
                    ".github",
                    ".gitignore",
                    ".gitattributes",
                    ".pre-commit-config.yaml",
                ),
            )

            print("\n[3] Git metadata excluded during copy.")

            print("\n[4] Finalizing description...")

            staging_dir.rename(TARGET_DIR)

        finally:
            # Clean up an incomplete staging copy.
            if staging_dir.exists():
                shutil.rmtree(staging_dir)

        # Save provenance metadata.
        version_info = (
            f"Robot: {ROBOT_NAME}\n"
            f"Source: {SOURCE_REPOSITORY}\n"
            f"Commit: {upstream_commit}\n"
            f"Description: robot_descriptions.ur10e_description\n"
        )

        VERSION_FILE.write_text(
            version_info,
            encoding="utf-8",
        )

    print("\n========================================")
    print("ROBOT DESCRIPTION SETUP COMPLETE")
    print("========================================")

    print(f"\nRobot: {ROBOT_NAME}")
    print(f"Location: {TARGET_DIR}")
    print(f"Version file: {VERSION_FILE}")
    print(f"Upstream commit: {upstream_commit}")
    print("\nNested Git tracking: REMOVED")
    print("Existing application files: UNCHANGED")


# --------------------------------------------------
# Entry point
# --------------------------------------------------

if __name__ == "__main__":
    setup_robot_description()