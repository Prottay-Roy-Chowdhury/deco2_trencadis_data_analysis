"""
UR10e Robot Model
================

Reusable robot description and kinematic model loader.

Responsibilities
----------------
1. Register the local Universal Robots description.
2. Expand the UR10e Xacro.
3. Generate the URDF.
4. Load the robot into Robotics Toolbox.
5. Expose robot properties and kinematic operations.
6. Validate externally supplied joint configurations.

Not responsible for
-------------------
- Defining a HOME configuration.
- Motion planning.
- Collision checking.
- Visualization.
- Connecting to the real robot.
- Executing robot motion.

Dependencies
------------
roboticstoolbox-python
spatialmath-python
xacrodoc
numpy

The local robot description must already exist at:

python/robot/description/ur_description/

All joint configurations are expressed in radians.

No robot movement commands are used.
"""

from pathlib import Path

import numpy as np

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot

from xacrodoc import XacroDoc
from xacrodoc import packages


class URRobotModel:
    """
    Reusable Universal Robots model loader.

    The class manages the robot description and kinematics.

    Robot configurations are supplied externally by:
        - Configuration files
        - Application composers
        - Motion planning services
        - Robot state interfaces

    The class does not define or automatically apply HOME.
    """

    # --------------------------------------------------
    # INITIALIZATION
    # --------------------------------------------------

    def __init__(
        self,
        robot_type="ur10e",
        robot_name=None,
        tool_frame="tool0",
        description_dir=None,
        generated_dir=None,
        verbose=True,
    ):

        self.robot_type = robot_type

        self.robot_name = (
            robot_name
            if robot_name is not None
            else robot_type
        )

        self.tool_frame = tool_frame

        self.verbose = verbose

        # ----------------------------------------------
        # Directory configuration
        # ----------------------------------------------

        self.module_dir = Path(__file__).resolve().parent

        if description_dir is None:

            self.description_dir = (
                self.module_dir
                / "description"
                / "ur_description"
            )

        else:

            self.description_dir = Path(
                description_dir
            ).resolve()

        if generated_dir is None:

            self.generated_dir = (
                self.module_dir
                / "description"
                / "_generated"
            )

        else:

            self.generated_dir = Path(
                generated_dir
            ).resolve()

        # ----------------------------------------------
        # Description paths
        # ----------------------------------------------

        self.xacro_path = (
            self.description_dir
            / "urdf"
            / "ur.urdf.xacro"
        )

        self.urdf_path = (
            self.generated_dir
            / f"{self.robot_name}.urdf"
        )

        # ----------------------------------------------
        # Robot model
        # ----------------------------------------------

        self.robot = None

        self._loaded = False

    # --------------------------------------------------
    # INTERNAL UTILITIES
    # --------------------------------------------------

    def _log(self, message):

        if self.verbose:
            print(message)

    def _require_loaded(self):
        """
        Ensure the robot model has been loaded.
        """

        if not self._loaded or self.robot is None:

            raise RuntimeError(
                "Robot model is not loaded. "
                "Call load() first."
            )

    # --------------------------------------------------
    # DESCRIPTION VALIDATION
    # --------------------------------------------------

    def validate_description(self):
        """
        Check that the local robot description exists.
        """

        if not self.description_dir.is_dir():

            raise FileNotFoundError(
                "Robot description directory not found:\n"
                f"{self.description_dir}"
            )

        if not self.xacro_path.is_file():

            raise FileNotFoundError(
                "UR Xacro file not found:\n"
                f"{self.xacro_path}"
            )

        return True

    # --------------------------------------------------
    # REGISTER LOCAL ROS PACKAGE
    # --------------------------------------------------

    def register_description(self):
        """
        Register the local ur_description package.

        This allows Xacro to resolve package references
        from the locally stored robot description.

        The global robot_descriptions cache is not used.
        """

        self.validate_description()

        packages.update_package_cache(
            {
                "ur_description": str(
                    self.description_dir
                )
            }
        )

        self._log(
            "[OK] Registered local package: ur_description"
        )

        self._log(
            f"     Path: {self.description_dir}"
        )

        return self.description_dir

    # --------------------------------------------------
    # XACRO EXPANSION
    # --------------------------------------------------

    def generate_urdf(self):
        """
        Expand the official Universal Robots Xacro.

        Save the generated URDF inside the project.
        """

        self.validate_description()

        self.generated_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        xacro_args = {
            "ur_type": self.robot_type,
            "name": self.robot_name,
        }

        self._log(
            f"[INFO] Expanding {self.robot_type} Xacro..."
        )

        document = XacroDoc.from_file(
            str(self.xacro_path),
            subargs=xacro_args,
        )

        urdf_content = document.to_urdf_string()

        self.urdf_path.write_text(
            urdf_content,
            encoding="utf-8"
        )

        self._log(
            "[OK] Generated URDF:"
        )

        self._log(
            f"     {self.urdf_path}"
        )

        return self.urdf_path

    # --------------------------------------------------
    # LOAD ROBOT MODEL
    # --------------------------------------------------

    def load(
        self,
        regenerate_urdf=True,
    ):
        """
        Load the robot description into Robotics Toolbox.

        Parameters
        ----------
        regenerate_urdf : bool
            True:
                Expand Xacro before loading.

            False:
                Use the previously generated URDF.

        Returns
        -------
        robot : URDFRobot
            Loaded Robotics Toolbox robot model.

        Notes
        -----
        Loading does not set a HOME configuration.
        """

        self._log(
            "\n========================================"
        )

        self._log(
            "UR ROBOT MODEL LOADER"
        )

        self._log(
            "========================================"
        )

        # ----------------------------------------------
        # Register local description
        # ----------------------------------------------

        self._log(
            "\n[1] Registering local description..."
        )

        self.register_description()

        # ----------------------------------------------
        # Generate URDF
        # ----------------------------------------------

        self._log(
            "\n[2] Preparing URDF..."
        )

        if regenerate_urdf:

            self.generate_urdf()

        elif not self.urdf_path.is_file():

            raise FileNotFoundError(
                "Generated URDF not found:\n"
                f"{self.urdf_path}\n"
                "Use regenerate_urdf=True."
            )

        # ----------------------------------------------
        # Load Robotics Toolbox model
        # ----------------------------------------------

        self._log(
            "\n[3] Loading Robotics Toolbox model..."
        )

        self.robot = URDFRobot(
            str(self.urdf_path)
        )

        self._loaded = True

        self._log(
            "[OK] Robot model loaded successfully."
        )

        self._log(
            f"     Robot: {self.robot.name}"
        )

        self._log(
            f"     Joints: {self.robot.n}"
        )

        self._log(
            f"     Tool frame: {self.tool_frame}"
        )

        self._log(
            "\nNo robot configuration was applied."
        )

        return self.robot

    # --------------------------------------------------
    # ROBOT PROPERTIES
    # --------------------------------------------------

    @property
    def is_loaded(self):
        """
        Return whether the model is loaded.
        """

        return self._loaded

    @property
    def n_joints(self):
        """
        Number of robot joints.
        """

        self._require_loaded()

        return self.robot.n

    @property
    def joint_limits(self):
        """
        Return the robot joint limits.

        Returns
        -------
        numpy.ndarray
            Shape: (2, n)

            Row 0: lower limits
            Row 1: upper limits

            Units: radians

        Notes
        -----
        These limits come from the loaded URDF.
        They are not a substitute for controller safety
        limits or installation-specific restrictions.
        """

        self._require_loaded()

        return np.asarray(
            self.robot.qlim,
            dtype=float
        ).copy()

    @property
    def joint_names(self):
        """
        Return the ordered joint names.
        """

        self._require_loaded()

        return [
            link.name
            for link in self.robot.links
            if link.isjoint
        ]

    # --------------------------------------------------
    # JOINT CONFIGURATION VALIDATION
    # --------------------------------------------------

    def validate_configuration(
        self,
        q,
        check_limits=False,
    ):
        """
        Validate an externally supplied joint configuration.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        check_limits : bool
            Whether to check the URDF joint limits.

        Returns
        -------
        numpy.ndarray
            Validated joint configuration.

        Notes
        -----
        This function does not change the robot state.

        It does not perform collision checking.
        """

        self._require_loaded()

        q = np.asarray(
            q,
            dtype=float
        )

        # ----------------------------------------------
        # Validate dimensions
        # ----------------------------------------------

        if q.shape != (self.n_joints,):

            raise ValueError(
                f"Expected {self.n_joints} joint values, "
                f"received shape {q.shape}"
            )

        # ----------------------------------------------
        # Validate numerical values
        # ----------------------------------------------

        if not np.all(np.isfinite(q)):

            raise ValueError(
                "Joint configuration contains "
                "NaN or infinite values."
            )

        # ----------------------------------------------
        # Optional joint limit validation
        # ----------------------------------------------

        if check_limits:

            limits = self.joint_limits

            lower = limits[0, :]
            upper = limits[1, :]

            if not np.all(np.isfinite(limits)):

                raise ValueError(
                    "URDF joint limits are not fully defined."
                )

            tolerance = 1e-9

            if np.any(q < lower - tolerance):

                raise ValueError(
                    "Joint configuration violates "
                    "lower joint limits."
                )

            if np.any(q > upper + tolerance):

                raise ValueError(
                    "Joint configuration violates "
                    "upper joint limits."
                )

        return q.copy()

    # --------------------------------------------------
    # FORWARD KINEMATICS
    # --------------------------------------------------

    def fk(
        self,
        q,
        end=None,
    ):
        """
        Compute forward kinematics.

        Parameters
        ----------
        q : array-like
            Joint configuration in radians.

        end : str, optional
            End-effector frame.

            Defaults to self.tool_frame.

        Returns
        -------
        spatialmath.SE3
            End-effector transformation.

        Notes
        -----
        The end-effector frame is selected explicitly
        because the URDF contains multiple terminal
        frames.

        This does not modify the robot configuration.
        """

        self._require_loaded()

        q = self.validate_configuration(q)

        if end is None:
            end = self.tool_frame

        return self.robot.fkine(
            q,
            end=end
        )

    # --------------------------------------------------
    # CONFIGURATION APPLICATION
    # --------------------------------------------------

    def set_configuration(
        self,
        q,
        check_limits=False,
    ):
        """
        Apply a joint configuration to the local model.

        Parameters
        ----------
        q : array-like
            Joint angles in radians.

        check_limits : bool
            Check URDF joint limits before applying.

        Notes
        -----
        This updates the local Robotics Toolbox model
        configuration.

        It does not move the physical robot.
        """

        self._require_loaded()

        q = self.validate_configuration(
            q,
            check_limits=check_limits
        )

        self.robot.q = q.copy()

        return q

    # --------------------------------------------------
    # GET CURRENT MODEL CONFIGURATION
    # --------------------------------------------------

    def get_configuration(self):
        """
        Return the current local model configuration.

        Notes
        -----
        This is the configuration stored by the
        Robotics Toolbox model.

        It is NOT the physical robot's actual joints.
        """

        self._require_loaded()

        return np.asarray(
            self.robot.q,
            dtype=float
        ).copy()

    # --------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------

    def summary(self):
        """
        Print a summary of the loaded robot model.
        """

        self._require_loaded()

        print(
            "\n========================================"
        )

        print(
            "ROBOT MODEL SUMMARY"
        )

        print(
            "========================================"
        )

        print(
            f"Robot name       : {self.robot.name}"
        )

        print(
            f"Robot type       : {self.robot_type}"
        )

        print(
            f"Number of joints : {self.n_joints}"
        )

        print(
            f"Tool frame       : {self.tool_frame}"
        )

        print(
            f"Description      : {self.description_dir}"
        )

        print(
            f"Generated URDF   : {self.urdf_path}"
        )

        print(
            "\nJoint names:"
        )

        for index, name in enumerate(
            self.joint_names,
            start=1
        ):

            print(
                f"  Joint {index}: {name}"
            )

        print(
            "\nJoint limits (degrees):"
        )

        limits_deg = np.rad2deg(
            self.joint_limits
        )

        for index in range(
            self.n_joints
        ):

            print(
                f"  Joint {index + 1}: "
                f"{limits_deg[0, index]:9.3f} "
                f"to "
                f"{limits_deg[1, index]:9.3f}"
            )

        print(
            "\nNo HOME configuration is defined "
            "inside the model."
        )

        print(
            "========================================"
        )