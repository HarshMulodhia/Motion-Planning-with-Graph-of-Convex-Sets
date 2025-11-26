"""
Create Gripper URDF File

Generates a simple parallel-jaw gripper URDF model for simulation.
"""

import os
import logging

logger = logging.getLogger(__name__)

GRIPPER_URDF = """<?xml version="1.0" ?>
<robot name="simple_gripper">
  <link name="base_link">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.0002" iyy="0.0002" izz="0.0002" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.04 0.04 0.08"/>
      </geometry>
      <material name="gray">
        <color rgba="0.6 0.6 0.6 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.04 0.04 0.08"/>
      </geometry>
    </collision>
  </link>

  <link name="left_finger">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" iyy="0.00001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.02 0.01 0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.01 0.05"/>
      </geometry>
    </collision>
  </link>

  <link name="right_finger">
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00001" iyy="0.00001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <visual>
      <geometry>
        <box size="0.02 0.01 0.05"/>
      </geometry>
      <material name="blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.02 0.01 0.05"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_finger_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="left_finger"/>
    <origin xyz="-0.03 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="0.05" effort="10" velocity="1"/>
  </joint>

  <joint name="right_finger_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="right_finger"/>
    <origin xyz="0.03 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-0.05" upper="0" effort="10" velocity="1"/>
  </joint>
</robot>
"""


def create_gripper_urdf(output_dir: str = "data") -> str:
    """
    Create and save gripper URDF file.

    Args:
        output_dir: Directory to save URDF file

    Returns:
        Path to created URDF file

    Raises:
        OSError: If file write fails
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        urdf_path = os.path.join(output_dir, "gripper.urdf")

        with open(urdf_path, "w") as f:
            f.write(GRIPPER_URDF)

        logger.info(f"✓ Created gripper URDF: {urdf_path}")
        return urdf_path
    except OSError as e:
        logger.error(f"Failed to create gripper URDF: {e}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        path = create_gripper_urdf()
        print(f"Gripper URDF created at: {path}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
