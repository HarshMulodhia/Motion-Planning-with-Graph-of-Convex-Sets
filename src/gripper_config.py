"""
6D Gripper Configuration Space

Manages position (x, y, z) and orientation (roll, pitch, yaw) for gripper manipulator.
"""

import numpy as np
import pybullet as p
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfigSpace:
    """6D configuration space bounds for gripper."""

    x_range: Tuple[float, float] = (-1.0, 1.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    z_range: Tuple[float, float] = (0.0, 1.5)
    roll_range: Tuple[float, float] = (-np.pi, np.pi)
    pitch_range: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    yaw_range: Tuple[float, float] = (-np.pi, np.pi)

    def get_bounds(self) -> np.ndarray:
        """Get configuration space bounds as (6, 2) array."""
        return np.array([
            self.x_range,
            self.y_range,
            self.z_range,
            self.roll_range,
            self.pitch_range,
            self.yaw_range
        ])


class GripperConfiguration:
    """Manages 6D gripper configuration and collision detection."""

    def __init__(self, gripper_urdf_path: str):
        """
        Initialize gripper configuration.

        Args:
            gripper_urdf_path: Path to gripper URDF file

        Raises:
            FileNotFoundError: If URDF file not found
        """
        import os
        if not os.path.exists(gripper_urdf_path):
            raise FileNotFoundError(f"Gripper URDF not found: {gripper_urdf_path}")

        self.gripper_urdf = gripper_urdf_path
        self.config_space = ConfigSpace()
        logger.info(f"[Gripper] Initialized with URDF: {gripper_urdf_path}")

    def sample_random_config(self) -> np.ndarray:
        """
        Sample random configuration within bounds.

        Returns:
            6D configuration vector [x, y, z, roll, pitch, yaw]
        """
        bounds = self.config_space.get_bounds()
        config = np.array([
            np.random.uniform(bounds[0, 0], bounds[0, 1]),
            np.random.uniform(bounds[1, 0], bounds[1, 1]),
            np.random.uniform(bounds[2, 0], bounds[2, 1]),
            np.random.uniform(bounds[3, 0], bounds[3, 1]),
            np.random.uniform(bounds[4, 0], bounds[4, 1]),
            np.random.uniform(bounds[5, 0], bounds[5, 1]),
        ])
        return config

    def sample_multiple_configs(self, num_samples: int) -> np.ndarray:
        """
        Sample multiple configurations.

        Args:
            num_samples: Number of samples to generate

        Returns:
            (num_samples, 6) array of configurations
        """
        configs = np.array([self.sample_random_config() for _ in range(num_samples)])
        logger.info(f"[Gripper] Sampled {num_samples} configurations")
        return configs

    def is_collision_free(self, config: np.ndarray, gripper_id: int,
                         scene_objects: List[int], verbose: bool = False) -> bool:
        """
        Check if configuration is collision-free using PyBullet.

        Args:
            config: 6D configuration [x, y, z, roll, pitch, yaw]
            gripper_id: PyBullet body ID of gripper
            scene_objects: List of PyBullet body IDs to check collision against
            verbose: Print collision info

        Returns:
            True if collision-free
        """
        # Update gripper position
        pos = config[:3]
        quat = p.getQuaternionFromEuler(config[3:])
        p.resetBasePositionAndOrientation(gripper_id, pos, quat)

        # Check collisions
        for obj_id in scene_objects:
            contact_points = p.getContactPoints(gripper_id, obj_id)
            if contact_points:
                if verbose:
                    logger.debug(f"Collision detected between gripper and object {obj_id}")
                return False

        return True

    def config_to_pose(self, config: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 6D config to position and quaternion.

        Args:
            config: 6D configuration [x, y, z, roll, pitch, yaw]

        Returns:
            Tuple of (position [3], quaternion [4])
        """
        pos = config[:3]
        quat = p.getQuaternionFromEuler(config[3:])
        return pos, quat

    def pose_to_config(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Convert position and quaternion to 6D config.

        Args:
            pos: Position [x, y, z]
            quat: Quaternion [x, y, z, w]

        Returns:
            6D configuration
        """
        euler = p.getEulerFromQuaternion(quat)
        config = np.concatenate([pos, euler])
        return config

    def is_within_bounds(self, config: np.ndarray) -> bool:
        """
        Check if configuration is within configuration space bounds.

        Args:
            config: 6D configuration

        Returns:
            True if within bounds
        """
        bounds = self.config_space.get_bounds()
        for i, (lower, upper) in enumerate(bounds):
            if config[i] < lower or config[i] > upper:
                return False
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Gripper Configuration module loaded successfully")
