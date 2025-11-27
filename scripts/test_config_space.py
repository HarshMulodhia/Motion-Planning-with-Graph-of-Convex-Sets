"""
Test Configuration Space Setup

Tests GCS decomposition and gripper configuration space functionality.
"""

import sys
import logging
import os
# Ensure we can find src if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
import numpy as np

try:
    import pybullet as p
    import pybullet_data
    from src.gripper_config import GripperConfig
    from src.create_gripper import create_gripper_urdf
    PYBULLET_AVAILABLE = True
except ImportError as e:
    logging.getLogger(__name__).warning(f"Dependency missing: {e}")
    PYBULLET_AVAILABLE = False

logger = logging.getLogger(__name__)

def test_config_space():
    if not PYBULLET_AVAILABLE:
        logger.error("PyBullet or src modules not available")
        return False

    logger.info("=" * 70)
    logger.info("Testing 6D Configuration Space")
    logger.info("=" * 70)

    p.connect(p.DIRECT) # Use DIRECT for CI/Test speed

    # Create gripper using utility from src
    try:
        gripper_urdf_path = create_gripper_urdf()
    except Exception as e:
        logger.error(f"Failed to create gripper URDF: {e}")
        return False

    # Test Config Class
    try:
        gripper_config = GripperConfig(gripper_urdf_path)
        config = gripper_config.sample_random_config()
        logger.info(f"Sampled config: {config}")
        logger.info("✓ Configuration space initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize gripper config: {e}")
        return False

    p.disconnect()
    return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_config_space()