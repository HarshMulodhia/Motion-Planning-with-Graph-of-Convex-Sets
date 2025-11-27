#!/usr/bin/env python3
"""
Project initialization script.
Sets up directory structure and validates all imports.
"""

import os
import sys
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_directories():
    """Create project directory structure."""
    logger.info("Creating directory structure...")
    
    dirs = [
        'src',
        'scripts',
        'models/ycb_grasp/best',
        'logs/ycb_grasp',
        'results',
        'data',
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(f"  ✓ {d}")
    
    logger.info("✓ Directory structure created")


def create_init_files():
    """Create __init__.py files for packages."""
    logger.info("Creating __init__.py files...")
    
    init_files = [
        'src/__init__.py',
    ]
    
    for f in init_files:
        Path(f).touch()
        logger.info(f"  ✓ {f}")


def create_config():
    """Create default configuration file."""
    logger.info("Creating default configuration...")
    
    config = {
        'ycb_objects': [
            'rubiks_cube',
            'racquetball',
            'hammer',
            'plate',
            'windex_bottle'
        ],
        'num_regions': 20,
        'max_steps': 100,
        'num_envs': 2,
        'total_timesteps': 5000,
        'learning_rate': 3e-4,
        'algorithm': 'PPO'
    }
    
    os.makedirs('models/ycb_grasp', exist_ok=True)
    config_path = 'models/ycb_grasp/config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"  ✓ {config_path}")


def test_environment():
    """Test basic environment creation."""
    logger.info("Testing environment creation...")
    
    try:
        from src.ycb_grasp_rl_env import YCBGraspEnv
        
        env = YCBGraspEnv(
            ycb_objects=['rubiks_cube'],
            num_regions=10,
            max_steps=50,
            render=False
        )
        
        obs, _ = env.reset()
        logger.info(f"  ✓ Environment created")
        logger.info(f"  ✓ Observation shape: {obs.shape}")
        logger.info(f"  ✓ Action space: {env.action_space}")
        
        # Test a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
        
        logger.info(f"  ✓ Environment steps executed")
        env.close()
        
        logger.info("✓ Environment test passed")
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Environment test failed: {e}")
        return False


def main():
    """Run all initialization steps."""
    logger.info("=" * 70)
    logger.info("GCS + RL Grasp Project Initialization")
    logger.info("=" * 70)
    
    steps = [
        ("Directory Structure", create_directories),
        ("Init Files", create_init_files),
        ("Configuration", create_config),
        ("Environment Test", test_environment),
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        try:
            logger.info(f"\n[{step_name}]")
            result = step_func()
            if result is False:
                failed_steps.append(step_name)
        except Exception as e:
            logger.error(f"✗ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    # Summary
    logger.info("\n" + "=" * 70)
    if failed_steps:
        logger.error("✗ Initialization FAILED")
        logger.error(f"Failed steps: {', '.join(failed_steps)}")
        return 1
    else:
        logger.info("✓ Initialization SUCCESSFUL")
        logger.info("\nNext steps:")
        logger.info("1. Train model: python scripts/train_ycb_grasp.py")
        logger.info("2. Evaluate: python scripts/evaluate_grasp.py")
        logger.info("3. Compare: python scripts/compare_methods.py")
        logger.info("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
