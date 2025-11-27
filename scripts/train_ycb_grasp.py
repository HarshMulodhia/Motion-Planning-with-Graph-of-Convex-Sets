"""
Train YCB Grasp RL Policy

Trains PPO or SAC agent on YCB object grasping task using parallel vectorized environments.
Saves trained model, config, and logs to models/ directory.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

logger = logging.getLogger(__name__)

# Try to import custom environment - will fail gracefully if not available
try:
    from src.ycb_grasp_rl_env import YCBGraspEnv
    YCB_ENV_AVAILABLE = True
except ImportError:
    logger.warning("YCBGraspEnv not available - using mock for testing")
    YCB_ENV_AVAILABLE = False


# Configuration
CONFIG = {
    'ycb_objects': [
        "rubiks_cube", "racquetball", "hammer", "plate", "windex_bottle", 
        "spoon", "sponge", "scissors", "mug", "power_drill", 
    ],
    'num_regions': 50,
    'max_steps': 100,
    'num_envs': 4,
    'total_timesteps': 1_000,
    'learning_rate': 3e-4,
    'algorithm': 'PPO',  # or 'SAC'
}


def make_env(env_id: int):
    """
    Factory function for parallel environments.

    Args:
        env_id: Environment index (used for rendering only first env)

    Returns:
        Environment initialization function
    """
    def _init():
        if not YCB_ENV_AVAILABLE:
            raise RuntimeError("YCBGraspEnv not available")

        env = YCBGraspEnv(
            ycb_objects=CONFIG['ycb_objects'],
            num_regions=CONFIG['num_regions'],
            max_steps=CONFIG['max_steps'],
            render=False  # Render only first env
        )
        return env

    return _init


def setup_directories():
    """Create necessary output directories."""
    dirs = ['models/ycb_grasp', 'logs/ycb_grasp', 'results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        logger.info(f"✓ Created directory: {d}")


def train_model():
    """
    Main training function.

    Returns:
        Trained model
    """
    logger.info("=" * 70)
    logger.info("YCB Grasp Policy Training")
    logger.info("=" * 70)

    # Setup directories
    setup_directories()

    # Create vectorized environments
    logger.info("[Training] Creating vectorized environments...")
    try:
        env = make_vec_env(make_env(0), n_envs=CONFIG['num_envs'])
        logger.info(f"✓ Created {CONFIG['num_envs']} parallel environments")
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        raise

    # Setup callbacks
    logger.info("[Training] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='models/ycb_grasp',
        name_prefix='ppo_model'
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10000,
        n_eval_episodes=10,
        best_model_save_path='models/ycb_grasp/best',
        log_path='logs/ycb_grasp'
    )

    # Create model
    logger.info(f"[Training] Creating {CONFIG['algorithm']} model...")
    if CONFIG['algorithm'] == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=CONFIG['learning_rate'],
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log='logs/ycb_grasp'
        )
    elif CONFIG['algorithm'] == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=CONFIG['learning_rate'],
            buffer_size=int(1e6),
            train_freq=(1, 'step'),
            gamma=0.99,
            tau=0.005,
            verbose=1,
            tensorboard_log='logs/ycb_grasp'
        )
    else:
        raise ValueError(f"Unknown algorithm: {CONFIG['algorithm']}")

    # Train
    logger.info("[Training] Starting training loop...")
    logger.info(f"Total timesteps: {CONFIG['total_timesteps']:,}")

    try:
        model.learn(
            total_timesteps=CONFIG['total_timesteps'],
            #callback=[checkpoint_callback, eval_callback],
            callback=[checkpoint_callback],
            progress_bar=True
        )
        logger.info("✓ Training completed successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        env.close()

    # Save final model
    logger.info("[Training] Saving final model...")
    model_path = 'models/ycb_grasp/final_model'
    model.save(model_path)
    logger.info(f"✓ Model saved to: {model_path}")

    # Save config
    config_path = 'models/ycb_grasp/config.json'
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    logger.info(f"✓ Config saved to: {config_path}")

    logger.info("=" * 70)
    logger.info("✓ Training complete!")
    logger.info("=" * 70)

    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        train_model()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
