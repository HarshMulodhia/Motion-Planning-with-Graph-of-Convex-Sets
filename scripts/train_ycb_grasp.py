# scripts/train_ycb_grasp.py (CORRECTED)
"""
Train YCB Grasp RL Policy using PPO or SAC.
"""

import os
import json
import logging
from pathlib import Path
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

logger = logging.getLogger(__name__)

try:
    from src.ycb_grasp_rl_env import YCBGraspEnv
    YCB_ENV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"YCBGraspEnv not available: {e}")
    YCB_ENV_AVAILABLE = False


# Training configuration
CONFIG = {
    'ycb_objects': [
        "rubiks_cube", "racquetball", "hammer", "plate", "windex_bottle", 
        "spoon", "sponge", "scissors", "mug", "power_drill", 
    ],
    'num_regions': 50,
    'max_steps': 100,
    'num_envs': 4,
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'algorithm': 'PPO',  # or 'SAC'
}


def make_env(env_id: int):
    """
    Factory function for creating parallel environments.
    
    Args:
        env_id: Environment index
        
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
            render=False
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
    
    # Setup
    setup_directories()
    
    # Create vectorized environments
    logger.info("[Training] Creating vectorized environments...")
    
    try:
        # Create environment factory for each worker
        env_fns = [make_env(i) for i in range(CONFIG['num_envs'])]
        
        # Use SubprocVecEnv for true parallelization
        from stable_baselines3.common.vec_env import SubprocVecEnv
        env = SubprocVecEnv(env_fns)
        
        logger.info(f"✓ Created {CONFIG['num_envs']} parallel environments")
    except Exception as e:
        logger.error(f"Failed to create environments: {e}")
        raise
    
    # Setup callback
    logger.info("[Training] Setting up callbacks...")
    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path='models/ycb_grasp',
        name_prefix='ycb_model'
    )
    
    # Create model
    logger.info(f"[Training] Creating {CONFIG['algorithm']} model...")
    
    try:
        if CONFIG['algorithm'] == 'PPO':
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=CONFIG['learning_rate'],
                n_steps=512,
                batch_size=64,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log='logs/ycb_grasp'
            )
        elif CONFIG['algorithm'] == 'SAC':
            model = SAC(
                'MlpPolicy',
                env,
                learning_rate=CONFIG['learning_rate'],
                buffer_size=int(1e5),
                train_freq=(1, 'step'),
                gamma=0.99,
                tau=0.005,
                verbose=1,
                tensorboard_log='logs/ycb_grasp'
            )
        else:
            raise ValueError(f"Unknown algorithm: {CONFIG['algorithm']}")
        
        logger.info("✓ Model created")
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        env.close()
        raise
    
    # Training loop
    logger.info("[Training] Starting training loop...")
    logger.info(f"Total timesteps: {CONFIG['total_timesteps']:,}")
    logger.info(f"Number of environments: {CONFIG['num_envs']}")
    
    try:
        model.learn(
            total_timesteps=CONFIG['total_timesteps'],
            callback=[checkpoint_callback],
            progress_bar=True,
            log_interval=10
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
    logger.info("✓ Training Complete!")
    logger.info("=" * 70)
    
    return model


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        model = train_model()
        logger.info("✓ Training pipeline completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
