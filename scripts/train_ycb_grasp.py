# scripts/train_ycb_grasp.py
import os
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from src.ycb_grasp_rl_env import YCBGraspEnv
import json

# Configuration
CONFIG = {
    'ycb_objects': [
        '001_chips_can',
        '002_master_chef_can',
        '003_cracker_box',
        '004_sugar_box',
        '005_tomato_soup_can',
        '006_mustard_bottle',
        '007_tuna_fish_can',
        '008_pudding_box',
        '009_gelatin_box',
        '010_potted_meat_can'
    ],
    'num_regions': 50,
    'max_steps': 100,
    'num_envs': 4,
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'algorithm': 'PPO'  # or 'SAC'
}

def make_env(env_id: int):
    """Factory function for parallel environments"""
    def _init():
        env = YCBGraspEnv(
            ycb_objects=CONFIG['ycb_objects'],
            num_regions=CONFIG['num_regions'],
            max_steps=CONFIG['max_steps'],
            render=(env_id == 0)  # Render only first env
        )
        return env
    return _init

# Create vectorized environments
print("[Training] Creating vectorized environments...")
env = make_vec_env(
    make_env(0),
    n_envs=CONFIG['num_envs'],
    start_method='fork'  # Use fork for Unix/Linux
)

# Create output directory
os.makedirs('models/ycb_grasp', exist_ok=True)
os.makedirs('logs/ycb_grasp', exist_ok=True)

# Setup callbacks
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

# Train
print("[Training] Starting training loop...")
print(f"Total timesteps: {CONFIG['total_timesteps']:,}")

model.learn(
    total_timesteps=CONFIG['total_timesteps'],
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True
)

# Save final model
model.save('models/ycb_grasp/final_model')
print("[Training] ✓ Training complete! Model saved to models/ycb_grasp/final_model")

# Save config
with open('models/ycb_grasp/config.json', 'w') as f:
    json.dump(CONFIG, f, indent=2)
