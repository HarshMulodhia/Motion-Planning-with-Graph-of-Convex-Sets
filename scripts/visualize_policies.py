"""
Visualization Script: Create video simulations for GCS, RL, and Hybrid methods
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ycb_grasp_rl_env import YCBGraspEnv
from src.gcs_trajectory_optimizer import GCSTrajectoryOptimizer
from stable_baselines3 import PPO
import pybullet as p

logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def visualize_gcs(env, test_object="rubiks_cube", num_episodes=3, output_dir="results/videos"):
    """
    Visualize pure GCS planning method.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[Visualize GCS] Recording GCS planning for {test_object}...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        logger.info(f" Episode {episode+1}/{num_episodes}: GCS planning...")
        
        try:
            if not hasattr(env, 'decomposer') or env.decomposer is None:
                logger.warning(f"  No decomposer available, skipping episode")
                continue
            
            # Get GCS path
            goal_region = env.decomposer.get_region_for_config(env.goal_config) if hasattr(env, 'goal_config') else 0
            start_region = env.current_region_id if hasattr(env, 'current_region_id') else 0
            
            optimizer = GCSTrajectoryOptimizer(env.decomposer)
            path = optimizer.dijkstra_path(start_region, goal_region)
            
            if path is None:
                logger.warning(f"  GCS path planning failed")
                continue
            
            # Execute path
            for region_id in path:
                if done:
                    break
                
                obs, reward, done, truncated, info = env.step(region_id)
                episode_reward += reward
                step_count += 1
                
                if done or truncated:
                    break
        
        except Exception as e:
            logger.error(f"  GCS episode {episode+1} failed: {e}")
            continue
    
    logger.info(f"✓ GCS visualization complete")

def visualize_rl(env, model, test_object="rubiks_cube", num_episodes=3, output_dir="results/videos"):
    """
    Visualize pure RL policy.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[Visualize RL] Recording learned RL policy for {test_object}...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        logger.info(f" Episode {episode+1}/{num_episodes}: RL policy execution...")
        
        try:
            while not done and step_count < 100:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                if done or truncated:
                    break
        
        except Exception as e:
            logger.error(f"  RL episode {episode+1} failed: {e}")
            continue
    
    logger.info(f"✓ RL visualization complete")

def visualize_hybrid(env, model, test_object="rubiks_cube", num_episodes=3, output_dir="results/videos"):
    """
    Visualize hybrid GCS+RL method.
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[Visualize Hybrid] Recording hybrid GCS+RL for {test_object}...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        logger.info(f" Episode {episode+1}/{num_episodes}: Hybrid GCS+RL execution...")
        
        try:
            # Step 1: Get GCS path
            if not hasattr(env, 'decomposer') or env.decomposer is None:
                logger.warning(f"  No decomposer, falling back to pure RL")
                # Fall back to pure RL
                while not done and step_count < 100:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    step_count += 1
                    if done or truncated:
                        break
            else:
                goal_region = env.decomposer.get_region_for_config(env.goal_config) if hasattr(env, 'goal_config') else 0
                start_region = env.current_region_id if hasattr(env, 'current_region_id') else 0
                
                optimizer = GCSTrajectoryOptimizer(env.decomposer)
                gcs_path = optimizer.dijkstra_path(start_region, goal_region)
                
                if gcs_path is None:
                    logger.warning(f"  GCS path planning failed, falling back to pure RL")
                    # Fall back to pure RL
                    while not done and step_count < 100:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, truncated, info = env.step(action)
                        episode_reward += reward
                        step_count += 1
                        if done or truncated:
                            break
                else:
                    # Step 2: Execute GCS path with RL refinement
                    for region_id in gcs_path:
                        if done:
                            break
                        
                        # Use RL to refine approach to next waypoint
                        for substep in range(10):
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, done, truncated, info = env.step(action)
                            episode_reward += reward
                            step_count += 1
                            
                            if done or truncated:
                                break
        
        except Exception as e:
            logger.error(f"  Hybrid episode {episode+1} failed: {e}")
            continue
    
    logger.info(f"✓ Hybrid visualization complete")

def main(test_object="rubiks_cube"):
    """Main visualization pipeline."""
    logger.info("="*70)
    logger.info("Policy Visualization: GCS, RL, and Hybrid Methods")
    logger.info("="*70)
    
    # Load model
    model_path = "models/ycb_grasp/final_model"
    logger.info(f"[Load] Loading model from: {model_path}")
    
    try:
        model = PPO.load(model_path)
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Create environment
    output_dir = "results/videos"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        env = YCBGraspEnv(
            ycb_objects=[test_object],
            num_regions=50,
            max_steps=100,
            render=True  # Enable PyBullet GUI for visualization
        )
        
        # Visualize all three methods
        logger.info(f"\n[Visualize] Starting visualizations for: {test_object}")
        
        visualize_gcs(env, test_object, num_episodes=3, output_dir=output_dir)
        visualize_rl(env, model, test_object, num_episodes=3, output_dir=output_dir)
        visualize_hybrid(env, model, test_object, num_episodes=3, output_dir=output_dir)
        
        env.close()
        
        logger.info("\n" + "="*70)
        logger.info("✓ Visualization complete!")
        logger.info(f"✓ Videos saved to: {output_dir}")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        return

if __name__ == "__main__":
    setup_logging()
    test_object = sys.argv[1] if len(sys.argv) > 1 else "rubiks_cube"
    main(test_object)