"""
Visualize pure GCS method - No RL dependency
"""
import sys
import os
import logging
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ycb_grasp_rl_env import YCBGraspEnv
from src.gcs_trajectory_optimizer import GCSTrajectoryOptimizer
import pybullet as p

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def visualize_gcs_only(test_object="knife", num_episodes=5, output_dir="results/videos"):
    """
    Visualize PURE GCS - No RL model needed!
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[GCS Only] Visualizing pure GCS planning for: {test_object}")
    
    env = YCBGraspEnv(
        ycb_objects=[test_object],
        num_regions=50,
        max_steps=100,
        render=True  # Enable visualization
    )
    
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        episode_reward = 0.0
        
        logger.info(f"\n Episode {episode+1}/{num_episodes}: Pure GCS Planning")
        
        try:
            # Get decomposer
            if not hasattr(env, 'decomposer') or env.decomposer is None:
                logger.warning("  No decomposer available")
                continue
            
            if not hasattr(env.decomposer, 'regions') or len(env.decomposer.regions) == 0:
                logger.warning("  Decomposer has no valid regions")
                continue
            
            # Get GCS path
            goal_region = env.decomposer.get_region_for_config(
                env.goal_config) if hasattr(env, 'goal_config') else 0
            start_region = env.current_region_id if hasattr(env, 'current_region_id') else 0
            
            logger.info(f"  Start region: {start_region}, Goal region: {goal_region}")
            
            # Plan using GCS
            optimizer = GCSTrajectoryOptimizer(env.decomposer)
            path = optimizer.dijkstra_path(start_region, goal_region)
            
            if path is None:
                logger.warning(f"  GCS path planning failed")
                continue
            
            logger.info(f"  GCS found path with {len(path)} waypoints")
            
            # Execute path
            for i, region_id in enumerate(path):
                if done:
                    break
                
                obs, reward, done, truncated, info = env.step(region_id)
                episode_reward += reward
                step_count += 1
                
                logger.info(f"    Step {step_count}: Region {region_id}, Reward: {reward:.2f}")
                
                if done or truncated:
                    break
            
            if episode_reward > 40:  # Success threshold
                success_count += 1
                logger.info(f"  ✓ Episode {episode+1} SUCCESS - Reward: {episode_reward:.2f}")
            else:
                logger.info(f"  ✗ Episode {episode+1} Failed - Reward: {episode_reward:.2f}")
        
        except Exception as e:
            logger.error(f"  Episode {episode+1} error: {e}", exc_info=True)
            continue
    
    env.close()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Pure GCS Results for {test_object}:")
    logger.info(f"  Success Rate: {100*success_count/num_episodes:.1f}%")
    logger.info(f"  Successful Episodes: {success_count}/{num_episodes}")
    logger.info(f"{'='*70}")
    
    return success_count, num_episodes

if __name__ == "__main__":
    setup_logging()
    test_object = sys.argv[1] if len(sys.argv) > 1 else "knife"
    visualize_gcs_only(test_object, num_episodes=5)
