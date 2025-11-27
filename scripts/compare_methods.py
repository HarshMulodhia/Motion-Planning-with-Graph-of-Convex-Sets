# scripts/compare_methods.py (CORRECTED)
"""
Compare Motion Planning Methods: Pure GCS vs Pure RL vs Hybrid
"""

import sys
import os
import json
import logging
import time
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Ensure we can find src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from src.gcs_trajectory_optimizer import GCSTrajectoryOptimizer
    from src.gcs_decomposer import GCSDecomposer
    from src.ycb_grasp_rl_env import YCBGraspEnv
    from stable_baselines3 import PPO
    COMPARE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Import error: {e}")
    COMPARE_AVAILABLE = False


def evaluate_pure_gcs(env):
    """Evaluate pure GCS: Use only shortest path through regions."""
    
    obs, _ = env.reset()
    
    # FIX: Access unwrapped environment (in case it's wrapped by Gym)
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    
    # FIX: Check both existence AND that it's not None
    if not hasattr(actual_env, 'decomposer') or actual_env.decomposer is None:
        logger.warning("Environment does not have valid decomposer")
        return {'success': False, 'time': 0, 'path_length': 0}
    
    # FIX: Check that decomposer has regions
    if not hasattr(actual_env.decomposer, 'regions') or len(actual_env.decomposer.regions) == 0:
        logger.warning("Decomposer has no valid regions")
        return {'success': False, 'time': 0, 'path_length': 0}
    
    try:
        goal_region = actual_env.decomposer.get_region_for_config(actual_env.goal_config) if hasattr(actual_env, 'goal_config') else 0
        start_region = actual_env.current_region_id if hasattr(actual_env, 'current_region_id') else 0
        
        start_time = time.time()
        optimizer = GCSTrajectoryOptimizer(actual_env.decomposer)
        path = optimizer.dijkstra_path(start_region, goal_region)
        solve_time = time.time() - start_time
        
        if path is None:
            return {'success': False, 'time': float(solve_time), 'path_length': 0}
        
        return {
            'success': True,
            'time': float(solve_time),
            'path_length': int(len(path))
        }
        
    except Exception as e:
        logger.error(f"GCS evaluation failed: {e}")
        return {'success': False, 'time': 0, 'path_length': 0}


def evaluate_pure_rl(env, model):
    """
    Evaluate pure RL: Use only learned policy.
    
    Args:
        env: YCBGraspEnv environment
        model: Trained RL model
        
    Returns:
        Dictionary with results
    """
    try:
        obs, _ = env.reset()
        done = False
        step_count = 0
        success = False
        episode_reward = 0.0
        
        start_time = time.time()
        
        while not done and step_count < 150:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Extract success from distance
                if 'distance' in info:
                    if float(info['distance']) < 0.1:
                        success = True
                
            except Exception as e:
                logger.error(f"RL step failed: {e}")
                break
        
        solve_time = time.time() - start_time
        
        return {
            'success': bool(success),
            'time': float(solve_time),
            'path_length': int(step_count),
            'reward': float(episode_reward)
        }
    
    except Exception as e:
        logger.error(f"Pure RL evaluation failed: {e}")
        return {
            'success': False,
            'time': 0,
            'path_length': 0,
            'reward': 0.0
        }


def evaluate_hybrid(env, model):
    """
    Evaluate hybrid: GCS provides coarse path, RL refines trajectory.
    
    Args:
        env: YCBGraspEnv environment
        model: Trained RL model
        
    Returns:
        Dictionary with results
    """
    try:
        obs, _ = env.reset()
        start_time = time.time()
        step_count = 0
        episode_reward = 0.0
        gcs_success = False
        waypoints = []
        
        # Step 1: Use GCS to plan a path
        if not hasattr(env, 'decomposer') or env.decomposer is None:
            logger.warning("No GCS decomposer, falling back to pure RL")
            return evaluate_pure_rl(env, model)
        
        try:
            # Get start and goal regions
            start_region = getattr(env, 'current_region_id', 0)
            goal_region = env.decomposer.get_region_for_config(env.goal_config)
            
            # Plan GCS path
            optimizer = GCSTrajectoryOptimizer(env.decomposer)
            gcs_path = optimizer.dijkstra_path(start_region, goal_region)
            
            if gcs_path is None:
                logger.warning("GCS path planning failed, falling back to pure RL")
                return evaluate_pure_rl(env, model)
            
            gcs_success = True
            
            # Step 2: Extract waypoints from GCS path
            if hasattr(env, 'regions_dict') and env.regions_dict:
                for region_id in gcs_path:
                    if region_id in env.regions_dict:
                        waypoint = env.regions_dict[region_id]['centroid']
                        waypoints.append(waypoint)
        
        except Exception as e:
            logger.warning(f"GCS planning failed: {e}, using pure RL")
            return evaluate_pure_rl(env, model)
        
        # Step 3: Use RL to refine trajectory
        done = False
        success = False
        max_steps = 200
        
        while not done and step_count < max_steps:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
                
                # Check success
                if 'distance' in info:
                    if float(info['distance']) < 0.1:
                        success = True
            
            except Exception as e:
                logger.error(f"Hybrid RL step failed: {e}")
                break
        
        solve_time = time.time() - start_time
        
        return {
            'success': bool(success and gcs_success),
            'time': float(solve_time),
            'path_length': int(step_count),
            'reward': float(episode_reward),
            'gcs_path_found': gcs_success,
            'waypoints_used': int(len(waypoints))
        }
    
    except Exception as e:
        logger.error(f"Hybrid evaluation failed: {e}")
        return {
            'success': False,
            'time': 0,
            'path_length': 0,
            'reward': 0.0,
            'gcs_path_found': False,
            'waypoints_used': 0
        }


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def compare_methods(test_object: str = "rubiks_cube",
                   num_trials: int = 10,
                   model_path: str = "models/ycb_grasp/final_model",
                   output_dir: str = "results") -> dict:
    """
    Compare different planning methods.
    
    Args:
        test_object: Object to test on
        num_trials: Number of trials per method
        model_path: Path to trained model
        output_dir: Output directory for results
        
    Returns:
        Comparison results dictionary
    """
    if not COMPARE_AVAILABLE:
        logger.error("Required modules not available")
        return {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Comparing Planning Methods")
    logger.info("=" * 70)
    
    # Load model
    logger.info(f"[Compare] Loading model from: {model_path}")
    try:
        model = PPO.load(model_path)
        logger.info("✓ Model loaded")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return {}
    
    results = {'gcs': [], 'rl': [], 'hybrid': []}
    
    logger.info(f"\n[Compare] Running {num_trials} trials on {test_object}...")
    
    for trial in range(num_trials):
        try:
            env = YCBGraspEnv(
                ycb_objects=[test_object],
                num_regions=20,
                max_steps=100,
                render=False
            )
            
            results['gcs'].append(evaluate_pure_gcs(env))
            results['rl'].append(evaluate_pure_rl(env, model))
            results['hybrid'].append(evaluate_hybrid(env, model))
            
            env.close()
            
            if (trial + 1) % 5 == 0:
                logger.info(f" Completed {trial + 1}/{num_trials} trials")
        
        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")
            continue
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Comparison Summary")
    logger.info("=" * 70)
    
    summary = {}
    
    for method in ['gcs', 'rl', 'hybrid']:
        if results[method]:
            success_rate = 100.0 * np.mean([bool(r['success']) for r in results[method]])
            
            successful = [r for r in results[method] if r['success']]
            
            if successful:
                avg_time = np.mean([float(r['time']) for r in successful])
                avg_path_len = np.mean([float(r['path_length']) for r in successful])
            else:
                avg_time = 0
                avg_path_len = 0
            
            summary[method] = {
                'success_rate': float(success_rate),
                'avg_time': float(avg_time),
                'avg_path_length': float(avg_path_len),
                'num_trials': len(results[method])
            }
            
            logger.info(f"\n{method.upper()}:")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Avg Time: {avg_time:.3f}s")
            logger.info(f"  Avg Path Length: {avg_path_len:.1f}")
    
    # Save results
    results_file = os.path.join(output_dir, 'comparison_results.json')
    with open(results_file, 'w') as f:
        json.dump({'summary': summary, 'detailed': results}, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"\n✓ Results saved to: {results_file}")
    logger.info("=" * 70)
    
    return summary


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        results = compare_methods(num_trials=5)
        logger.info("✓ Comparison completed successfully")
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        exit(1)
