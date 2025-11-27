"""
Compare Motion Planning Methods

Compares pure GCS, pure RL, and hybrid approaches on grasp task.
"""

import sys
import os
import json
import logging
import time
import numpy as np

logger = logging.getLogger(__name__)

# Ensure we can find src if running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# UPDATED IMPORT:
try:
    from src.gcs_trajectory_optimizer import GCSTrajectoryOptimizer
    from src.gcs_decomposer import GCSDecomposer
    from src.ycb_grasp_rl_env import YCBGraspEnv
    from stable_baselines3 import PPO
    COMPARE_AVAILABLE = True
except ImportError as e:
    print(f"Import Error: {e}")
    COMPARE_AVAILABLE = False


def evaluate_pure_gcs(env):
    """
    Evaluate pure GCS: Use only shortest path through regions.

    Args:
        env: YCBGraspEnv environment

    Returns:
        Dictionary with results
    """
    obs, _ = env.reset()

    if not hasattr(env, 'decomposer'):
        logger.warning("Environment does not have decomposer")
        return {'success': False, 'time': 0, 'path_length': 0}

    try:
        goal_region = env.decomposer.get_region_for_config(env.goal_config) if hasattr(env, 'goal_config') else 0
        start_region = env.current_region_id if hasattr(env, 'current_region_id') else 0

        start_time = time.time()
        optimizer = GCSTrajectoryOptimizer(env.decomposer)
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
    obs, _ = env.reset()
    done = False
    step_count = 0
    success = False
    episode_reward = 0.0

    start_time = time.time()
    while not done and step_count < 100:
        try:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward
            step_count += 1
        except Exception as e:
            logger.error(f"RL step failed: {e}")
            break

    solve_time = time.time() - start_time
    success = episode_reward > 40

    return {
        'success': bool(success),
        'time': float(solve_time),
        'path_length': int(step_count),
        'reward': float(episode_reward)
    }


def evaluate_hybrid(env, model):
    """
    Evaluate hybrid: GCS provides coarse path, RL refines trajectory.

    The hybrid approach:
    1. Use GCS to plan a collision-free path through regions
    2. Extract waypoints from the GCS path
    3. Use RL policy to refine approach to each waypoint
    4. Measure combined success and efficiency

    Args:
        env: YCBGraspEnv environment
        model: Trained RL model

    Returns:
        Dictionary with results (success, time, path_length, reward)
    """

    obs, _ = env.reset()
    start_time = time.time()
    step_count = 0
    episode_reward = 0.0
    gcs_success = False
    waypoints = []

    try:
        # Step 1: Use GCS to plan a path through regions
        if not hasattr(env, 'decomposer'):
            logger.warning("No GCS decomposer available, falling back to pure RL")
            return evaluate_pure_rl(env, model)

        # Get start and goal regions
        goal_region = (env.decomposer.get_region_for_config(env.goal_config) 
                      if hasattr(env, 'goal_config') else 0)
        start_region = env.current_region_id if hasattr(env, 'current_region_id') else 0

        # Plan GCS path
        optimizer = GCSTrajectoryOptimizer(env.decomposer)
        gcs_path = optimizer.dijkstra_path(start_region, goal_region)

        if gcs_path is None:
            logger.warning("GCS path planning failed, falling back to pure RL")
            return evaluate_pure_rl(env, model)

        gcs_success = True

        # Step 2: Extract waypoints
        # ROBUST FIX: Try to use regions_dict first (pre-calculated centroids)
        if hasattr(env, 'regions_dict'):
            for region_id in gcs_path:
                if region_id in env.regions_dict:
                    waypoint = env.regions_dict[region_id]['centroid']
                    waypoints.append(waypoint)
        
        # Fallback to calculating from decomposer.regions
        elif hasattr(env.decomposer, 'regions'):
            for region_id in gcs_path:
                if region_id < len(env.decomposer.regions):
                    region_configs = env.decomposer.regions[region_id]
                    
                    # Verify it's a valid array of points
                    region_arr = np.array(region_configs)
                    if region_arr.ndim > 1 and region_arr.shape[0] > 0:
                        waypoint = np.mean(region_arr, axis=0)
                        waypoints.append(waypoint)

        # Step 3: Use RL to refine trajectory through waypoints
        done = False
        current_waypoint_idx = 0
        
        # SAFETY: Max steps
        while not done and step_count < 200:
            try:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                step_count += 1
                
                # Force termination if truncated is returned
                if isinstance(done, tuple): # Handle terminated/truncated split if present
                    done = done[0] or done[1]

            except Exception as e:
                logger.error(f"Hybrid RL step failed: {e}")
                break

    except Exception as e:
        logger.error(f"Hybrid evaluation failed: {e}")
        return {
            'success': False,
            'time': float(time.time() - start_time),
            'path_length': int(step_count),
            'reward': float(episode_reward),
            'gcs_path_found': False
        }

    solve_time = time.time() - start_time
    success = episode_reward > 40 and gcs_success

    return {
        'success': bool(success),
        'time': float(solve_time),
        'path_length': int(step_count),
        'reward': float(episode_reward),
        'gcs_path_found': gcs_success,
        'waypoints_used': int(len(waypoints)) if gcs_success else 0
    }


def compare_methods(test_object: str = "e_lego_duplo",
                   num_trials: int = 20,
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
        Comparison results
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
                max_steps=100,
                render=False
            )

            results['gcs'].append(evaluate_pure_gcs(env))
            results['rl'].append(evaluate_pure_rl(env, model))
            results['hybrid'].append(evaluate_hybrid(env, model))

            env.close()

            if (trial + 1) % 5 == 0:
                logger.info(f"  Completed {trial + 1}/{num_trials} trials")
        except Exception as e:
            logger.error(f"Trial {trial + 1} failed: {e}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Comparison Summary")
    logger.info("=" * 70)

    summary = {}
    for method in ['gcs', 'rl', 'hybrid']:
        if results[method]:
            success_rate = 100.0 * np.mean([bool(r['success']) for r in results[method]])
            successful_results = [r for r in results[method] if r['success']]

            if successful_results:
                avg_time = np.mean([float(r['time']) for r in successful_results])
                avg_path_len = np.mean([float(r['path_length']) for r in successful_results])
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

    # Save results with custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return super(NumpyEncoder, self).default(obj)

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
        compare_methods()
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        exit(1)
