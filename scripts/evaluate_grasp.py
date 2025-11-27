# scripts/evaluate_grasp.py
"""
Evaluate trained YCB Grasp RL policy on unseen objects.
"""

import numpy as np
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from src.ycb_grasp_rl_env import YCBGraspEnv
    EVAL_AVAILABLE = True
except ImportError as e:
    logger.error(f"Import error: {e}")
    EVAL_AVAILABLE = False


def load_model_and_config(model_path: str = 'models/ycb_grasp/final_model',
                          config_path: str = 'models/ycb_grasp/config.json'):
    """
    Load trained model and configuration.
    
    Args:
        model_path: Path to trained model
        config_path: Path to config file
        
    Returns:
        Tuple of (model, config) or (None, None) if loading fails
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
        logger.info(f"✓ Config loaded from: {config_path}")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        config = {
            'num_regions': 20,
            'max_steps': 100,
            'ycb_objects': ['rubiks_cube', 'racquetball', 'hammer']
        }
    
    try:
        model = PPO.load(model_path)
        logger.info(f"✓ Model loaded from: {model_path}")
        return model, config
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, config


def evaluate_policy(env, model, num_episodes: int = 20):
    """
    Evaluate policy on environment.
    
    Args:
        env: YCBGraspEnv environment
        model: Trained PPO model
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary with evaluation results
    """
    results = {
        'success_rate': [],
        'avg_path_length': [],
        'avg_distance': [],
        'rewards': [],
        'num_episodes': 0
    }
    
    for episode in range(num_episodes):
        try:
            obs, _ = env.reset()
            done = False
            step_count = 0
            episode_reward = 0.0
            final_distance = 1.0
            MAX_STEPS = 150
            
            while not done and step_count < MAX_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Extract distance from info dict
                if 'distance' in info:
                    final_distance = info['distance']
                
                episode_reward += reward
                step_count += 1
                done = terminated or truncated
            
            # Success metric: reached goal with reasonable path
            success = (final_distance < 0.1)
            
            results['success_rate'].append(success)
            results['avg_path_length'].append(step_count)
            results['avg_distance'].append(final_distance)
            results['rewards'].append(episode_reward)
            results['num_episodes'] += 1
            
            status = '✓ Success' if success else '✗ Failed'
            logger.info(f" Episode {episode+1}/{num_episodes}: {status} | "
                       f"Distance: {final_distance:.4f} | Steps: {step_count} | "
                       f"Reward: {episode_reward:.2f}")
        
        except Exception as e:
            logger.error(f"Episode {episode+1} failed: {e}")
            continue
    
    return results


def main():
    """Main evaluation function."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not EVAL_AVAILABLE:
        logger.error("Required modules not available")
        return
    
    logger.info("=" * 70)
    logger.info("YCB Grasp Policy Evaluation")
    logger.info("=" * 70)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load model and config
    model, config = load_model_and_config()
    
    if model is None:
        logger.error("Failed to load model, exiting")
        return
    
    # Test objects
    test_objects = [
        'rubiks_cube',
        'racquetball',
        'hammer'
    ]
    
    all_results = {}
    
    # Evaluate on each object
    for obj in test_objects:
        logger.info(f"\n[Evaluation] Testing on: {obj}")
        
        try:
            env = YCBGraspEnv(
                ycb_objects=[obj],
                num_regions=config.get('num_regions', 20),
                max_steps=config.get('max_steps', 100),
                render=False
            )
            
            obj_results = evaluate_policy(env, model, num_episodes=20)
            all_results[obj] = obj_results
            env.close()
            
        except Exception as e:
            logger.error(f"Evaluation on {obj} failed: {e}")
            all_results[obj] = {'error': str(e)}
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Evaluation Summary")
    logger.info("=" * 70)
    
    summary = {}
    for obj, results in all_results.items():
        if 'error' in results:
            logger.info(f"\n{obj}: FAILED - {results['error']}")
            summary[obj] = {'error': results['error']}
        else:
            success_rate = 100 * np.mean(results['success_rate'])
            avg_distance = np.mean(results['avg_distance'])
            avg_steps = np.mean(results['avg_path_length'])
            avg_reward = np.mean(results['rewards'])
            
            summary[obj] = {
                'success_rate': float(success_rate),
                'avg_distance': float(avg_distance),
                'avg_steps': float(avg_steps),
                'avg_reward': float(avg_reward),
                'num_episodes': results['num_episodes']
            }
            
            logger.info(f"\n{obj}:")
            logger.info(f"  Success Rate: {success_rate:.1f}%")
            logger.info(f"  Avg Distance: {avg_distance:.4f}")
            logger.info(f"  Avg Steps: {avg_steps:.1f}")
            logger.info(f"  Avg Reward: {avg_reward:.2f}")
    
    # Save results
    results_file = 'results/grasp_eval_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ Results saved to: {results_file}")
    
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
