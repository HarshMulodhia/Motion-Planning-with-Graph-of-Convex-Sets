# evaluate_gcs_with_obstacles.py
# Compare GCS planning with and without obstacles

import sys
import os
import json
import time
import numpy as np
import logging
from trash.gcs_motion_planner_with_obstacles import GCSMotionPlannerWithObstacles

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_with_obstacles(object_name: str = "banana",
                           num_trials: int = 10,
                           num_regions: int = 50,
                           with_obstacles: bool = True,
                           output_dir: str = "results") -> dict:
    """
    Evaluate GCS motion planner with or without obstacles.
    
    Args:
        object_name: YCB object to test
        num_trials: Number of evaluation trials
        num_regions: Number of regions for decomposition
        with_obstacles: Whether to include obstacles
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    condition = "WITH OBSTACLES" if with_obstacles else "WITHOUT OBSTACLES"
    logger.info("=" * 80)
    logger.info(f"GCS Motion Planner Evaluation {condition}")
    logger.info("=" * 80)
    logger.info(f"Object: {object_name}")
    logger.info(f"Trials: {num_trials}")
    logger.info(f"Regions: {num_regions}")
    logger.info(f"Condition: {condition}")
    
    results = {
        'condition': condition,
        'planning_success': [],
        'planning_time': [],
        'path_lengths': [],
        'execution_time': [],
        'execution_success': [],
        'planning_difficulty': []  # NEW: complexity metric
    }
    
    planner = GCSMotionPlannerWithObstacles(max_regions=num_regions, use_obstacles=with_obstacles)
    
    # Setup environment once
    try:
        object_path = f"./datasets/ycb_data/{object_name}.urdf"
        planner.setup_environment(object_path, num_samples=2000)
        
        # Show obstacle info
        if with_obstacles:
            obs_info = planner.get_obstacle_info()
            logger.info(f"✓ Environment setup with {obs_info['num_obstacles']} obstacles")
            for obs in obs_info['obstacles']:
                logger.info(f"  - {obs['name']} ({obs['type']})")
        else:
            logger.info(f"✓ Environment setup (no obstacles)")
            
    except Exception as e:
        logger.error(f"✗ Failed to setup environment: {e}")
        return results
    
    # Run trials
    for trial in range(num_trials):
        try:
            logger.info(f"\n[Trial {trial+1}/{num_trials}]")
            
            # Sample random start and goal
            start_idx = np.random.randint(len(planner.free_configs))
            goal_idx = np.random.randint(len(planner.free_configs))
            
            start_config = planner.free_configs[start_idx]
            goal_config = planner.free_configs[goal_idx]
            
            # Plan path
            plan_start = time.time()
            path = planner.plan_path(start_config, goal_config)
            plan_time = time.time() - plan_start
            
            if path is None:
                logger.info(f"  ✗ Planning failed")
                results['planning_success'].append(False)
                results['planning_time'].append(plan_time)
                results['path_lengths'].append(0)
                results['execution_time'].append(0)
                results['execution_success'].append(False)
                results['planning_difficulty'].append(0)
                continue
            
            logger.info(f"  ✓ Path planned in {plan_time:.3f}s ({len(path)} regions)")
            results['planning_success'].append(True)
            results['planning_time'].append(plan_time)
            results['path_lengths'].append(len(path))
            
            # Calculate planning difficulty (longer paths = more difficult)
            difficulty = len(path) / num_regions
            results['planning_difficulty'].append(difficulty)
            
            # Execute path
            exec_success, exec_time, num_steps = planner.execute_path(path, planner.free_configs)
            logger.info(f"  ✓ Execution: {num_steps} steps in {exec_time:.3f}s")
            results['execution_time'].append(exec_time)
            results['execution_success'].append(exec_success)
            
        except Exception as e:
            logger.error(f"  ✗ Trial failed: {e}")
            results['planning_success'].append(False)
            results['planning_time'].append(0)
            results['path_lengths'].append(0)
            results['execution_time'].append(0)
            results['execution_success'].append(False)
            results['planning_difficulty'].append(0)
    
    # Cleanup
    planner.cleanup()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    planning_success_rate = sum(results['planning_success'])/len(results['planning_success']) if results['planning_success'] else 0
    execution_success_rate = sum(results['execution_success'])/len(results['execution_success']) if results['execution_success'] else 0
    
    avg_plan_time = np.mean(results['planning_time']) if results['planning_time'] else 0
    avg_exec_time = np.mean(results['execution_time']) if results['execution_time'] else 0
    avg_path_length = np.mean(results['path_lengths']) if results['path_lengths'] else 0
    avg_difficulty = np.mean(results['planning_difficulty']) if results['planning_difficulty'] else 0
    
    logger.info(f"Planning Success Rate: {planning_success_rate:.1f}%")
    logger.info(f"Execution Success Rate: {execution_success_rate:.1f}%")
    logger.info(f"Avg Planning Time: {avg_plan_time:.3f}s")
    logger.info(f"Avg Execution Time: {avg_exec_time:.3f}s")
    logger.info(f"Avg Path Length: {avg_path_length:.1f} regions")
    logger.info(f"Avg Planning Difficulty: {avg_difficulty:.2f}")
    
    # Save results
    summary = {
        'object': object_name,
        'condition': condition,
        'trials': num_trials,
        'regions': num_regions,
        'planning_success_rate': float(planning_success_rate),
        'execution_success_rate': float(execution_success_rate),
        'avg_planning_time': float(avg_plan_time),
        'avg_execution_time': float(avg_exec_time),
        'avg_path_length': float(avg_path_length),
        'avg_planning_difficulty': float(avg_difficulty),
        'total_time': float(sum(results['planning_time']) + sum(results['execution_time']))
    }
    
    return summary

def main():
    """Run comparison: with obstacles vs without obstacles"""
    
    print("\n" + "=" * 80)
    print("GCS Motion Planning: Obstacle Avoidance Comparison")
    print("=" * 80 + "\n")
    
    object_name = "banana"
    num_trials = 5
    num_regions = 50
    
    # Run without obstacles
    logger.info("PHASE 1: Planning WITHOUT Obstacles")
    logger.info("-" * 80)
    results_no_obstacles = evaluate_with_obstacles(
        object_name, num_trials, num_regions,
        with_obstacles=False
    )
    
    # Run with obstacles
    logger.info("\n\nPHASE 2: Planning WITH Obstacles")
    logger.info("-" * 80)
    results_with_obstacles = evaluate_with_obstacles(
        object_name, num_trials, num_regions,
        with_obstacles=True
    )
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: With Obstacles vs Without Obstacles")
    print("=" * 80 + "\n")
    
    print(f"{'Metric':<30} {'Without Obstacles':<25} {'With Obstacles':<25}")
    print("-" * 80)
    print("Available keys:", results_no_obstacles.keys())
    
    print(f"{'Planning Success':<30} {results_no_obstacles['planning_success_rate']:>6.1f}% {' '*17} {results_with_obstacles['planning_success_rate']:>6.1f}%")
    print(f"{'Execution Success':<30} {results_no_obstacles['execution_success_rate']:>6.1f}% {' '*17} {results_with_obstacles['execution_success_rate']:>6.1f}%")
    print(f"{'Avg Planning Time (ms)':<30} {results_no_obstacles['avg_planning_time']*1000:>8.2f} {' '*13} {results_with_obstacles['avg_planning_time']*1000:>8.2f}")
    print(f"{'Avg Execution Time (ms)':<30} {results_no_obstacles['avg_execution_time']*1000:>8.2f} {' '*13} {results_with_obstacles['avg_execution_time']*1000:>8.2f}")
    print(f"{'Avg Path Length':<30} {results_no_obstacles['avg_path_length']:>8.2f} {' '*13} {results_with_obstacles['avg_path_length']:>8.2f}")
    print(f"{'Avg Difficulty':<30} {results_no_obstacles['avg_planning_difficulty']:>8.2f} {' '*13} {results_with_obstacles['avg_planning_difficulty']:>8.2f}")
    
    # Calculate differences
    print("\n" + "-" * 80)
    print("Impact of Obstacles:")
    print("-" * 80)
    
    success_diff = results_with_obstacles['planning_success_rate'] - results_no_obstacles['planning_success_rate']
    time_diff_percent = ((results_with_obstacles['avg_planning_time'] - results_no_obstacles['avg_planning_time']) / results_no_obstacles['avg_planning_time'] * 100) if results_no_obstacles['avg_planning_time'] > 0 else 0
    path_diff = results_with_obstacles['avg_path_length'] - results_no_obstacles['avg_path_length']
    
    print(f"Planning Success Change: {success_diff:+.1f}%")
    print(f"Planning Time Change: {time_diff_percent:+.1f}%")
    print(f"Path Length Change: {path_diff:+.2f} regions")
    
    # Save comparison
    os.makedirs('results', exist_ok=True)
    comparison = {
        'object': object_name,
        'trials': num_trials,
        'regions': num_regions,
        'without_obstacles': results_no_obstacles,
        'with_obstacles': results_with_obstacles,
        'impact': {
            'planning_success_change_percent': float(success_diff),
            'planning_time_change_percent': float(time_diff_percent),
            'path_length_change': float(path_diff)
        }
    }
    
    with open('results/gcs_obstacle_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Comparison saved to: results/gcs_obstacle_comparison.json")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
