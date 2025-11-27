# evaluate_gcs_only.py
# Pure GCS evaluation - no RL dependencies

import sys
import os
import json
import time
import numpy as np
import logging
from src.gcs_motion_planner import GCSMotionPlanner

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_gcs_planner(object_name: str = "banana",
                        num_trials: int = 10,
                        num_regions: int = 50,
                        output_dir: str = "results") -> dict:
    """
    Evaluate GCS motion planner on YCB object.
    
    Args:
        object_name: YCB object to test
        num_trials: Number of evaluation trials
        num_regions: Number of regions for decomposition
        output_dir: Directory to save results
        
    Returns:
        Dictionary with evaluation results
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("GCS Motion Planner Evaluation")
    logger.info("=" * 80)
    logger.info(f"Object: {object_name}")
    logger.info(f"Trials: {num_trials}")
    logger.info(f"Regions: {num_regions}")
    
    results = {
        'planning_success': [],
        'planning_time': [],
        'path_lengths': [],
        'execution_time': [],
        'execution_success': []
    }
    
    planner = GCSMotionPlanner(max_regions=num_regions)
    
    # Setup environment once
    try:
        object_path = f"./datasets/ycb_data/{object_name}.urdf"
        planner.setup_environment(object_path, num_samples=2000)
        logger.info(f"✓ Environment setup complete")
    except Exception as e:
        logger.error(f"✗ Failed to setup environment: {e}")
        return results
    
    # Run trials
    for trial in range(num_trials):
        try:
            logger.info(f"\n[Trial {trial+1}/{num_trials}]")
            
            # Sample random start and goal
            start_config = planner.free_configs[np.random.randint(len(planner.free_configs))]
            goal_config = planner.free_configs[np.random.randint(len(planner.free_configs))]
            
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
                continue
            
            logger.info(f"  ✓ Path planned in {plan_time:.3f}s ({len(path)} regions)")
            results['planning_success'].append(True)
            results['planning_time'].append(plan_time)
            results['path_lengths'].append(len(path))
            
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
    
    # Cleanup
    planner.cleanup()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    
    planning_success_rate = 100.0 * np.mean(results['planning_success']) if results['planning_success'] else 0
    execution_success_rate = 100.0 * np.mean(results['execution_success']) if results['execution_success'] else 0
    
    avg_plan_time = np.mean(results['planning_time']) if results['planning_time'] else 0
    avg_exec_time = np.mean(results['execution_time']) if results['execution_time'] else 0
    avg_path_length = np.mean(results['path_lengths']) if results['path_lengths'] else 0
    
    logger.info(f"Planning Success Rate: {planning_success_rate:.1f}%")
    logger.info(f"Execution Success Rate: {execution_success_rate:.1f}%")
    logger.info(f"Avg Planning Time: {avg_plan_time:.3f}s")
    logger.info(f"Avg Execution Time: {avg_exec_time:.3f}s")
    logger.info(f"Avg Path Length: {avg_path_length:.1f} regions")
    
    # Save results
    summary = {
        'object': object_name,
        'trials': num_trials,
        'regions': num_regions,
        'planning_success_rate': float(planning_success_rate),
        'execution_success_rate': float(execution_success_rate),
        'avg_planning_time': float(avg_plan_time),
        'avg_execution_time': float(avg_exec_time),
        'avg_path_length': float(avg_path_length),
        'total_time': float(sum(results['planning_time']) + sum(results['execution_time']))
    }
    
    output_file = os.path.join(output_dir, 'gcs_evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    # Test on multiple objects
    test_objects = ['banana', 'apple', 'lemon']
    all_results = {}
    
    for obj in test_objects:
        try:
            results = evaluate_gcs_planner(obj, num_trials=5, num_regions=50)
            all_results[obj] = results
        except Exception as e:
            logger.error(f"Evaluation for {obj} failed: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)
