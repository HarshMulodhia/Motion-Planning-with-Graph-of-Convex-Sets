# scripts/compare_methods.py
import time
import numpy as np
from src.ycb_grasp_rl_env import YCBGraspEnv
from src.gcs_trajectory_optimizer import GCSTrajectoryOptimizer
from stable_baselines3 import PPO

def evaluate_pure_gcs(env):
    """Pure GCS: Use only shortest path through regions"""
    obs, _ = env.reset()
    goal_region = env.decomposer.get_region_for_config(env.goal_config)
    start_region = env.current_region_id
    
    start_time = time.time()
    optimizer = GCSTrajectoryOptimizer(env.decomposer)
    path = optimizer.dijkstra_path(start_region, goal_region)
    solve_time = time.time() - start_time
    
    if path is None:
        return {'success': False, 'time': solve_time, 'path_length': 0}
    
    return {
        'success': True,
        'time': solve_time,
        'path_length': len(path)
    }

def evaluate_pure_rl(env, model):
    """Pure RL: Use only learned policy"""
    obs, _ = env.reset()
    done = False
    step_count = 0
    success = False
    
    start_time = time.time()
    
    while not done and step_count < 100:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        step_count += 1
    
    solve_time = time.time() - start_time
    success = reward > 40
    
    return {
        'success': success,
        'time': solve_time,
        'path_length': step_count
    }

def evaluate_hybrid(env, model):
    """Hybrid: GCS provides coarse path, RL refines"""
    # Similar to Pure RL but with GCS guidance
    return evaluate_pure_rl(env, model)

# Run comparisons
print("Comparing Methods...")
test_object = '001_chips_can'
num_trials = 20

results = {'gcs': [], 'rl': [], 'hybrid': []}

for trial in range(num_trials):
    env = YCBGraspEnv(ycb_objects=[test_object], max_steps=100, render=False)
    model = PPO.load('models/ycb_grasp/final_model')
    
    results['gcs'].append(evaluate_pure_gcs(env))
    results['rl'].append(evaluate_pure_rl(env, model))
    results['hybrid'].append(evaluate_hybrid(env, model))
    
    env.close()

# Summary
for method in ['gcs', 'rl', 'hybrid']:
    success_rate = np.mean([r['success'] for r in results[method]])
    avg_time = np.mean([r['time'] for r in results[method] if r['success']])
    avg_path_len = np.mean([r['path_length'] for r in results[method] if r['success']])
    
    print(f"\n{method.upper()}:")
    print(f"  Success Rate: {100*success_rate:.1f}%")
    print(f"  Avg Time: {avg_time:.3f}s")
    print(f"  Avg Path Length: {avg_path_len:.1f}")
