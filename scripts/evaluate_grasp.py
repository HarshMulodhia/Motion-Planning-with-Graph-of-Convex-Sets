# scripts/evaluate_grasp.py
import numpy as np
from stable_baselines3 import PPO
from ycb_grasp_rl_env import YCBGraspEnv
import json

# Load trained model and config
config_path = 'models/ycb_grasp/config.json'
with open(config_path) as f:
    config = json.load(f)

model = PPO.load('models/ycb_grasp/final_model')

# Test on unseen objects (objects 11-15)
test_objects = [
    '011_banana',
    '012_strawberry',
    '013_apple',
    '014_lemon',
    '015_peach'
]

print("=" * 60)
print("YCB Grasp Policy Evaluation")
print("=" * 60)

results = {
    'success_rate': [],
    'avg_path_length': [],
    'avg_distance': [],
    'num_episodes': 0
}

# Run evaluation episodes
for obj in test_objects:
    print(f"\nTesting on: {obj}")
    
    env = YCBGraspEnv(
        ycb_objects=[obj],
        num_regions=config['num_regions'],
        max_steps=config['max_steps'],
        render=False
    )
    
    for episode in range(5):  # 5 episodes per object
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        success = episode_reward > 40  # Threshold for success
        results['success_rate'].append(success)
        results['avg_path_length'].append(step_count)
        results['avg_distance'].append(info.get('distance', 0))
        results['num_episodes'] += 1
        
        print(f"  Episode {episode+1}: {'✓ Success' if success else '✗ Failed'} | "
              f"Length: {step_count} | Reward: {episode_reward:.2f}")
    
    env.close()

# Summary statistics
print("\n" + "=" * 60)
print("Summary Statistics")
print("=" * 60)
print(f"Total Episodes: {results['num_episodes']}")
print(f"Success Rate: {100*np.mean(results['success_rate']):.1f}%")
print(f"Average Path Length: {np.mean(results['avg_path_length']):.1f} steps")
print(f"Average Final Distance: {np.mean(results['avg_distance']):.4f}")

# Save results
import json
with open('results/grasp_eval_results.json', 'w') as f:
    json.dump({
        'success_rate': float(np.mean(results['success_rate'])),
        'avg_path_length': float(np.mean(results['avg_path_length'])),
        'avg_distance': float(np.mean(results['avg_distance']))
    }, f, indent=2)
