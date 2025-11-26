# src/ycb_grasp_rl_env.py
import gymnasium as gym
import numpy as np
import pybullet as p
from typing import Tuple, Dict
from src.gcs_decomposer import GCSDecomposer
from src.gripper_config import ConfigurationSpaceSampler

class YCBGraspEnv(gym.Env):
    """
    RL Environment: Learn grasp approach policies across YCB objects
    
    State: [current_config(6) | goal_config(6) | object_id(1) | region_features(10)]
    Action: Discrete - select next adjacent region or continuous approach motion
    Reward: -distance_to_goal - collision_penalty - grasp_instability
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, ycb_objects: List[str], num_regions: int = 50, 
                 max_steps: int = 100, render: bool = False):
        """
        Args:
            ycb_objects: List of YCB object names ['001_chips_can', '002_master_chef_can']
            num_regions: Number of GCS regions to create
            max_steps: Maximum steps per episode
            render: Enable PyBullet visualization
        """
        super().__init__()
        
        self.ycb_objects = ycb_objects
        self.num_regions = num_regions
        self.max_steps = max_steps
        self.render_mode = "human" if render else None
        
        # PyBullet setup
        self.physics_client = None
        self.object_id = None
        self.gripper_id = None
        
        # GCS decomposition
        self.decomposer = None
        self.sampler = None
        
        # State tracking
        self.current_config = None
        self.goal_config = None
        self.current_region_id = 0
        self.step_count = 0
        
        # Action space: discrete selection of adjacent regions
        self.action_space = gym.spaces.Discrete(num_regions)
        
        # Observation space: [current(6) | goal(6) | object_id(1) | region_features(10)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float32
        )
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render_mode == "human" else p.DIRECT)
            p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # Cleanup from previous episode
        if self.object_id is not None:
            p.removeBody(self.object_id, physicsClientId=self.physics_client)
        if self.gripper_id is not None:
            p.removeBody(self.gripper_id, physicsClientId=self.physics_client)
        
        # Load random YCB object
        object_name = np.random.choice(self.ycb_objects)
        object_path = f"~/datasets/ycb_data/ycb_models/{object_name}/google_16k/textured.ply"
        object_path = os.path.expanduser(object_path)
        
        self.object_id = p.loadURDF(object_path, [0, 0, 0.5], 
                                    physicsClientId=self.physics_client)
        
        # Load gripper
        self.gripper_id = p.loadURDF("data/gripper.urdf", [0, 0, 1], 
                                     physicsClientId=self.physics_client)
        
        # Sample free configurations
        workspace = {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (0.2, 1.2)
        }
        
        self.sampler = ConfigurationSpaceSampler(
            workspace, self.object_id, self.physics_client
        )
        
        free_configs = self.sampler.sample_free_configurations(
            num_samples=5000, gripper_id=self.gripper_id
        )
        
        # GCS decomposition
        self.decomposer = GCSDecomposer(free_configs, self.num_regions)
        self.decomposer.decompose()
        
        # Sample start and goal
        self.current_config = free_configs[np.random.randint(len(free_configs))]
        self.goal_config = free_configs[np.random.randint(len(free_configs))]
        
        self.current_region_id = self.decomposer.get_region_for_config(self.current_config)
        self.step_count = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action: move to next region"""
        next_region_id = action
        
        # Check if action is valid (adjacent region)
        if not self.decomposer.are_adjacent(self.current_region_id, next_region_id):
            reward = -10.0  # Penalty for invalid transition
            done = False
            obs = self._get_observation()
            return obs, reward, done, False, {'error': 'invalid_transition'}
        
        # Move to next region centroid
        next_config = self.decomposer.regions[next_region_id]['centroid']
        
        # Interpolate trajectory and check collisions
        num_checks = 10
        collision_detected = False
        
        for alpha in np.linspace(0, 1, num_checks):
            interpolated = (1 - alpha) * self.current_config + alpha * next_config
            p.resetBasePositionAndOrientation(
                self.gripper_id,
                interpolated[:3],
                p.getQuaternionFromEuler(interpolated[3:6]),
                physicsClientId=self.physics_client
            )
            
            contacts = p.getContactPoints(
                self.gripper_id, self.object_id, 
                physicsClientId=self.physics_client
            )
            
            if len(contacts) > 0 and alpha < 0.9:  # Allow end contact
                collision_detected = True
                break
        
        if collision_detected:
            reward = -20.0  # Large penalty for collision
            done = True
            obs = self._get_observation()
            return obs, reward, done, False, {'error': 'collision'}
        
        # Update state
        self.current_config = next_config
        self.current_region_id = next_region_id
        self.step_count += 1
        
        # Calculate reward
        dist_to_goal = np.linalg.norm(self.current_config - self.goal_config)
        
        # Rewards: distance penalty + step penalty
        reward = -dist_to_goal * 0.1 - 0.05
        
        # Success bonus if reached goal
        done = False
        if dist_to_goal < 0.1:
            reward += 50.0
            done = True
        
        # Max steps termination
        if self.step_count >= self.max_steps:
            done = True
        
        obs = self._get_observation()
        return obs, reward, done, False, {'distance': dist_to_goal}
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # Region features
        region = self.decomposer.regions[self.current_region_id]
        region_features = np.concatenate([
            region['centroid'][:3],  # 3D region center
            [region['num_samples']],  # Region density
            np.zeros(6)  # Padding
        ])[:10]
        
        obs = np.concatenate([
            self.current_config,           # 6D current gripper config
            self.goal_config,              # 6D goal config
            [self.current_region_id / self.num_regions],  # Normalized region ID
            region_features                # Region features
        ]).astype(np.float32)
        
        return obs
    
    def render(self):
        """Render using PyBullet GUI"""
        if self.render_mode == "human":
            pass  # PyBullet GUI auto-renders
    
    def close(self):
        """Clean up resources"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
