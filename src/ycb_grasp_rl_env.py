# src/ycb_grasp_rl_env.py
import gymnasium as gym
import numpy as np
import pybullet as p
from typing import Tuple, Dict
from src.gcs_decomposer import GCSDecomposer
from src.gripper_config import ConfigSpace, GripperConfig
from typing import List
import os


class YCBGraspEnv(gym.Env):
    """
    RL Environment: Learn grasp approach policies across YCB objects
    
    State: [current_config(6) | goal_config(6) | object_id(1) | region_features(10)]
    Action: Discrete - select next adjacent region or continuous approach motion
    Reward: -distance_to_goal - collision_penalty - grasp_instability
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, ycb_objects: List[str], num_regions: int = 50,
             max_steps: int = 1000, render: bool = False):
        """
        Args:
            ycb_objects: List of YCB object names
            num_regions: Number of GCS regions
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
        self.regions_dict = {}  # ADD THIS - structured regions
        
        # State tracking
        self.current_config = None
        self.prev_config = None  # ADD THIS - for progress tracking
        self.goal_config = None
        self.current_region_id = 0
        self.step_count = 0
        
        # Action space
        self.action_space = gym.spaces.Discrete(num_regions)
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
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
        object_path = f"./datasets/ycb_data/{object_name}.urdf"
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
        
        gripper_urdf_path = "data/gripper.urdf"
        self.sampler = GripperConfig(gripper_urdf_path)
        free_configs = self.sampler.sample_multiple_configs(num_samples=5000)
        
        # GCS decomposition
        self.decomposer = GCSDecomposer(free_configs, self.num_regions)
        self.decomposer.decompose()
        
        # Create structured regions dict (Handle dict or list)
        self.regions_dict = {}
        
        # Determine how to iterate based on type
        if isinstance(self.decomposer.regions, dict):
            iterator = self.decomposer.regions.items()  # Iterate (id, points)
        else:
            iterator = enumerate(self.decomposer.regions)  # Iterate (index, points)
            
        for i, region_points in iterator:
                        # Convert to numpy array safely
            points_array = np.asarray(region_points)
            
            # FIXED: Robustly check for empty or scalar regions
            if points_array.ndim == 0 or points_array.size == 0:
                # This region is empty or invalid, create a default
                centroid = np.zeros(6)  # Default 6D zero-config
                num_samples = 0
                points_array = np.array([]) # Ensure consistent type
            else:
                # This is a valid region with points
                centroid = np.mean(points_array, axis=0)
                num_samples = points_array.shape[0]
            
            self.regions_dict[i] = {
                'centroid': centroid,
                'num_samples': num_samples,
                'points': points_array
            }
        
        # Sample start and goal
        self.current_config = free_configs[np.random.randint(len(free_configs))]
        self.goal_config = free_configs[np.random.randint(len(free_configs))]
        self.prev_config = self.current_config.copy()  # ADD THIS
        self.current_region_id = self.decomposer.get_region_for_config(self.current_config)
        self.step_count = 0
        
        obs = self._get_observation()
        return obs, {}

    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action: move to next region"""
        next_region_id = action
        
        # ===== SAFETY: Bounds check =====
        if next_region_id < 0 or next_region_id >= self.num_regions:
            # Compute current distance first
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            
            reward = -5.0
            terminated = False  # Don't end episode on bad action
            truncated = False
            obs = self._get_observation()
            
            return obs, reward, terminated, truncated, {
                'error': 'invalid_action_index',
                'distance': curr_dist
            }
        
        # ===== CHECK ADJACENCY: Invalid transition =====
        try:
            is_adjacent = self.decomposer.are_adjacent(self.current_region_id, next_region_id)
        except:
            is_adjacent = True  # Fallback if method missing
        
        if not is_adjacent:
            # Compute current distance
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            
            reward = -5.0
            terminated = False  # Don't end episode, just penalize
            truncated = False
            obs = self._get_observation()
            
            return obs, reward, terminated, truncated, {
                'error': 'invalid_transition',
                'distance': curr_dist
            }
        
        # ===== GET NEXT CONFIG =====
        try:
            next_config = self.regions_dict[next_region_id]['centroid']
        except Exception as e:
            # Fallback if regions_dict not available
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            reward = -5.0
            terminated = False
            truncated = False
            
            return self._get_observation(), reward, terminated, truncated, {
                'error': f'region_access_failed: {str(e)}',
                'distance': curr_dist
            }
        
        # ===== COLLISION CHECK: Interpolate trajectory =====
        num_checks = 10
        collision_detected = False
        
        for alpha in np.linspace(0, 1, num_checks):
            interpolated = (1 - alpha) * self.current_config + alpha * next_config
            
            try:
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
                
                # Collision detected (but allow end contact)
                if len(contacts) > 0 and alpha < 0.9:
                    collision_detected = True
                    break
            except:
                # Handle PyBullet errors gracefully
                pass
        
        if collision_detected:
            # Compute current distance (gripper didn't move due to collision)
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            
            reward = -5.0
            terminated = False  # Don't end, just penalize
            truncated = False
            obs = self._get_observation()
            
            return obs, reward, terminated, truncated, {
                'error': 'collision',
                'distance': curr_dist
            }
        
        # ===== UPDATE STATE: Move was successful =====
        prev_config = self.current_config.copy()
        self.current_config = next_config
        self.current_region_id = next_region_id
        self.step_count += 1
        
        # ===== CALCULATE REWARD =====
        curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
        prev_dist = float(np.linalg.norm(prev_config - self.goal_config))
        
        # Progress-based reward (positive for moving closer)
        reward_progress = (prev_dist - curr_dist) * 5.0
        
        # Small per-step cost
        reward = reward_progress - 0.01
        
        # ===== CHECK TERMINATION CONDITIONS =====
        terminated = False
        truncated = False
        
        # SUCCESS: Goal reached!
        if curr_dist < 0.1:
            reward += 100.0
            terminated = True
        
        # TRUNCATION: Max steps exceeded
        if self.step_count >= self.max_steps:
            truncated = True
        
        obs = self._get_observation()
        
        # ===== RETURN: Both values matter! =====
        return obs, reward, terminated, truncated, {'distance': curr_dist}

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # Region features from structured dict
        region = self.regions_dict[self.current_region_id]
        
        region_features = np.concatenate([
            region['centroid'][:3],  # 3D region center
            [region['num_samples']],  # Region density
            np.zeros(6)  # Padding
        ])[:10]
        
        obs = np.concatenate([
            self.current_config,  # 6D current gripper config
            self.goal_config,  # 6D goal config
            [self.current_region_id / self.num_regions],  # Normalized region ID
            region_features  # Region features
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
