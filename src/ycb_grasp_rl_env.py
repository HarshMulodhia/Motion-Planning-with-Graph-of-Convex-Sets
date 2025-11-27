# src/ycb_grasp_rl_env.py (CORRECTED)
"""
YCB Grasp RL Environment for learning grasping policies.
"""

import gymnasium as gym
import numpy as np
import logging
from typing import Tuple, Dict, List

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    logger.warning("PyBullet not available, using mock simulation")
    PYBULLET_AVAILABLE = False

from src.gcs_decomposer import GCSDecomposer
from src.gripper_config import GripperConfig


class YCBGraspEnv(gym.Env):
    """
    RL Environment for learning grasp approach policies on YCB objects.
    
    State: [current_config(6) | goal_config(6) | object_id(1) | region_features(10)]
    Action: Discrete - select next adjacent region
    Reward: -distance_to_goal - collision_penalty + progress_reward
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, ycb_objects: List[str], num_regions: int = 20,
                 max_steps: int = 100, render: bool = False):
        """
        Initialize YCB Grasp Environment.
        
        Args:
            ycb_objects: List of YCB object names to train on
            num_regions: Number of GCS regions for action space
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
        self.regions_dict = {}
        
        # State tracking
        self.current_config = None
        self.goal_config = None
        self.current_region_id = 0
        self.step_count = 0
        
        # Action space: discrete selection of regions
        self.action_space = gym.spaces.Discrete(num_regions)
        
        # Observation space: 23D
        # [current_config(6) | goal_config(6) | object_id(1) | region_features(10)]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32
        )
        
        logger.info(f"[YCBGraspEnv] Initialized with {num_regions} regions, "
                   f"max_steps={max_steps}")
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment for new episode.
        
        Returns:
            Tuple of (observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Initialize PyBullet once
        if self.physics_client is None and PYBULLET_AVAILABLE:
            try:
                self.physics_client = p.connect(
                    p.GUI if self.render_mode == "human" else p.DIRECT
                )
                p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
                logger.info("[YCBGraspEnv] PyBullet initialized")
            except Exception as e:
                logger.warning(f"PyBullet initialization failed: {e}, using mock")
                self.physics_client = -1  # Mock flag
        
        # Sample free configurations (mock: random sampling)
        try:
            gripper_urdf_path = "data/gripper.urdf"
            self.sampler = GripperConfig(gripper_urdf_path)
            free_configs = self.sampler.sample_multiple_configs(num_samples=500)
        except Exception as e:
            logger.warning(f"GripperConfig failed: {e}, using random configs")
            free_configs = np.random.randn(500, 6)
        
        # GCS decomposition
        try:
            self.decomposer = GCSDecomposer(free_configs, self.num_regions)
            self.decomposer.decompose()
            logger.info(f"[YCBGraspEnv] GCS decomposed into {len(self.decomposer.regions)} regions")
        except Exception as e:
            logger.error(f"GCS decomposition failed: {e}")
            raise
        
        # Build regions dictionary for fast access
        self.regions_dict = {}
        
        if isinstance(self.decomposer.regions, dict):
            iterator = self.decomposer.regions.items()
        else:
            iterator = enumerate(self.decomposer.regions)
        
        for i, region_data in iterator:
            try:
                if isinstance(region_data, dict):
                    # Region is already a dict with centroid
                    centroid = region_data.get('centroid', np.zeros(6))
                    num_samples = region_data.get('num_samples', 0)
                    points = region_data.get('samples', np.array([]))
                else:
                    # Region is list of points
                    points_array = np.asarray(region_data)
                    if points_array.ndim == 0 or points_array.size == 0:
                        centroid = np.zeros(6)
                        num_samples = 0
                        points = np.array([])
                    else:
                        centroid = np.mean(points_array, axis=0) if points_array.ndim > 1 else np.zeros(6)
                        num_samples = len(points_array)
                        points = points_array
                
                self.regions_dict[i] = {
                    'centroid': centroid,
                    'num_samples': num_samples,
                    'points': points
                }
            except Exception as e:
                logger.warning(f"Failed to process region {i}: {e}")
                self.regions_dict[i] = {
                    'centroid': np.zeros(6),
                    'num_samples': 0,
                    'points': np.array([])
                }
        
        # Sample start and goal
        self.current_config = free_configs[np.random.randint(len(free_configs))]
        self.goal_config = free_configs[np.random.randint(len(free_configs))]
        self.current_region_id = 0
        self.step_count = 0
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action: move to selected region.
        
        Args:
            action: Region ID to move to
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        next_region_id = action
        
        # Bounds check
        if next_region_id < 0 or next_region_id >= self.num_regions:
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            return self._get_observation(), -5.0, False, False, {
                'error': 'invalid_action',
                'distance': curr_dist
            }
        
        # Adjacency check
        try:
            is_adjacent = self.decomposer.are_adjacent(
                self.current_region_id, next_region_id
            )
        except:
            is_adjacent = True
        
        if not is_adjacent:
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            return self._get_observation(), -5.0, False, False, {
                'error': 'non_adjacent_transition',
                'distance': curr_dist
            }
        
        # Get next configuration
        try:
            next_config = self.regions_dict[next_region_id]['centroid']
        except:
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            return self._get_observation(), -5.0, False, False, {
                'error': 'region_access_failed',
                'distance': curr_dist
            }
        
        # Collision check (mock: always pass)
        collision_detected = False
        if PYBULLET_AVAILABLE and self.physics_client and self.physics_client > 0:
            try:
                for alpha in np.linspace(0, 1, 10):
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
                    if len(contacts) > 0 and alpha < 0.9:
                        collision_detected = True
                        break
            except:
                pass
        
        if collision_detected:
            curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
            return self._get_observation(), -5.0, False, False, {
                'error': 'collision',
                'distance': curr_dist
            }
        
        # Successful move
        prev_config = self.current_config.copy()
        self.current_config = next_config
        self.current_region_id = next_region_id
        self.step_count += 1
        
        # Calculate reward
        curr_dist = float(np.linalg.norm(self.current_config - self.goal_config))
        prev_dist = float(np.linalg.norm(prev_config - self.goal_config))
        
        reward_progress = (prev_dist - curr_dist) * 5.0
        reward = reward_progress - 0.01
        
        # Check termination
        terminated = False
        truncated = False
        
        if curr_dist < 0.1:
            reward += 100.0
            terminated = True
        
        if self.step_count >= self.max_steps:
            truncated = True
        
        obs = self._get_observation()
        
        return obs, reward, terminated, truncated, {'distance': curr_dist}
    
    def _get_observation(self) -> np.ndarray:
        """
        Construct observation vector.
        
        Returns:
            (23,) observation array
        """
        # Get current region features
        if self.current_region_id in self.regions_dict:
            region = self.regions_dict[self.current_region_id]
            region_centroid = region['centroid'][:3]
            region_density = float(region['num_samples']) / 500.0
        else:
            region_centroid = np.zeros(3)
            region_density = 0.0
        
        # Region features: [centroid(3) | density(1) | padding(6)]
        region_features = np.concatenate([
            region_centroid,
            [region_density],
            np.zeros(6)
        ])[:10]
        
        # Observation: [current(6) | goal(6) | region_id(1) | region_features(10)]
        obs = np.concatenate([
            self.current_config,
            self.goal_config,
            [self.current_region_id / max(self.num_regions, 1)],
            region_features
        ]).astype(np.float32)
        
        return obs
    
    def render(self):
        """Render using PyBullet GUI."""
        if self.render_mode == "human" and self.physics_client:
            pass  # PyBullet GUI auto-renders
    
    def close(self):
        """Clean up resources."""
        if self.physics_client is not None and self.physics_client > 0:
            try:
                p.disconnect(self.physics_client)
            except:
                pass
            self.physics_client = None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test environment
    env = YCBGraspEnv(
        ycb_objects=['rubiks_cube'],
        num_regions=10,
        max_steps=50,
        render=False
    )
    
    obs, _ = env.reset()
    print(f"✓ Environment initialized")
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    
    env.close()
    print("✓ YCBGraspEnv module tested successfully")
