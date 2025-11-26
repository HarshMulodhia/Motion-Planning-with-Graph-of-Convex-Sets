# src/gripper_config.py
import numpy as np
from dataclasses import dataclass

@dataclass
class GripperConfig:
    """6D gripper configuration: 3D position + 3D rotation"""
    x: float  # End-effector X position
    y: float  # End-effector Y position
    z: float  # End-effector Z position
    roll: float   # Rotation around X-axis
    pitch: float  # Rotation around Y-axis
    yaw: float    # Rotation around Z-axis
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z, self.roll, self.pitch, self.yaw])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'GripperConfig':
        return GripperConfig(*arr)

class ConfigurationSpaceSampler:
    """Sample collision-free 6D gripper configurations"""
    
    def __init__(self, workspace_bounds: dict, object_id: int, physics_client: int):
        """
        Args:
            workspace_bounds: {'x': (min, max), 'y': (min, max), 'z': (min, max)}
            object_id: PyBullet object ID
            physics_client: PyBullet client ID
        """
        self.workspace_bounds = workspace_bounds
        self.object_id = object_id
        self.physics_client = physics_client
        self.collision_shape_id = None
        
    def sample_random_config(self) -> GripperConfig:
        """Sample random 6D configuration"""
        x = np.random.uniform(*self.workspace_bounds['x'])
        y = np.random.uniform(*self.workspace_bounds['y'])
        z = np.random.uniform(*self.workspace_bounds['z'])
        
        # Sample random orientations (Euler angles)
        roll = np.random.uniform(0, 2*np.pi)
        pitch = np.random.uniform(0, np.pi)
        yaw = np.random.uniform(0, 2*np.pi)
        
        return GripperConfig(x, y, z, roll, pitch, yaw)
    
    def is_collision_free(self, config: GripperConfig, gripper_id: int) -> bool:
        """Check if configuration is collision-free using PyBullet"""
        import pybullet as p
        
        # Move gripper to configuration
        p.resetBasePositionAndOrientation(
            gripper_id,
            [config.x, config.y, config.z],
            p.getQuaternionFromEuler([config.roll, config.pitch, config.yaw]),
            physicsClientId=self.physics_client
        )
        
        # Check collision with object
        contact_points = p.getContactPoints(
            bodyA=gripper_id,
            bodyB=self.object_id,
            physicsClientId=self.physics_client
        )
        
        # Collision-free if no contact points
        return len(contact_points) == 0
    
    def sample_free_configurations(self, num_samples: int = 5000, 
                                   gripper_id: int = None) -> np.ndarray:
        """Sample collision-free configurations"""
        free_configs = []
        attempts = 0
        max_attempts = num_samples * 50  # Allow many failed attempts
        
        while len(free_configs) < num_samples and attempts < max_attempts:
            config = self.sample_random_config()
            
            if self.is_collision_free(config, gripper_id):
                free_configs.append(config.to_array())
            
            attempts += 1
        
        print(f"Sampled {len(free_configs)}/{num_samples} collision-free configs")
        print(f"Success rate: {100*len(free_configs)/attempts:.1f}%")
        
        return np.array(free_configs)
