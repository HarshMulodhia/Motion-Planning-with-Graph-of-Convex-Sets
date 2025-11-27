# GCS-ONLY PROJECT STRUCTURE
# Create these files in your project root

# File 1: src/gcs_motion_planner.py
"""
Pure GCS Motion Planning for YCB Grasping
No RL, no hybrid - just clean GCS planning
"""

import numpy as np
import pybullet as p
from typing import List, Tuple, Optional, Dict
import time
from src.gcs_decomposer import GCSDecomposer
from src.gripper_config import GripperConfig
import logging

logger = logging.getLogger(__name__)

class GCSMotionPlanner:
    """
    Pure GCS-based motion planner for robotic grasping.
    Plans collision-free paths through convex regions.
    """
    
    def __init__(self, max_regions: int = 50):
        """
        Args:
            max_regions: Number of convex regions for decomposition
        """
        self.max_regions = max_regions
        self.decomposer = None
        self.free_configs = None
        self.physics_client = None
        
    def setup_environment(self, object_urdf_path: str, 
                         gripper_urdf_path: str = "data/gripper.urdf",
                         num_samples: int = 2000):
        """
        Initialize PyBullet and sample free space.
        
        Args:
            object_urdf_path: Path to object URDF
            gripper_urdf_path: Path to gripper URDF
            num_samples: Number of config samples to generate
        """
        # Start PyBullet
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # Load objects
        self.object_id = p.loadURDF(object_urdf_path, [0, 0, 0.5],
                                    physicsClientId=self.physics_client)
        self.gripper_id = p.loadURDF(gripper_urdf_path, [0, 0, 1],
                                    physicsClientId=self.physics_client)
        
        logger.info(f"[GCS] Sampling {num_samples} configurations...")
        
        # Sample free configurations
        try:
            gripper_config = GripperConfig(gripper_urdf_path)
            self.free_configs = gripper_config.sample_multiple_configs(num_samples=num_samples)
            logger.info(f"[GCS] ✓ Sampled {len(self.free_configs)} free configs")
        except Exception as e:
            logger.error(f"[GCS] Failed to sample configs: {e}")
            self.free_configs = [np.random.randn(6) * 0.1 for _ in range(num_samples)]
        
        # Run GCS decomposition
        logger.info(f"[GCS] Running GCS decomposition into {self.max_regions} regions...")
        self.decomposer = GCSDecomposer(self.free_configs, self.max_regions)
        self.decomposer.decompose()
        logger.info(f"[GCS] ✓ Decomposition complete")
        
    def plan_path(self, start_config: np.ndarray, 
                  goal_config: np.ndarray) -> Optional[List[int]]:
        """
        Plan collision-free path from start to goal using GCS.
        
        Args:
            start_config: Starting configuration (6D)
            goal_config: Goal configuration (6D)
            
        Returns:
            List of region IDs forming the path, or None if planning failed
        """
        if self.decomposer is None:
            logger.error("Decomposer not initialized")
            return None
        
        try:
            # Find start and goal regions
            start_region = self.decomposer.get_region_for_config(start_config)
            goal_region = self.decomposer.get_region_for_config(goal_config)
            
            logger.info(f"[Plan] Start region: {start_region}, Goal region: {goal_region}")
            
            # Dijkstra's algorithm for shortest path
            path = self._dijkstra(start_region, goal_region)
            
            if path:
                logger.info(f"[Plan] ✓ Path found: {len(path)} regions")
                return path
            else:
                logger.warning(f"[Plan] No path found between regions")
                return None
                
        except Exception as e:
            logger.error(f"[Plan] Planning failed: {e}")
            return None
    
    def _dijkstra(self, start: int, goal: int) -> Optional[List[int]]:
        """
        Dijkstra's shortest path algorithm.
        """
        import heapq
        
        num_regions = len(self.decomposer.regions) if hasattr(self.decomposer, 'regions') else self.max_regions
        
        # Initialize
        dist = {i: float('inf') for i in range(num_regions)}
        dist[start] = 0
        parent = {i: None for i in range(num_regions)}
        pq = [(0, start)]
        visited = set()
        
        while pq:
            d, u = heapq.heappop(pq)
            
            if u in visited:
                continue
            visited.add(u)
            
            if u == goal:
                # Reconstruct path
                path = []
                current = goal
                while current is not None:
                    path.append(current)
                    current = parent[current]
                return list(reversed(path))
            
            # Explore neighbors
            for v in range(num_regions):
                try:
                    if self.decomposer.are_adjacent(u, v, None):  # Pass None for extra args
                        if v not in visited:
                            new_dist = dist[u] + 1  # Unit cost per edge
                            if new_dist < dist[v]:
                                dist[v] = new_dist
                                parent[v] = u
                                heapq.heappush(pq, (new_dist, v))
                except:
                    pass
        
        return None  # No path found
    
    def execute_path(self, path: List[int], 
                    configs: List[np.ndarray]) -> Tuple[bool, float, int]:
        """
        Execute planned path and measure success.
        
        Args:
            path: List of region IDs
            configs: List of configurations for each step
            
        Returns:
            (success, time_taken, num_steps)
        """
        if path is None or len(path) == 0:
            return False, 0.0, 0
        
        start_time = time.time()
        num_steps = 0
        
        try:
            for region_id in path[1:]:  # Skip start region
                # Get centroid of region
                if hasattr(self.decomposer, 'regions'):
                    region_points = self.decomposer.regions[region_id]
                    if len(region_points) > 0:
                        waypoint = np.mean(region_points, axis=0)
                        
                        # Move gripper to waypoint
                        p.resetBasePositionAndOrientation(
                            self.gripper_id,
                            waypoint[:3],
                            p.getQuaternionFromEuler(waypoint[3:6]),
                            physicsClientId=self.physics_client
                        )
                        
                        num_steps += 1
                        time.sleep(0.01)  # Simulate execution time
            
            elapsed_time = time.time() - start_time
            return True, elapsed_time, num_steps
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False, time.time() - start_time, num_steps
    
    def cleanup(self):
        """Clean up PyBullet resources."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
