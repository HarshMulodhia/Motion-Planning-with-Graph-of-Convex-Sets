# gcs_motion_planner_with_obstacles.py
# Enhanced GCS Motion Planning with Obstacle Avoidance

import numpy as np
import pybullet as p
from typing import List, Tuple, Optional, Dict
import time
from src.gcs_decomposer import GCSDecomposer
from src.gripper_config import GripperConfig
import logging

logger = logging.getLogger(__name__)

class ObstacleManager:
    """Manages obstacles in the configuration space"""
    
    def __init__(self, physics_client):
        self.physics_client = physics_client
        self.obstacles = []
        self.obstacle_ids = []
    
    def add_box_obstacle(self, position: np.ndarray, size: np.ndarray, name: str = "box"):
        """Add a box obstacle to the environment"""
        try:
            obstacle_id = p.loadURDF(
                "r2d2.urdf",  # Use simple object as placeholder
                position,
                p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.physics_client
            )
            self.obstacle_ids.append(obstacle_id)
            self.obstacles.append({
                'type': 'box',
                'position': position.copy(),
                'size': size.copy(),
                'id': obstacle_id,
                'name': name
            })
            logger.info(f"[Obstacles] Added box obstacle: {name}")
            return True
        except:
            logger.warning(f"[Obstacles] Failed to load obstacle URDF")
            return False
    
    def add_sphere_obstacle(self, position: np.ndarray, radius: float, name: str = "sphere"):
        """Add a sphere obstacle"""
        try:
            # Create a simple sphere collision shape
            collision_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self.physics_client
            )
            obstacle_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision_shape,
                basePosition=position,
                physicsClientId=self.physics_client
            )
            self.obstacle_ids.append(obstacle_id)
            self.obstacles.append({
                'type': 'sphere',
                'position': position.copy(),
                'radius': radius,
                'id': obstacle_id,
                'name': name
            })
            logger.info(f"[Obstacles] Added sphere obstacle: {name}")
            return True
        except Exception as e:
            logger.warning(f"[Obstacles] Failed to add sphere obstacle: {e}")
            return False
    
    def check_collision(self, config: np.ndarray, gripper_id: int, object_id: int) -> bool:
        """Check if configuration collides with any obstacle"""
        try:
            # Move gripper to configuration
            p.resetBasePositionAndOrientation(
                gripper_id,
                config[:3],
                p.getQuaternionFromEuler(config[3:6]),
                physicsClientId=self.physics_client
            )
            
            # Check collision with each obstacle
            for obstacle_id in self.obstacle_ids:
                contacts = p.getContactPoints(
                    gripper_id, obstacle_id,
                    physicsClientId=self.physics_client
                )
                if len(contacts) > 0:
                    return True
            
            return False
        except:
            return False
    
    def check_path_collision(self, start: np.ndarray, end: np.ndarray, 
                            gripper_id: int, num_checks: int = 10) -> bool:
        """Check if path between start and end collides with obstacles"""
        for alpha in np.linspace(0, 1, num_checks):
            interpolated = (1 - alpha) * start + alpha * end
            if self.check_collision(interpolated, gripper_id, None):
                return True
        return False
    
    def cleanup(self):
        """Remove all obstacles from simulation"""
        for obstacle_id in self.obstacle_ids:
            try:
                p.removeBody(obstacle_id, physicsClientId=self.physics_client)
            except:
                pass
        self.obstacle_ids.clear()
        self.obstacles.clear()


class GCSMotionPlannerWithObstacles:
    """
    Enhanced GCS motion planner with obstacle avoidance
    """
    
    def __init__(self, max_regions: int = 50, use_obstacles: bool = True):
        """
        Args:
            max_regions: Number of convex regions
            use_obstacles: Whether to include obstacles
        """
        self.max_regions = max_regions
        self.use_obstacles = use_obstacles
        self.decomposer = None
        self.free_configs = None
        self.physics_client = None
        self.obstacle_manager = None
        
    def setup_environment(self, object_urdf_path: str,
                         gripper_urdf_path: str = "data/gripper.urdf",
                         num_samples: int = 2000):
        """Initialize PyBullet and sample free space with obstacles"""
        
        # Start PyBullet
        self.physics_client = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        
        # Load objects
        self.object_id = p.loadURDF(object_urdf_path, [0, 0, 0.5],
                                    physicsClientId=self.physics_client)
        self.gripper_id = p.loadURDF(gripper_urdf_path, [0, 0, 1],
                                    physicsClientId=self.physics_client)
        
        # Initialize obstacle manager
        self.obstacle_manager = ObstacleManager(self.physics_client)
        
        # Add obstacles if enabled
        if self.use_obstacles:
            self._add_default_obstacles()
        
        logger.info(f"[GCS] Sampling {num_samples} configurations...")
        
        # Sample free configurations (avoiding obstacles)
        try:
            gripper_config = GripperConfig(gripper_urdf_path)
            all_configs = gripper_config.sample_multiple_configs(num_samples=num_samples*2)
            
            # Filter out collision configurations
            self.free_configs = []
            for config in all_configs:
                if not self.obstacle_manager.check_collision(config, self.gripper_id, self.object_id):
                    self.free_configs.append(config)
                    if len(self.free_configs) >= num_samples:
                        break
            
            logger.info(f"[GCS] ✓ Sampled {len(self.free_configs)} free configs (obstacles filtered)")
        except Exception as e:
            logger.error(f"[GCS] Failed to sample configs: {e}")
            self.free_configs = [np.random.randn(6) * 0.1 for _ in range(num_samples)]
        
        # Run GCS decomposition
        logger.info(f"[GCS] Running GCS decomposition into {self.max_regions} regions...")
        self.decomposer = GCSDecomposer(self.free_configs, self.max_regions)
        self.decomposer.decompose()
        logger.info(f"[GCS] ✓ Decomposition complete ({len(self.decomposer.regions)} regions)")
    
    def _add_default_obstacles(self):
        """Add a set of challenging obstacles to the workspace"""
        logger.info("[Obstacles] Adding default obstacles...")
        
        # Central obstacle (vertical wall)
        self.obstacle_manager.add_sphere_obstacle(
            np.array([0.0, 0.0, 0.5]),
            radius=0.3,
            name="central_wall"
        )
        
        # Side obstacles (creating narrow passage)
        self.obstacle_manager.add_sphere_obstacle(
            np.array([-0.4, 0.0, 0.5]),
            radius=0.25,
            name="left_obstacle"
        )
        
        self.obstacle_manager.add_sphere_obstacle(
            np.array([0.4, 0.0, 0.5]),
            radius=0.25,
            name="right_obstacle"
        )
        
        # Upper obstacles
        self.obstacle_manager.add_sphere_obstacle(
            np.array([0.0, 0.3, 0.8]),
            radius=0.2,
            name="upper_left_obstacle"
        )
        
        self.obstacle_manager.add_sphere_obstacle(
            np.array([0.0, -0.3, 0.8]),
            radius=0.2,
            name="upper_right_obstacle"
        )
        
        logger.info("[Obstacles] ✓ Default obstacles added (5 obstacles)")
    
    def plan_path(self, start_config: np.ndarray,
                  goal_config: np.ndarray) -> Optional[List[int]]:
        """Plan collision-free path avoiding obstacles"""
        
        if self.decomposer is None:
            logger.error("Decomposer not initialized")
            return None
        
        try:
            # Find start and goal regions
            start_region = self.decomposer.get_region_for_config(start_config)
            goal_region = self.decomposer.get_region_for_config(goal_config)
            
            logger.info(f"[Plan] Start region: {start_region}, Goal region: {goal_region}")
            
            # Dijkstra's algorithm with obstacle-aware edge checking
            path = self._dijkstra_with_obstacles(start_region, goal_region, start_config, goal_config)
            
            if path:
                logger.info(f"[Plan] ✓ Collision-free path found: {len(path)} regions")
                return path
            else:
                logger.warning(f"[Plan] No collision-free path found")
                return None
                
        except Exception as e:
            logger.error(f"[Plan] Planning failed: {e}")
            return None
    
    def _dijkstra_with_obstacles(self, start: int, goal: int,
                                 start_config: np.ndarray,
                                 goal_config: np.ndarray) -> Optional[List[int]]:
        """Dijkstra with obstacle-aware edge validation"""
        import heapq
        
        num_regions = len(self.decomposer.regions) if hasattr(self.decomposer, 'regions') else self.max_regions
        
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
            
            # Explore neighbors with obstacle checking
            for v in range(num_regions):
                try:
                    if self.decomposer.are_adjacent(u, v, None):
                        if v not in visited:
                            # Check if edge is collision-free
                            edge_valid = self._check_edge_validity(u, v)
                            
                            if edge_valid:
                                new_dist = dist[u] + 1
                                if new_dist < dist[v]:
                                    dist[v] = new_dist
                                    parent[v] = u
                                    heapq.heappush(pq, (new_dist, v))
                except:
                    pass
        
        return None
    
    def _check_edge_validity(self, region1: int, region2: int) -> bool:
        """Check if edge between regions is collision-free"""
        try:
            if not hasattr(self.decomposer, 'regions'):
                return True
            
            # Get centroids of regions
            r1_points = self.decomposer.regions[region1]
            r2_points = self.decomposer.regions[region2]
            
            if len(r1_points) == 0 or len(r2_points) == 0:
                return False
            
            r1_centroid = np.mean(r1_points, axis=0)
            r2_centroid = np.mean(r2_points, axis=0)
            
            # Check path between centroids
            return not self.obstacle_manager.check_path_collision(
                r1_centroid, r2_centroid, self.gripper_id, num_checks=5
            )
        except:
            return True
    
    def execute_path(self, path: List[int],
                    configs: List[np.ndarray]) -> Tuple[bool, float, int]:
        """Execute planned path with obstacle verification"""
        
        if path is None or len(path) == 0:
            return False, 0.0, 0
        
        start_time = time.time()
        num_steps = 0
        path_valid = True
        
        try:
            for i in range(len(path) - 1):
                region_id = path[i+1]
                
                # Get centroid of region
                if hasattr(self.decomposer, 'regions'):
                    region_points = self.decomposer.regions[region_id]
                    if len(region_points) > 0:
                        waypoint = np.mean(region_points, axis=0)
                        
                        # Check collision before moving
                        if self.obstacle_manager.check_collision(waypoint, self.gripper_id, self.object_id):
                            logger.warning(f"[Exec] Waypoint collision detected at region {region_id}")
                            path_valid = False
                            break
                        
                        # Move gripper to waypoint
                        p.resetBasePositionAndOrientation(
                            self.gripper_id,
                            waypoint[:3],
                            p.getQuaternionFromEuler(waypoint[3:6]),
                            physicsClientId=self.physics_client
                        )
                        
                        num_steps += 1
                        time.sleep(0.01)
            
            elapsed_time = time.time() - start_time
            return path_valid, elapsed_time, num_steps
            
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return False, time.time() - start_time, num_steps
    
    def get_obstacle_info(self) -> Dict:
        """Get information about obstacles"""
        if self.obstacle_manager is None:
            return {'num_obstacles': 0, 'obstacles': []}
        
        return {
            'num_obstacles': len(self.obstacle_manager.obstacles),
            'obstacles': [
                {
                    'name': obs['name'],
                    'type': obs['type'],
                    'position': obs['position'].tolist()
                }
                for obs in self.obstacle_manager.obstacles
            ]
        }
    
    def cleanup(self):
        """Clean up PyBullet resources"""
        if self.obstacle_manager is not None:
            self.obstacle_manager.cleanup()
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
