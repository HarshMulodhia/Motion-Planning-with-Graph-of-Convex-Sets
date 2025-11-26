# src/gcs_trajectory_optimizer.py
import numpy as np
from typing import List, Optional
from src.gcs_decomposer import GCSDecomposer

class GCSTrajectoryOptimizer:
    """Find optimal path through GCS regions for collision-free grasp approach"""
    
    def __init__(self, decomposer: GCSDecomposer):
        self.decomposer = decomposer
        self.adjacency_graph = decomposer.build_adjacency_graph()
    
    def dijkstra_path(self, start_region_id: int, goal_region_id: int) -> List[int]:
        """Find shortest path through region adjacency graph using Dijkstra"""
        import heapq
        
        # Priority queue: (cost, current_region)
        pq = [(0, start_region_id)]
        visited = set()
        came_from = {}
        costs = {start_region_id: 0}
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == goal_region_id:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start_region_id)
                return list(reversed(path))
            
            # Explore neighbors
            for neighbor in self.adjacency_graph.get(current, []):
                if neighbor in visited:
                    continue
                
                # Cost = distance between region centroids
                edge_cost = np.linalg.norm(
                    self.decomposer.regions[current]['centroid'] - 
                    self.decomposer.regions[neighbor]['centroid']
                )
                
                new_cost = current_cost + edge_cost
                
                if neighbor not in costs or new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))
        
        return None  # No path found
    
    def generate_trajectory_waypoints(self, region_path: List[int], 
                                     num_waypoints_per_region: int = 3) -> np.ndarray:
        """Generate smooth trajectory by interpolating through region centroids"""
        waypoints = []
        
        for region_id in region_path:
            centroid = self.decomposer.regions[region_id]['centroid']
            
            # Add multiple waypoints in/near the region
            if len(waypoints) == 0:
                waypoints.append(centroid)
            else:
                # Interpolate from previous to current
                prev = waypoints[-1]
                for alpha in np.linspace(0.25, 1.0, num_waypoints_per_region):
                    wp = (1 - alpha) * prev + alpha * centroid
                    waypoints.append(wp)
        
        return np.array(waypoints)
