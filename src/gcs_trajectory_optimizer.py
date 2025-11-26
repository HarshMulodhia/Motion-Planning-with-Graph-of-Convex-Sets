"""
GCS Trajectory Optimizer

Finds optimal collision-free trajectory through GCS regions using Dijkstra's algorithm
and generates smooth waypoint interpolation.
"""

import numpy as np
from typing import List, Optional, Dict
import heapq
import logging

logger = logging.getLogger(__name__)


class GCSTrajectoryOptimizer:
    """Find optimal path through GCS regions for collision-free grasp approach."""

    def __init__(self, decomposer):
        """
        Initialize trajectory optimizer with GCS decomposer.

        Args:
            decomposer: GCSDecomposer instance with decomposed regions
        """
        if not hasattr(decomposer, 'regions') or len(decomposer.regions) == 0:
            raise ValueError("Decomposer must have valid regions")

        self.decomposer = decomposer
        self.adjacency_graph = decomposer.build_adjacency_graph()
        logger.info(f"[GCS-Traj] Initialized optimizer for {len(decomposer.regions)} regions")

    def dijkstra_path(self, start_region_id: int, goal_region_id: int) -> Optional[List[int]]:
        """
        Find shortest path through region adjacency graph using Dijkstra.

        Args:
            start_region_id: Starting region ID
            goal_region_id: Goal region ID

        Returns:
            List of region IDs forming the path, or None if no path exists
        """
        if start_region_id == goal_region_id:
            return [start_region_id]

        # Priority queue: (cost, current_region)
        pq = [(0.0, start_region_id)]
        visited = set()
        came_from = {}
        costs = {start_region_id: 0.0}

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
                logger.info(f"[GCS-Traj] Found path: {len(path)} regions, cost={current_cost:.4f}")
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

        logger.warning(f"[GCS-Traj] No path found from {start_region_id} to {goal_region_id}")
        return None

    def generate_trajectory_waypoints(self, region_path: List[int],
                                     num_waypoints_per_region: int = 3) -> np.ndarray:
        """
        Generate smooth trajectory by interpolating through region centroids.

        Args:
            region_path: Sequence of region IDs
            num_waypoints_per_region: Number of interpolated waypoints per region

        Returns:
            (N, 6) array of trajectory waypoints
        """
        if not region_path:
            raise ValueError("Region path cannot be empty")

        waypoints = []

        for region_id in region_path:
            if region_id >= len(self.decomposer.regions):
                logger.warning(f"Region {region_id} out of bounds, skipping")
                continue

            centroid = self.decomposer.regions[region_id]['centroid']

            if len(waypoints) == 0:
                waypoints.append(centroid)
            else:
                # Interpolate from previous to current centroid
                prev = waypoints[-1]
                for alpha in np.linspace(0.25, 1.0, num_waypoints_per_region):
                    wp = (1 - alpha) * prev + alpha * centroid
                    waypoints.append(wp)

        waypoints_array = np.array(waypoints)
        logger.info(f"[GCS-Traj] Generated {len(waypoints_array)} waypoints")
        return waypoints_array

    def smooth_trajectory(self, waypoints: np.ndarray, smoothing_factor: int = 3) -> np.ndarray:
        """
        Apply simple smoothing to trajectory.

        Args:
            waypoints: Original waypoints
            smoothing_factor: Number of interpolation steps

        Returns:
            Smoothed waypoints
        """
        if len(waypoints) < 2:
            return waypoints

        smoothed = [waypoints[0]]

        for i in range(len(waypoints) - 1):
            for t in np.linspace(0, 1, smoothing_factor + 1)[1:]:
                smoothed.append((1 - t) * waypoints[i] + t * waypoints[i + 1])

        logger.info(f"[GCS-Traj] Smoothed trajectory: {len(waypoints)} -> {len(smoothed)} waypoints")
        return np.array(smoothed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GCS Trajectory Optimizer module loaded successfully")
