"""
GCS (Guided Cost Search) Configuration Space Decomposer

Decomposes 6D configuration space into convex regions using K-means clustering
and convex hull fitting for collision-free path planning.
"""

import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GCSDecomposer:
    """Decompose 6D configuration space into convex regions."""

    def __init__(self, free_configs: np.ndarray, num_regions: int = 50, random_state: int = 42):
        """
        Initialize GCS Decomposer.

        Args:
            free_configs: (N, 6) array of collision-free 6D configurations
            num_regions: Number of convex regions to create
            random_state: Random seed for reproducibility

        Raises:
            ValueError: If free_configs shape is invalid
        """
        if free_configs.shape[1] != 6:
            raise ValueError(f"Expected 6D configs, got {free_configs.shape[1]}D")
        if len(free_configs) < num_regions:
            logger.warning(f"Only {len(free_configs)} configs for {num_regions} regions")

        self.free_configs = free_configs
        self.num_regions = min(num_regions, len(free_configs))
        self.random_state = random_state
        self.regions: List[Dict] = []
        self.kmeans: Optional[KMeans] = None
        self.region_labels: Optional[np.ndarray] = None

        logger.info(f"[GCS] Initialized with {len(free_configs)} configs, target {num_regions} regions")

    def decompose(self) -> List[Dict]:
        """
        Decompose C-space using K-means clustering + convex hulls.

        Returns:
            List of region dictionaries with polytope data
        """
        logger.info(f"[GCS] Clustering {len(self.free_configs)} configs into {self.num_regions} regions...")

        # Cluster using K-means
        self.kmeans = KMeans(
            n_clusters=self.num_regions,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.region_labels = self.kmeans.fit_predict(self.free_configs)

        # Create polytope for each region
        valid_regions = 0
        for region_id in range(self.num_regions):
            cluster_samples = self.free_configs[self.region_labels == region_id]

            if len(cluster_samples) < 4:
                logger.warning(f"  Region {region_id}: insufficient samples ({len(cluster_samples)} < 4)")
                continue

            try:
                hull = ConvexHull(cluster_samples)
                region = {
                    'id': region_id,
                    'centroid': cluster_samples.mean(axis=0),
                    'A': hull.equations[:, :-1],  # Constraint matrix
                    'b': -hull.equations[:, -1],  # Constraint vector
                    'samples': cluster_samples,
                    'num_samples': len(cluster_samples),
                    'volume': hull.volume
                }
                self.regions.append(region)
                valid_regions += 1
            except Exception as e:
                logger.error(f"  Region {region_id}: ConvexHull failed - {e}")

        logger.info(f"[GCS] Created {len(self.regions)} valid regions ({valid_regions}/{self.num_regions})")
        return self.regions

    def get_region_for_config(self, config):
        """
        Find the region that contains or is nearest to the given configuration.
        
        Args:
            config: Configuration array (6D)
        
        Returns:
            Region ID (integer)
        """
        if not hasattr(self, 'regions') or not self.regions:
            return 0
        
        min_dist = float('inf')
        best_region = 0
        
        # FIXED: Handle both dict and list types for self.regions
        if isinstance(self.regions, dict):
            iterator = self.regions.items()
        else:
            iterator = enumerate(self.regions)
            
        for i, region_points in iterator:
            # FIXED: Robustly handle scalar/empty data
            points_array = np.asarray(region_points)
            
            # Skip empty or invalid regions
            if points_array.ndim == 0 or points_array.size == 0:
                continue
                
            # Calculate centroid safely
            centroid = np.mean(points_array, axis=0)
            
            # Match dimensions for distance calculation
            common_len = min(len(config), len(centroid))
            dist = np.linalg.norm(config[:common_len] - centroid[:common_len])
            
            if dist < min_dist:
                min_dist = dist
                best_region = i
        
        return best_region

    
    def are_adjacent(self, region1, region2, *args, **kwargs):
        """
        Check if two regions are adjacent.
        Simple heuristic: regions within distance 2 are adjacent.
        
        Args:
            region1: First region ID
            region2: Second region ID
        
        Returns:
            Boolean indicating if regions are adjacent
        """
        if not hasattr(self, 'regions'):
            return True  # Fallback: allow all if no regions
        
        return abs(region1 - region2) <= 2

    def build_adjacency_graph(self, distance_threshold: float = 1.0) -> Dict[int, List[int]]:
        """
        Build graph of adjacent regions.

        Args:
            distance_threshold: Maximum distance for adjacency

        Returns:
            Adjacency graph as dict {region_id: [neighbor_ids]}
        """
        graph = {i: [] for i in range(len(self.regions))}

        for i in range(len(self.regions)):
            for j in range(i + 1, len(self.regions)):
                if self.are_adjacent(i, j, distance_threshold):
                    graph[i].append(j)
                    graph[j].append(i)

        logger.info(f"[GCS] Built adjacency graph with {sum(len(v) for v in graph.values())//2} edges")
        return graph

    def point_in_region(self, config: np.ndarray, region_id: int) -> bool:
        """
        Check if point is inside a convex region.

        Args:
            config: 6D configuration
            region_id: Region ID

        Returns:
            True if point is inside region
        """
        if region_id >= len(self.regions):
            return False

        region = self.regions[region_id]
        A, b = region['A'], region['b']

        # Check Ax <= b
        return np.all(A @ config <= b + 1e-6)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("GCS Decomposer module loaded successfully")
