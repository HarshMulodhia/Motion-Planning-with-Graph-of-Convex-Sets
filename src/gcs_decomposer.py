# src/gcs_decomposer.py
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from typing import List, Tuple

class GCSDecomposer:
    """Decompose 6D configuration space into convex regions"""
    
    def __init__(self, free_configs: np.ndarray, num_regions: int = 50):
        """
        Args:
            free_configs: (N, 6) array of collision-free 6D configurations
            num_regions: Number of convex regions to create
        """
        self.free_configs = free_configs
        self.num_regions = num_regions
        self.regions = []
        self.kmeans = None
        self.region_labels = None
        
    def decompose(self) -> List[dict]:
        """
        Decompose C-space using K-means clustering + convex hulls
        Returns: List of region dictionaries with polytope data
        """
        print(f"[GCS] Clustering {len(self.free_configs)} configs into {self.num_regions} regions...")
        
        # Cluster using K-means
        self.kmeans = KMeans(n_clusters=self.num_regions, random_state=42, n_init=10)
        self.region_labels = self.kmeans.fit_predict(self.free_configs)
        
        # Create polytope for each region
        for region_id in range(self.num_regions):
            cluster_samples = self.free_configs[self.region_labels == region_id]
            
            if len(cluster_samples) < 4:
                print(f"  Warning: Region {region_id} has {len(cluster_samples)} samples")
                continue
            
            # Fit convex hull to cluster
            try:
                hull = ConvexHull(cluster_samples)
                
                region = {
                    'id': region_id,
                    'centroid': cluster_samples.mean(axis=0),
                    'A': hull.equations[:, :-1],      # Constraint matrix
                    'b': -hull.equations[:, -1],      # Constraint vector
                    'samples': cluster_samples,
                    'num_samples': len(cluster_samples)
                }
                
                self.regions.append(region)
                
            except Exception as e:
                print(f"  Error computing hull for region {region_id}: {e}")
        
        print(f"[GCS] Created {len(self.regions)} valid convex regions")
        return self.regions
    
    def get_region_for_config(self, config: np.ndarray) -> int:
        """Find which region contains the configuration"""
        distances = np.linalg.norm(self.kmeans.cluster_centers_ - config, axis=1)
        return np.argmin(distances)
    
    def are_adjacent(self, region_id1: int, region_id2: int, 
                     distance_threshold: float = 1.0) -> bool:
        """Check if two regions are adjacent (close centroids)"""
        if region_id1 >= len(self.regions) or region_id2 >= len(self.regions):
            return False
        
        c1 = self.regions[region_id1]['centroid']
        c2 = self.regions[region_id2]['centroid']
        
        dist = np.linalg.norm(c1 - c2)
        return dist < distance_threshold
    
    def build_adjacency_graph(self, distance_threshold: float = 1.0) -> dict:
        """Build graph of adjacent regions"""
        graph = {i: [] for i in range(len(self.regions))}
        
        for i in range(len(self.regions)):
            for j in range(i+1, len(self.regions)):
                if self.are_adjacent(i, j, distance_threshold):
                    graph[i].append(j)
                    graph[j].append(i)
        
        return graph
    
    def point_in_region(self, config: np.ndarray, region_id: int) -> bool:
        """Check if point is inside a convex region"""
        region = self.regions[region_id]
        A, b = region['A'], region['b']
        
        # Check Ax <= b
        return np.all(A @ config <= b + 1e-6)
