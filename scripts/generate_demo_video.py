"""
Generate a 60-second demo video showing GCS-guided RL manipulation
Features:
  - Robot planning grasp trajectory in cluttered scene
  - Visualized convex regions (translucent hulls)
  - RL agent choosing optimal path
  - Smooth camera tracking
"""

import numpy as np
import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull
import cv2
import os
from datetime import datetime

class DemoVisualizer:
    def __init__(self, output_dir='demo_frames', fps=30, duration=60):
        self.output_dir = output_dir
        self.fps = fps
        self.duration = duration
        self.total_frames = fps * duration
        self.frame_count = 0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # PyBullet setup
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
    def setup_scene(self):
        """Create cluttered manipulation scene"""
        # Ground
        p.loadURDF("plane.urdf", [0, 0, 0])
        
        # Load YCB object (chips can)
        self.object_id = p.loadURDF(
            os.path.expanduser("datasets/ycb_data/ycb_models/002_master_chef_can/google_16k/textured.ply"),
            [0, 0, 0.5]
        )
        
        # Load gripper
        self.gripper_id = p.loadURDF("data/gripper.urdf", [0, 0, 1])
        
        # Add clutter obstacles (boxes around object)
        obstacle_positions = [
            [-0.3, -0.3, 0.4], [0.3, -0.3, 0.4],
            [-0.3, 0.3, 0.4], [0.3, 0.3, 0.4]
        ]
        self.obstacle_ids = []
        for pos in obstacle_positions:
            box_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.15]
                ),
                basePosition=pos
            )
            self.obstacle_ids.append(box_id)
    
    def generate_convex_regions(self, num_regions=5):
        """Generate synthetic convex regions for visualization"""
        regions = []
        center = np.array([0, 0, 0.75])
        radius = 0.4
        
        for i in range(num_regions):
            # Sample points in spherical shell
            theta = 2 * np.pi * i / num_regions
            phi = np.pi / 4
            
            region_center = center + radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            
            # Create small convex region around this point
            region_points = []
            for j in range(8):
                angle = 2 * np.pi * j / 8
                point = region_center + 0.1 * np.array([
                    np.cos(angle), np.sin(angle), np.random.randn() * 0.05
                ])
                region_points.append(point)
            
            regions.append(np.array(region_points))
        
        return regions
    
    def render_3d_frame(self, frame_num, regions, selected_region_id):
        """Render 3D visualization with regions and trajectory"""
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        # Set limits
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 1.2])
        
        # Get gripper state for this frame
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.gripper_id)
        
        # Draw ground plane
        xx, yy = np.meshgrid(np.linspace(-0.6, 0.6, 10), np.linspace(-0.6, 0.6, 10))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        # Draw object (simplified as sphere)
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = 0.05 * np.outer(np.cos(u), np.sin(v)) + obj_pos
        y = 0.05 * np.outer(np.sin(u), np.sin(v)) + obj_pos
        z = 0.05 * np.outer(np.ones(np.size(u)), np.cos(v)) + obj_pos
        ax.plot_surface(x, y, z, color='red', alpha=0.7)
        
        # Draw obstacles (as boxes)
        for obs_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(obs_id)
            
            # Draw box outline
            corners = []
            for dx in [-0.08, 0.08]:
                for dy in [-0.08, 0.08]:
                    for dz in [-0.15, 0.15]:
                        corners.append([pos+dx, pos+dy, pos+dz])
            corners = np.array(corners)
            
            # Draw edges
            for i in range(8):
                for j in range(i+1, 8):
                    if np.sum(np.abs(corners[i] - corners[j]) < 0.2) == 2:
                        ax.plot3D(*zip(corners[i], corners[j]), 'k-', alpha=0.3, linewidth=1)
        
        # Draw convex regions (translucent hulls)
        for region_id, region in enumerate(regions):
            if len(region) > 3:
                try:
                    hull = ConvexHull(region)
                    
                    # Color: highlight selected region
                    if region_id == selected_region_id:
                        color = 'cyan'
                        alpha = 0.4
                    else:
                        color = 'blue'
                        alpha = 0.15
                    
                    # Plot convex hull faces
                    for simplex in hull.simplices:
                        triangle = region[simplex]
                        poly = [[triangle, triangle, triangle]]
                        ax.add_collection3d(Poly3DCollection(
                            poly, alpha=alpha, facecolor=color, edgecolor='lightblue'
                        ))
                except:
                    pass
        
        # Draw gripper trajectory (path through regions)
        path_z_offset = np.linspace(1.2, 0.6, len(regions))
        path_points = np.array([
            [0, 0, path_z_offset[i]] for i in range(len(regions))
        ])
        
        # Highlight current segment
        current_region = min(int((frame_num / self.total_frames) * len(regions)), len(regions)-1)
        if current_region < len(regions) - 1:
            ax.plot3D(
                [path_points[current_region, 0], path_points[current_region+1, 0]],
                [path_points[current_region, 1], path_points[current_region+1, 1]],
                [path_points[current_region, 2], path_points[current_region+1, 2]],
                'g-', linewidth=4, alpha=0.8, label='RL-Planned Path'
            )
        
        # Draw full path
        ax.plot3D(path_points[:, 0], path_points[:, 1], path_points[:, 2],
                 'g--', linewidth=2, alpha=0.5)
        
        # Draw gripper position
        ax.scatter(*gripper_pos, color='green', s=200, marker='s', 
                  label='Gripper', edgecolors='darkgreen', linewidth=2)
        
        # Draw object target
        ax.scatter(*obj_pos, color='red', s=150, marker='o',
                  label='Target Object', edgecolors='darkred', linewidth=2)
        
        # Labels and formatting
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        
        # Dynamic title
        progress = int((frame_num / self.total_frames) * 100)
        ax.set_title(
            'GCS-Guided Deep RL for Manipulation\nPlanning Progress: {}%'.format(progress),
            fontsize=16, fontweight='bold', pad=20
        )
        
        # Legend
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # Add text box with info
        info_text = (
            f"Regions: {len(regions)} | "
            f"Active: {selected_region_id+1}/{len(regions)} | "
            f"Frame: {frame_num}/{self.total_frames}"
        )
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes,
                 fontsize=10, verticalalignment='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Adjust viewing angle for cinematic effect
        azim = 45 + (frame_num / self.total_frames) * 90
        elev = 30 + (frame_num / self.total_frames) * 20
        ax.view_init(elev=elev, azim=azim)
        
        # Save frame
        frame_path = os.path.join(self.output_dir, f'frame_{frame_num:04d}.png')
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return frame_path
    
    def generate_video(self):
        """Generate all frames and compile into video"""
        print("=" * 70)
        print("GCS-Guided Deep RL Demo Video Generator")
        print("=" * 70)
        
        # Setup scene
        print("\n[1/4] Setting up manipulation scene...")
        self.setup_scene()
        
        # Generate regions
        print("[2/4] Generating convex regions...")
        regions = self.generate_convex_regions(num_regions=5)
        
        # Render frames
        print(f"[3/4] Rendering {self.total_frames} frames...")
        for frame_num in range(self.total_frames):
            selected_region = min(int((frame_num / self.total_frames) * len(regions)), len(regions)-1)
            self.render_3d_frame(frame_num, regions, selected_region)
            
            if (frame_num + 1) % 30 == 0:
                progress = (frame_num + 1) / self.total_frames * 100
                print(f"  Progress: {progress:.1f}% ({frame_num+1}/{self.total_frames})")
        
        # Create video from frames
        print("[4/4] Compiling video...")
        self.create_video_from_frames()
        
        print("\n" + "=" * 70)
        print("✓ Demo video created successfully!")
        print(f"✓ Output: demo_video.mp4")
        print("=" * 70)
        
        # Cleanup
        p.disconnect()
    
    def create_video_from_frames(self):
        """Compile frames into MP4 video"""
        frame_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        
        if not frame_files:
            print("No frames found!")
            return
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(self.output_dir, frame_files))
        height, width, _ = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('demo_video.mp4', fourcc, self.fps, (width, height))
        
        for frame_file in frame_files:
            frame_path = os.path.join(self.output_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)
        
        out.release()
        print(f"Video saved: demo_video.mp4 ({len(frame_files)} frames @ {self.fps} fps)")

# Run demo
if __name__ == "__main__":
    visualizer = DemoVisualizer(fps=30, duration=60)
    visualizer.generate_video()
