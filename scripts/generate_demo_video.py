"""
Generate GCS-Guided Deep RL Demo Video

Creates a 60-second visualization of robot manipulation with GCS regions,
trajectories, and RL planning. Outputs MP4 video file.

Note: Requires matplotlib, cv2, pybullet, numpy, scipy
"""

import os
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import visualization dependencies
try:
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    import cv2
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization dependency missing: {e}")
    VISUALIZATION_AVAILABLE = False

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


class DemoVisualizer:
    """Generate 3D visualization frames and compile into video."""

    def __init__(self, output_dir: str = "demo_frames", fps: int = 30, duration: int = 60):
        """
        Initialize demo visualizer.

        Args:
            output_dir: Directory to save frames
            fps: Frames per second for output video
            duration: Duration in seconds
        """
        if not VISUALIZATION_AVAILABLE or not PYBULLET_AVAILABLE:
            raise RuntimeError("Visualization dependencies not available")

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

        logger.info(f"[Demo] Initialized visualizer: {fps}fps, {duration}s, {self.total_frames} frames")

    def setup_scene(self):
        """Create cluttered manipulation scene."""
        logger.info("[Demo] Setting up scene...")

        # Ground
        p.loadURDF("plane.urdf", [0, 0, 0])

        # Load YCB object (chips can) - use basic box if not available
        try:
            self.object_id = p.createMultiBody(
                baseMass=0.2,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=0.04, height=0.1
                ),
                basePosition=[0, 0, 0.5]
            )
        except Exception as e:
            logger.warning(f"Failed to load YCB model: {e}, using simple cylinder")
            self.object_id = p.createMultiBody(
                baseMass=0.2,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.05]
                ),
                basePosition=[0, 0, 0.5]
            )

        # Load gripper
        try:
            self.gripper_id = p.loadURDF("data/gripper.urdf", [0, 0, 1])
        except Exception as e:
            logger.warning(f"Gripper URDF not found: {e}")
            self.gripper_id = p.createMultiBody(
                baseMass=0.3,
                baseCollisionShapeIndex=p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=[0.02, 0.02, 0.04]
                ),
                basePosition=[0, 0, 1]
            )

        # Add clutter obstacles
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

        logger.info("✓ Scene setup complete")

    def generate_convex_regions(self, num_regions: int = 5) -> list:
        """Generate synthetic convex regions for visualization."""
        regions = []
        center = np.array([0, 0, 0.75])
        radius = 0.4

        for i in range(num_regions):
            theta = 2 * np.pi * i / num_regions
            phi = np.pi / 4

            region_center = center + radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])

            # Create small convex region
            region_points = []
            for j in range(8):
                angle = 2 * np.pi * j / 8
                point = region_center + 0.1 * np.array([
                    np.cos(angle),
                    np.sin(angle),
                    np.random.randn() * 0.05
                ])
                region_points.append(point)

            regions.append(np.array(region_points))

        return regions

    def render_3d_frame(self, frame_num: int, regions: list, selected_region_id: int) -> str:
        """
        Render 3D frame with regions and trajectory.

        Args:
            frame_num: Frame number
            regions: List of convex region points
            selected_region_id: Currently active region

        Returns:
            Path to saved frame
        """
        fig = plt.figure(figsize=(16, 9), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        # Set limits
        ax.set_xlim([-0.6, 0.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_zlim([0, 1.2])

        # Ground plane
        xx, yy = np.meshgrid(
            np.linspace(-0.6, 0.6, 10),
            np.linspace(-0.6, 0.6, 10)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

        # Object
        obj_pos, _ = p.getBasePositionAndOrientation(self.object_id)
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = 0.05 * np.outer(np.cos(u), np.sin(v)) + obj_pos[0]
        y = 0.05 * np.outer(np.sin(u), np.sin(v)) + obj_pos[1]
        z = 0.05 * np.outer(np.ones(np.size(u)), np.cos(v)) + obj_pos[2]
        ax.plot_surface(x, y, z, color='red', alpha=0.7)

        # Obstacles
        for obs_id in self.obstacle_ids:
            pos, _ = p.getBasePositionAndOrientation(obs_id)
            corners = []
            for dx in [-0.08, 0.08]:
                for dy in [-0.08, 0.08]:
                    for dz in [-0.15, 0.15]:
                        corners.append([pos[0] + dx, pos[1] + dy, pos[2] + dz])
            corners = np.array(corners)

            for i in range(8):
                for j in range(i + 1, 8):
                    if np.sum(np.abs(corners[i] - corners[j]) < 0.2) == 2:
                        ax.plot3D(*zip(corners[i], corners[j]), 'k-', alpha=0.3, linewidth=1)

        # Convex regions
        for region_id, region in enumerate(regions):
            if len(region) > 3:
                try:
                    hull = ConvexHull(region)
                    color = 'cyan' if region_id == selected_region_id else 'blue'
                    alpha = 0.4 if region_id == selected_region_id else 0.15

                    for simplex in hull.simplices:
                        triangle = region[simplex]
                        poly = [[triangle, triangle, triangle]]
                        ax.add_collection3d(Poly3DCollection(
                            poly, alpha=alpha, facecolor=color, edgecolor='lightblue'
                        ))
                except:
                    pass

        # Trajectory
        path_z_offset = np.linspace(1.2, 0.6, len(regions))
        path_points = np.array([
            [0, 0, path_z_offset[i]] for i in range(len(regions))
        ])

        current_region = min(
            int((frame_num / self.total_frames) * len(regions)),
            len(regions) - 1
        )

        if current_region < len(regions) - 1:
            ax.plot3D(
                [path_points[current_region, 0], path_points[current_region + 1, 0]],
                [path_points[current_region, 1], path_points[current_region + 1, 1]],
                [path_points[current_region, 2], path_points[current_region + 1, 2]],
                'g-', linewidth=4, alpha=0.8, label='RL-Planned Path'
            )

        ax.plot3D(
            path_points[:, 0], path_points[:, 1], path_points[:, 2],
            'g--', linewidth=2, alpha=0.5
        )

        # Gripper
        gripper_pos, _ = p.getBasePositionAndOrientation(self.gripper_id)
        ax.scatter(*gripper_pos, color='green', s=200, marker='s',
                  label='Gripper', edgecolors='darkgreen', linewidth=2)

        # Target
        ax.scatter(*obj_pos, color='red', s=150, marker='o',
                  label='Target Object', edgecolors='darkred', linewidth=2)

            # Labels and title
        ax.set_xlabel('X (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (m)', fontsize=12, fontweight='bold')
        ax.set_zlabel('Z (m)', fontsize=12, fontweight='bold')
        
        progress = int((frame_num / self.total_frames) * 100)
        ax.set_title(
            f'GCS-Guided Deep RL for Manipulation\nPlanning Progress: {progress}%',
            fontsize=16, fontweight='bold', pad=20
        )
        
        ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
        
        # Info text box
        info_text = (
            f"Regions: {len(regions)} | "
            f"Active: {selected_region_id + 1}/{len(regions)} | "
            f"Frame: {frame_num}/{self.total_frames}"
        )
        ax.text2D(0.02, 0.02, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Animated camera angle
        azim = 45 + (frame_num / self.total_frames) * 90
        elev = 30 + (frame_num / self.total_frames) * 20
        ax.view_init(elev=elev, azim=azim)
        
        # Save frame to file
        frame_path = os.path.join(self.output_dir, f'frame_{frame_num:04d}.png')
        plt.tight_layout()
        plt.savefig(frame_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return frame_path


    def create_video_from_frames(self) -> bool:
        """Compile frames into MP4 video."""
        logger.info("[Demo] Compiling video from frames...")

        frame_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.png')])

        if not frame_files:
            logger.error("No frames found!")
            return False

        # Read first frame for dimensions
        first_frame = cv2.imread(os.path.join(self.output_dir, frame_files[0]))
        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join('results', 'demo_video.mp4')
        os.makedirs('results', exist_ok=True)

        out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(self.output_dir, frame_file)
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()
        logger.info(f"✓ Video saved: {output_path} ({len(frame_files)} frames @ {self.fps}fps)")
        return True

    def generate_video(self):
        """Generate all frames and compile into video."""
        logger.info("=" * 70)
        logger.info("GCS-Guided Deep RL Demo Video Generator")
        logger.info("=" * 70)

        # Setup scene
        logger.info("\n[1/4] Setting up manipulation scene...")
        self.setup_scene()

        # Generate regions
        logger.info("[2/4] Generating convex regions...")
        regions = self.generate_convex_regions(num_regions=5)

        # Render frames
        logger.info(f"[3/4] Rendering {self.total_frames} frames...")
        for frame_num in range(self.total_frames):
            selected_region = min(
                int((frame_num / self.total_frames) * len(regions)),
                len(regions) - 1
            )
            self.render_3d_frame(frame_num, regions, selected_region)

            if (frame_num + 1) % 30 == 0:
                progress = (frame_num + 1) / self.total_frames * 100
                logger.info(f"  Progress: {progress:.1f}% ({frame_num + 1}/{self.total_frames})")

        # Create video
        logger.info("[4/4] Compiling video...")
        success = self.create_video_from_frames()

        logger.info("\n" + "=" * 70)
        if success:
            logger.info("✓ Demo video created successfully!")
            logger.info("✓ Output: results/demo_video.mp4")
        else:
            logger.error("✗ Failed to create video")
        logger.info("=" * 70)

        # Cleanup
        p.disconnect()


def generate_demo(fps: int = 30, duration: int = 60, output_dir: str = "demo_frames"):
    """
    Main function to generate demo video.

    Args:
        fps: Frames per second
        duration: Duration in seconds
        output_dir: Output directory for frames
    """
    try:
        visualizer = DemoVisualizer(fps=fps, duration=duration, output_dir=output_dir)
        visualizer.generate_video()
        return True
    except Exception as e:
        logger.error(f"Failed to generate demo: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        success = generate_demo(fps=30, duration=60)
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
