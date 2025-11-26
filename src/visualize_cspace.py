# scripts/visualize_cspace.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gripper_config import ConfigurationSpaceSampler, GripperConfig
import pybullet as p
import pybullet_data

# Setup PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf", [0, 0, 0])

# Load YCB object (example: chips can)
ycb_obj = p.loadURDF(
    "~/datasets/ycb_data/ycb_models/001_chips_can/google_16k/textured.ply",
    [0, 0, 0.5]
)

# Load gripper model
gripper = p.loadURDF("data/gripper.urdf", [0, 0, 1])

# Define workspace
workspace = {
    'x': (-0.5, 0.5),
    'y': (-0.5, 0.5),
    'z': (0.2, 1.2)
}

# Sample free configurations
sampler = ConfigurationSpaceSampler(workspace, ycb_obj, p.getPhysicsEngineClient())
free_configs = sampler.sample_free_configurations(num_samples=3000, gripper_id=gripper)

# Visualize in 3D (positions only)
fig = plt.figure(figsize=(12, 5))

# XYZ positions
ax1 = fig.add_subplot(121, projection='3d')
positions = free_configs[:, :3]
ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
           c=positions[:, 2], cmap='viridis', alpha=0.6, s=10)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Free Configuration Space (Positions)')

# Histogram of Z heights (gripper approach angles)
ax2 = fig.add_subplot(122)
ax2.hist(free_configs[:, 2], bins=50, alpha=0.7)
ax2.set_xlabel('Z Height')
ax2.set_ylabel('Count')
ax2.set_title('Distribution of Gripper Heights')

plt.tight_layout()
plt.savefig('cspace_visualization.png', dpi=150)
plt.show()

p.disconnect()
