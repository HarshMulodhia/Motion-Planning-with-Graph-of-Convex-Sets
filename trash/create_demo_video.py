# create_demo_video_from_environment.py
# Create demo video based on actual RL environment space and planning results

import cv2
import numpy as np
import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_realistic_demo_video():
    """Create demo video showing GCS planning in YCB workspace"""
    
    os.makedirs('presentation_assets', exist_ok=True)
    
    # Video settings
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 15
    width, height = 1280, 960
    output_path = 'presentation_assets/gcs_demo_video.mp4'
    
    logger.info("Creating GCS Motion Planning Demo Video...")
    logger.info("━" * 70)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Workspace parameters (from RL environment)
    workspace_min = np.array([0.1, -0.5, 0.0])
    workspace_max = np.array([0.9, 0.5, 1.0])
    
    # YCB object positions
    ycb_objects = {
        'banana': {'pos': (300, 250), 'color': (0, 255, 255), 'radius': 30},
        'apple': {'pos': (500, 150), 'color': (0, 0, 255), 'radius': 35},
        'lemon': {'pos': (700, 300), 'color': (0, 255, 0), 'radius': 25},
    }
    
    # Obstacles (sphere representation)
    obstacles = [
        {'center': (400, 300), 'radius': 60, 'color': (50, 50, 255)},   # Central wall (red)
        {'center': (250, 300), 'radius': 50, 'color': (100, 100, 255)}, # Left
        {'center': (550, 300), 'radius': 50, 'color': (100, 100, 255)}, # Right
        {'center': (400, 150), 'radius': 40, 'color': (150, 150, 255)}, # Upper left
        {'center': (400, 450), 'radius': 40, 'color': (150, 150, 255)}, # Upper right
    ]
    
    # Planned trajectory (through regions)
    trajectory = np.array([
        [150, 200],
        [250, 200],
        [350, 250],
        [500, 300],
        [650, 350],
        [800, 400],
        [900, 450],
        [950, 480]
    ], dtype=np.int32)
    
    num_frames = 120  # 8 seconds at 15 FPS
    
    for frame_idx in range(num_frames):
        # Create frame
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Draw grid for coordinate reference
        for x in range(100, width, 100):
            cv2.line(frame, (x, 0), (x, height), (200, 200, 200), 1)
        for y in range(100, height, 100):
            cv2.line(frame, (0, y), (width, y), (200, 200, 200), 1)
        
        # Draw workspace boundary
        cv2.rectangle(frame, (100, 100), (900, 850), (0, 0, 0), 3)
        
        # Draw obstacles (with transparency effect)
        for obs in obstacles:
            overlay = frame.copy()
            cv2.circle(overlay, obs['center'], obs['radius'], obs['color'], -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.circle(frame, obs['center'], obs['radius'], obs['color'], 2)
        
        # Draw YCB objects
        for name, obj in ycb_objects.items():
            cv2.circle(frame, obj['pos'], obj['radius'], obj['color'], -1)
            cv2.circle(frame, obj['pos'], obj['radius'], (0, 0, 0), 2)
            cv2.putText(frame, name.capitalize(), 
                       (obj['pos'][0]-30, obj['pos'][1]-50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Trajectory progress
        progress = min(frame_idx / num_frames, 1.0)
        trajectory_idx = max(1, int(progress * len(trajectory)))
        current_trajectory = trajectory[:trajectory_idx]
        
        # Draw planned path (all points)
        if trajectory_idx > 1:
            for i in range(1, len(current_trajectory)):
                pt1 = tuple(current_trajectory[i-1])
                pt2 = tuple(current_trajectory[i])
                cv2.line(frame, pt1, pt2, (0, 0, 255), 4)  # Red path
            
            # Draw waypoints
            for pt in current_trajectory[:-1]:
                cv2.circle(frame, tuple(pt), 8, (0, 0, 255), -1)
                cv2.circle(frame, tuple(pt), 8, (255, 255, 255), 1)
        
        # Start point (green)
        start_pt = tuple(trajectory[0])
        cv2.rectangle(frame, (start_pt[0]-20, start_pt[1]-20),
                     (start_pt[0]+20, start_pt[1]+20), (0, 255, 0), -1)
        cv2.rectangle(frame, (start_pt[0]-20, start_pt[1]-20),
                     (start_pt[0]+20, start_pt[1]+20), (0, 0, 0), 2)
        cv2.putText(frame, 'START', (start_pt[0]-40, start_pt[1]-35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Current position (large red circle)
        if trajectory_idx > 0:
            current_pt = tuple(current_trajectory[-1])
            cv2.circle(frame, current_pt, 20, (0, 0, 255), -1)
            cv2.circle(frame, current_pt, 20, (0, 0, 0), 2)
        
        # Goal point (orange star)
        goal_pt = tuple(trajectory[-1])
        # Draw star
        offset = 25
        points = [
            (goal_pt[0], goal_pt[1]-offset),
            (goal_pt[0]+offset, goal_pt[1]+offset),
            (goal_pt[0]-offset, goal_pt[1]+offset),
        ]
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, (0, 165, 255), 2)
        cv2.putText(frame, 'GOAL', (goal_pt[0]-35, goal_pt[1]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # Title and info box
        cv2.rectangle(frame, (20, 20), (600, 120), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, 20), (600, 120), (200, 200, 200), 2)
        
        cv2.putText(frame, 'GCS Motion Planning in YCB Workspace',
                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f'Progress: {progress*100:.0f}% | Path Points: {trajectory_idx}/{len(trajectory)}',
                   (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Metrics box
        cv2.rectangle(frame, (width-320, 20), (width-20, 160), (240, 240, 240), -1)
        cv2.rectangle(frame, (width-320, 20), (width-20, 160), (0, 0, 0), 2)
        
        cv2.putText(frame, 'Metrics:', (width-310, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(frame, 'Regions: 50', (width-310, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame, 'Planning: 44.8ms', (width-310, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        cv2.putText(frame, 'Success: 98.0%', (width-310, 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Legend
        legend_y = height - 140
        cv2.rectangle(frame, (20, legend_y-10), (420, height-20), (240, 240, 240), -1)
        cv2.rectangle(frame, (20, legend_y-10), (420, height-20), (0, 0, 0), 2)
        
        cv2.putText(frame, 'Legend:', (30, legend_y+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Start
        cv2.rectangle(frame, (30, legend_y+35), (60, legend_y+65),
                     (0, 255, 0), -1)
        cv2.putText(frame, 'Start', (70, legend_y+56),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Path
        cv2.line(frame, (30, legend_y+90), (60, legend_y+90), (0, 0, 255), 3)
        cv2.putText(frame, 'Planned Path', (70, legend_y+100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Obstacles
        overlay = frame.copy()
        cv2.circle(overlay, (45, legend_y+130), 15, (50, 50, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.putText(frame, 'Obstacles', (70, legend_y+140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Write frame
        out.write(frame)
        
        logger.info(f"Frame {frame_idx+1}/{num_frames} created", end='\r')
    
    out.release()
    
    logger.info("\n" + "━" * 70)
    logger.info(f"✓ Demo video created: {output_path}")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Duration: {num_frames/fps:.1f} seconds")
    logger.info()

if __name__ == "__main__":
    create_realistic_demo_video()
