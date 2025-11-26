#!/usr/bin/env python3
"""
Verify YCB dataset is correctly installed
"""

import os
import sys

def verify_ycb():
    ycb_path = os.path.expanduser("~/datasets/ycb_data/ycb_models")
    
    print("="*70)
    print("YCB Dataset Verification")
    print("="*70)
    
    # Check directory exists
    if not os.path.exists(ycb_path):
        print(f"\n✗ Directory not found: {ycb_path}")
        print("\nPlease run:")
        print("  python scripts/download_ycb.py")
        return False
    
    print(f"\n✓ Found directory: {ycb_path}")
    
    # Count objects
    objects = sorted([d for d in os.listdir(ycb_path) 
                     if os.path.isdir(os.path.join(ycb_path, d))])
    
    print(f"\n✓ Found {len(objects)} objects")
    
    # Verify mesh files
    print("\nVerifying mesh files...")
    all_good = True
    
    for i, obj in enumerate(objects[:10]):
        obj_path = os.path.join(ycb_path, obj)
        
        # Look for mesh files
        mesh_found = False
        mesh_file = None
        
        for root, dirs, files in os.walk(obj_path):
            for f in files:
                if f.endswith(('.ply', '.obj', '.dae', '.urdf')):
                    mesh_found = True
                    mesh_file = os.path.join(root, f)
                    break
        
        if mesh_found:
            print(f"  ✓ {obj}: {os.path.basename(mesh_file)}")
        else:
            print(f"  ✗ {obj}: No mesh found!")
            all_good = False
    
    if len(objects) > 10:
        print(f"  ... and {len(objects)-10} more objects")
    
    if all_good:
        print("\n" + "="*70)
        print("✓ YCB Dataset Verified Successfully!")
        print("="*70)
        print(f"\nTotal objects: {len(objects)}")
        print(f"Location: {ycb_path}")
        print("\nYou can now train:")
        print("  python scripts/train_ycb_grasp.py")
        return True
    else:
        print("\n✗ Some objects are missing meshes")
        return False

if __name__ == "__main__":
    success = verify_ycb()
    sys.exit(0 if success else 1)
