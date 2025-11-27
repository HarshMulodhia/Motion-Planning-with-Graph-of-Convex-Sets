#!/usr/bin/env python3
"""
Direct YCB Dataset Downloader
Downloads from official YCB repository without external dependencies
"""

import os
import urllib.request
import tarfile
import sys
from pathlib import Path

# Configuration
YCB_BASE_URL = "https://ycb.download.robotics.ucsd.edu/byebullet/ycb/"
YCB_OUTPUT_PATH = os.path.expanduser("~/datasets/ycb_data/ycb_models")
OBJECTS_TO_DOWNLOAD = list(range(1, 21))  # Objects 1-20

def create_directories():
    """Create output directory"""
    os.makedirs(YCB_OUTPUT_PATH, exist_ok=True)
    print(f"✓ Output directory: {YCB_OUTPUT_PATH}")

def download_file(url, filepath, filename):
    """Download a file with progress bar"""
    try:
        print(f"  Downloading: {filename}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, filepath)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"✓ ({size_mb:.1f} MB)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def extract_tar_gz(filepath, extract_path):
    """Extract tar.gz file"""
    try:
        print(f"    Extracting...", end=" ", flush=True)
        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(path=extract_path)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def download_ycb_objects():
    """Download YCB objects"""
    print("="*70)
    print("YCB Dataset Direct Downloader")
    print("="*70)
    
    print(f"\n[1/3] Creating directories...")
    create_directories()
    
    print(f"\n[2/3] Downloading {len(OBJECTS_TO_DOWNLOAD)} objects...")
    print(f"      (This may take 10-30 minutes)\n")
    
    successful = 0
    failed = 0
    
    for obj_num in OBJECTS_TO_DOWNLOAD:
        # Format object name: 001, 002, etc.
        obj_name = f"{obj_num:03d}"
        
        # Download tar.gz from YCB repository
        url = f"{YCB_BASE_URL}{obj_name}*"
        tar_filename = f"ycb_{obj_name}.tar.gz"
        tar_filepath = os.path.join(YCB_OUTPUT_PATH, tar_filename)
        
        print(f"[{obj_num:2d}/20] Object {obj_name}:")
        
        # Download
        if download_file(url, tar_filepath, tar_filename):
            # Extract
            if extract_tar_gz(tar_filepath, YCB_OUTPUT_PATH):
                # Clean up tar file
                try:
                    os.remove(tar_filepath)
                    print(f"    Cleaned up tar file")
                except:
                    pass
                
                successful += 1
            else:
                failed += 1
        else:
            failed += 1
        
        print()
    
    print(f"[3/3] Summary:")
    print(f"      ✓ Successfully downloaded: {successful}/{len(OBJECTS_TO_DOWNLOAD)}")
    print(f"      ✗ Failed: {failed}/{len(OBJECTS_TO_DOWNLOAD)}")
    
    return successful > 0

def verify_download():
    """Verify downloaded files"""
    print(f"\nVerifying download...")
    
    if not os.path.exists(YCB_OUTPUT_PATH):
        print(f"✗ Directory not found: {YCB_OUTPUT_PATH}")
        return False
    
    objects = sorted([d for d in os.listdir(YCB_OUTPUT_PATH) 
                     if os.path.isdir(os.path.join(YCB_OUTPUT_PATH, d))])
    
    print(f"✓ Found {len(objects)} objects:")
    
    for obj in objects[:10]:
        obj_path = os.path.join(YCB_OUTPUT_PATH, obj)
        
        # Check for mesh files
        mesh_count = 0
        has_ply = False
        
        for root, dirs, files in os.walk(obj_path):
            for f in files:
                if f.endswith(('.ply', '.obj', '.dae', '.urdf')):
                    mesh_count += 1
                    if f.endswith('.ply'):
                        has_ply = True
        
        status = "✓" if has_ply else "⚠"
        print(f"  {status} {obj} ({mesh_count} mesh files)")
    
    if len(objects) > 10:
        print(f"  ... and {len(objects)-10} more")
    
    return len(objects) > 0

def main():
    try:
        success = download_ycb_objects()
        
        if success:
            if verify_download():
                print("\n" + "="*70)
                print("✓ YCB Dataset Ready!")
                print("="*70)
                print(f"\nLocation: {YCB_OUTPUT_PATH}")
                print("\nNext steps:")
                print("  1. python scripts/verify_ycb.py")
                print("  2. python scripts/test_ycb_env.py")
                print("  3. python scripts/train_ycb_grasp.py")
                return 0
        
        print("\n✗ Download failed. Try manual download instead.")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user")
        return 1
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
