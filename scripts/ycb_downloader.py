#Copyright 2015 Yale University - Grablab
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import sys
import json
import urllib.request
import urllib.error
from typing import List


# FIX #3: Use home directory expansion
output_directory = os.path.expanduser("datasets/ycb_data/ycb_models")

# You can either set this to "all" or a list of the objects that you'd like to download.
objects_to_download = ["002_master_chef_can", "003_cracker_box", "004_sugar_box","005_tomato_soup_can",
                       "006_mustard_bottle","007_tuna_fish_can","008_pudding_box","009_gelatin_box",
                       "010_potted_meat_can", "017_orange","019_pitcher_base","021_bleach_cleanser",
                       "024_bowl","026_sponge","032_knife","033_spatula","035_power_drill",
                       "036_wood_block","037_scissors","038_padlock", "076_timer", "077_rubiks_cube"]

files_to_download = ["google_16k"]

# Extract all files from the downloaded .tgz, and remove .tgz files.
# If false, will just download all .tgz files to output_directory
extract = True

base_url = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/"
objects_url = base_url + "objects.json"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

def fetch_objects(url):
    # FIX #2: Use urllib.request (Python 3)
    response = urllib.request.urlopen(url)
    html = response.read()
    objects = json.loads(html)
    return objects["objects"]

def download_file(url, filename):
    # FIX #2: Use urllib.request (Python 3)
    u = urllib.request.urlopen(url)
    f = open(filename, 'wb')
    meta = u.info()
    file_size = int(meta.get("Content-Length", 0))
    
    print("Downloading: %s (%s MB)" % (filename, file_size/1000000.0))
    
    file_size_dl = 0
    block_sz = 65536
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        
        file_size_dl += len(buffer)
        f.write(buffer)
        
        if file_size > 0:
            status = r"%10d  [%3.2f%%]" % (file_size_dl/1000000.0, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print(status, end='')
    
    print()  # New line
    f.close()

def tgz_url(object_name, file_type):
    # FIX #1: Add missing RETURN statement!
    if file_type in ["berkeley_rgbd", "berkeley_rgb_highres"]:
        return base_url + f"berkeley/{object_name}/{object_name}_{file_type}.tgz"
    elif file_type == "berkeley_processed":
        return base_url + f"berkeley/{object_name}/{object_name}_berkeley_meshes.tgz"
    else:  # google files
        return base_url + f"google/{object_name}_{file_type}.tgz"

def extract_tgz(filename, extract_dir):
    tar_command = "tar -xzf {filename} -C {dir}".format(filename=filename, dir=extract_dir)
    os.system(tar_command)
    os.remove(filename)

def check_url(url):
    try:
        # FIX #2: Use urllib.request (Python 3)
        request = urllib.request.Request(url)
        request.get_method = lambda: 'HEAD'
        response = urllib.request.urlopen(request, timeout=5)
        return True
    except Exception as e:
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("YCB Dataset Downloader")
    print("=" * 70)
    print(f"\nDownloading to: {output_directory}")
    print(f"Objects: {len(objects_to_download)}")
    print(f"File types: {files_to_download}\n")
    
    objects = objects_to_download
    
    successful = 0
    failed = 0
    
    for object_name in objects:
        if objects_to_download == "all" or object_name in objects_to_download:
            for file_type in files_to_download:
                url = tgz_url(object_name, file_type)
                
                if not check_url(url):
                    print(f"✗ {object_name} ({file_type}): URL not found")
                    failed += 1
                    continue
                
                filename = "{path}/{object}_{file_type}.tgz".format(
                    path=output_directory,
                    object=object_name,
                    file_type=file_type
                )
                
                try:
                    download_file(url, filename)
                    
                    if extract:
                        extract_tgz(filename, output_directory)
                        print(f"✓ {object_name}: Downloaded and extracted")
                    else:
                        print(f"✓ {object_name}: Downloaded")
                    
                    successful += 1
                    
                except Exception as e:
                    print(f"✗ {object_name}: {e}")
                    failed += 1
    
    print("\n" + "=" * 70)
    print(f"Download Summary")
    print("=" * 70)
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"\nDataset location: {output_directory}")
    print("=" * 70)