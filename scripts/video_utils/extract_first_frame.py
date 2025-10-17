#!/usr/bin/env python3
"""
Extracts and optionally resizes the first frame of a series of MP4 videos.

For each video:
  • Extracts the first frame (`-frames:v 1`).
  • Optionally resizes it (if `resize` is True) to the specified resolution.

Common resolutions:
  - 1080p → 1920 x 1080
  - 720p  → 1280 x 720
  - 480p  → 854 x 480
"""

import os, subprocess

# === Configuration ===
data_path = "data"
label = "basic_pick_place_"  
first_index = 10
last_index = 29

resize = True         # Set to False to keep original resolution
width, height = 1280, 720  

# === Processing Loop ===
for i in range(first_index, last_index + 1):
    input_path = os.path.join(data_path, f"{label}{i}.mp4")
    output_path = os.path.join(data_path, f"{label}{i}.jpg")

    print(f"Extracting first frame from {input_path} → {output_path}")

    cmd = [
    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
    "-i", input_path,
    "-frames:v", "1",
    ]
    if resize:
        cmd += ["-vf", f"scale={width}:{height}"]
    cmd += [output_path]

    subprocess.run(cmd, check=True)
    print(f"Saved {output_path}")