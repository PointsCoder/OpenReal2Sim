#!/usr/bin/env python3
"""
Converts videos recorded with legacy codecs (like MPEG-4 Part 2 or HEVC)
    into standard H.264 for better playback support.

For each video in the specified index range:
  1. Re-encodes it into H.264 video + AAC audio (web-safe, browser-compatible).
  2. Forces 8-bit YUV 4:2:0 pixel format (yuv420p), ensuring compatibility with
     Chromium-based players (e.g. VS Code, Chrome).
  3. Applies `-movflags +faststart`, which moves the MP4 header to the beginning
     of the file so it can start playing before the full file loads.
"""
import subprocess
import os
# === Configuration ===
data_path = "data"
label = "basic_pick_place_" 
first_index = 10
last_index = 29

# === Processing Loop ===
for i in range(first_index, last_index + 1):
    input_path = os.path.join(data_path, f"{label}{i}.mp4")
    tmp_output = os.path.join(data_path, f"{label}{i}_tmp.mp4")

    print(f"Re-encoding {input_path}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "160k",
        "-movflags", "+faststart",
        tmp_output
    ]
    subprocess.run(cmd, check=True)
    os.replace(tmp_output, input_path)