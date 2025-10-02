import os
import subprocess
"""
Downloads pretrained checkpoints for Mega-SaM.
More info: https://github.com/mega-sam/mega-sam?tab=readme-ov-file#downloading-pretrained-checkpoints
"""

def setup_checkpoints():

    current_path = os.path.dirname(os.path.abspath(__file__))
    repo_path = os.path.abspath(os.path.join(current_path, "..", "..",".."))

    # Paths
    depth_dir = os.path.join(repo_path, "third_party", "mega-sam", "Depth-Anything", "checkpoints")
    raft_dir = os.path.join(repo_path, "third_party", "mega-sam", "cvd_opt")

    os.makedirs(depth_dir, exist_ok=True)

    # Depth Anything
    depth_url = "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth"
    subprocess.run(["wget", "-nc", "-P", depth_dir, depth_url], check=True)

    # Raft Optimization
    raft_url = "https://huggingface.co/datasets/licesma/raft_things/resolve/main/raft-things.pth"
    subprocess.run(["wget", "-nc", "-P", raft_dir, raft_url], check=True)

if __name__ == "__main__":
    setup_checkpoints()