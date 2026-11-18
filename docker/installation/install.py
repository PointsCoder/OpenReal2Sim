import os
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download

def download_file(url, destination):
    """Downloads a file, creating the destination directory if needed."""
    dest_dir = os.path.dirname(destination)
    print(f"Ensuring directory exists: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    print(f"Downloading {file_name} to {destination}...")
    urlretrieve(url, destination)
    print("Download complete.")


def download_sam3d_checkpoints(destination_dir):
    """Downloads SAM 3D Objects checkpoints from public HuggingFace repo."""
    print(f"Downloading SAM 3D Objects checkpoints to {destination_dir}...")
    
    snapshot_download(
        repo_id="licesma/sam-3d-objects-weights",
        repo_type="model",
        local_dir=destination_dir,
    )
    
    print("SAM 3D Objects checkpoints downloaded successfully.")
    return True

def main():
    base_dir = "/app"
    os.chdir(base_dir)
    print(f"Working directory set to: {os.getcwd()}")

    # --- Mega-SAM Dependencies ---
    print("\n--- [1/6] Setting up Mega-SAM dependencies ---")
    download_file(
        "https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth",
        "third_party/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth"
    )
    download_file(
        "https://huggingface.co/datasets/licesma/raft_things/resolve/main/raft-things.pth",
        "third_party/mega-sam/cvd_opt/raft-things.pth"
    )

    # --- Segmentation Dependencies (Grounded-SAM-2) ---
    print("\n--- [2/6] Downloading Segmentation model ---")
    download_file(
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"
    )

    # --- FoundationPose Dependencies ---
    print("\n--- [3/6] Downloading FoundationPose weights ---")
    fp_weights_dir = os.path.join(base_dir, "third_party/FoundationPose/weights")
    os.makedirs(fp_weights_dir, exist_ok=True)
    snapshot_download(
        repo_id="licesma/foundationpose_weights",
        repo_type="dataset",
        local_dir=fp_weights_dir,
        cache_dir=os.path.join(base_dir, ".cache")
    )

    # --- WiLoR Dependencies (Hand Extraction) ---
    print("\n--- [4/6] Downloading WiLoR pretrained models ---")
    wilor_models_dir = os.path.join(base_dir, "third_party/WiLoR/pretrained_models")
    os.makedirs(wilor_models_dir, exist_ok=True)
    download_file(
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt",
        os.path.join(wilor_models_dir, "detector.pt")
    )
    download_file(
        "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt",
        os.path.join(wilor_models_dir, "wilor_final.ckpt")
    )

    #TODO: MANO params need to be downloaded after registering on certain website. This needs to be done manually.

    #--- Grasp Generation Checkpoints ---
    print("\n--- [5/6] Downloading GraspGen checkpoints ---")
    graspgen_models_dir = os.path.join(base_dir, "third_party/GraspGen/GraspGenModels")
    os.makedirs(graspgen_models_dir, exist_ok=True)
    snapshot_download(
        repo_id="adithyamurali/GraspGenModels",
        local_dir=graspgen_models_dir,
        cache_dir=os.path.join(base_dir, ".cache")
    )
    
    # --- SAM 3D Objects Checkpoints ---
    print("\n--- [6/6] Downloading SAM 3D Objects checkpoints ---")
    sam3d_ckpt_dir = os.path.join(base_dir, "third_party/sam-3d-objects/checkpoints")
    download_sam3d_checkpoints(sam3d_ckpt_dir)

    print("\n\n--- All dependencies set up successfully! ---")

if __name__ == "__main__":
    main()