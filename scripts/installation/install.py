import os
import subprocess
import sys
from urllib.request import urlretrieve
from huggingface_hub import snapshot_download

def run_command(command, cwd=None):
    """Runs a command in a subprocess and checks for errors."""
    print(f"Running command: {' '.join(command)} in {cwd or os.getcwd()}")
    process = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)
    # Print stdout/stderr only if there's an error, or for verbose logging if desired
    if process.returncode != 0:
        print(f"ERROR: Command failed with exit code {process.returncode}")
        print("--- STDOUT ---")
        print(process.stdout)
        print("--- STDERR ---")
        print(process.stderr)
        sys.exit(1) # Exit script on failure

def download_file(url, destination):
    """Downloads a file, creating the destination directory if needed."""
    dest_dir = os.path.dirname(destination)
    print(f"Ensuring directory exists: {dest_dir}")
    os.makedirs(dest_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    print(f"Downloading {file_name} to {destination}...")
    urlretrieve(url, destination)
    print("Download complete.")

def main():
    """Main function to set up all dependencies."""
    # Ensure we are in the correct base directory
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

    print("\n--- [3/6] Building Segmentation CUDA extension ---")
    build_cuda_path = os.path.join(base_dir, "third_party/Grounded-SAM-2")
    run_command(
        [sys.executable, "build_cuda.py", "build_ext", "--inplace", "-v"],
        cwd=build_cuda_path
    )

    # --- FoundationPose Dependencies ---
    print("\n--- [4/6] Downloading FoundationPose weights ---")
    fp_weights_dir = os.path.join(base_dir, "third_party/FoundationPose/weights")
    os.makedirs(fp_weights_dir, exist_ok=True)
    snapshot_download(
        repo_id="licesma/foundationpose_weights",
        repo_type="dataset",
        local_dir=fp_weights_dir
    )

    print("\n--- [5/6] Compiling FoundationPose C++ extension ---")
    fp_cpp_build_path = os.path.join(base_dir, "third_party/FoundationPose/mycpp/build")
    run_command(["rm", "-rf", fp_cpp_build_path])
    os.makedirs(fp_cpp_build_path, exist_ok=True)
    run_command(
        ["cmake", "..", f"-DPYTHON_EXECUTABLE={sys.executable}"],
        cwd=fp_cpp_build_path
    )
    run_command(["make", "-j11"], cwd=fp_cpp_build_path)

    print("\n--- [6/6] Compiling FoundationPose bundlesdf CUDA extension ---")
    fp_bundlesdf_path = os.path.join(base_dir, "third_party/FoundationPose/bundlesdf/mycuda")
    run_command(["rm", "-rf", "build"], cwd=fp_bundlesdf_path) 
    run_command(["rm", "-rf", "*.egg-info"], cwd=fp_bundlesdf_path) 
    run_command(
        [sys.executable, "-m", "pip", "install", ".", "--no-build-isolation"],
        cwd=fp_bundlesdf_path
    )

    print("\n\n--- All dependencies set up successfully! ---")

if __name__ == "__main__":
    main()
