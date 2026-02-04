
import glob
import os
import sys

# Try to import h5py
try:
    import h5py
except ImportError:
    print("Error: h5py module not found. Please run this script in an environment where h5py is installed.")
    print("If you are using Isaac Sim, try running with the kit python or ensure the environment is activated.")
    sys.exit(1)

def verify_hdf5(h5_dir, num_files=10):
    print(f"Checking HDF5 files in {h5_dir}")
    
    # Get all hdf5 files
    files = glob.glob(os.path.join(h5_dir, "*.hdf5"))
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    
    if not files:
        print("No HDF5 files found!")
        return

    latest_files = files[:num_files]
    print(f"Found {len(files)} files, checking latest {len(latest_files)}...")

    for i, filepath in enumerate(latest_files):
        print(f"\n[{i+1}/{len(latest_files)}] Checking: {os.path.basename(filepath)}")
        try:
            with h5py.File(filepath, 'r') as f:
                # 1. Check Meta
                if 'meta' in f:
                    print("  [OK] meta group found")
                    # Check frame count matches data
                    if 'frame_count' in f['meta'].attrs:
                        fc = f['meta'].attrs['frame_count']
                        print(f"       frame_count: {fc}")
                else:
                    print("  [FAIL] meta group MISSING")

                # 2. Check Observation/Camera/RGB
                # Note: code uses 'head_camera' or 'camera' depending on params
                cam_grp = None
                if 'observation/camera' in f:
                    cam_grp = f['observation/camera']
                elif 'observation/head_camera' in f:
                    cam_grp = f['observation/head_camera']
                
                if cam_grp:
                    print(f"  [OK] Camera group found: {cam_grp.name}")
                    if 'rgb' in cam_grp:
                        rgb = cam_grp['rgb']
                        print(f"       RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                        # Check compression if relevant (but here we just check access)
                    else:
                        print("       [FAIL] RGB dataset MISSING in camera group")
                else:
                    print("  [FAIL] observation/camera group MISSING")

                # 3. Check composed_rgb (if relevant, though usually it's just 'rgb' in the hdf5)
                # The implementation writes 'rgb' to the camera group.
                
                # 4. Check other keys
                expected_keys = ['action', 'joint_action', 'ee_pose']
                found_keys = []
                for k in expected_keys:
                    if k in f:
                        found_keys.append(k)
                print(f"  Found standard groups: {found_keys}")
                
        except Exception as e:
            print(f"  [ERROR] Failed to read file: {e}")

if __name__ == "__main__":
    # Default path based on user request
    target_dir = '/data/utkarsh/OpenReal2Sim/h5py/video_00467'
    verify_hdf5(target_dir)
