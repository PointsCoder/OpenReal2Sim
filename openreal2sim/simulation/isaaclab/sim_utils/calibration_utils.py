import numpy as np
import transforms3d

def _as_homo(T):
    A = np.asarray(T, dtype=np.float64)
    if A.shape == (4, 4):
        return A
    if A.shape == (3, 4):
        M = np.eye(4, dtype=np.float64); M[:3, :4] = A; return M
    if A.shape == (3, 3):
        M = np.eye(4, dtype=np.float64); M[:3, :3] = A; return M
    raise ValueError(f"Unsupported shape: {A.shape}")

def _blk3(R):
    M = np.eye(4, dtype=np.float64); M[:3, :3] = R; return M

def _quat_wxyz(R):
    q = transforms3d.quaternions.mat2quat(R)  # (w,x,y,z)
    return -q if q[0] < 0 else q

def load_extrinsics(config: dict, key: str) -> np.ndarray:
    """
    Load the extrinsic parameters from the scene JSON.
    """

    E_list = config.get("extrinsics", None)
    if E_list is None:
        print(f"Warning: No calibrated extrinsics found for {key}. Use random robot poses.")
        return None
    E = np.array(E_list).reshape(4, 4)
    print(f"Loaded extrinsics for {key}:")
    print(E)
    return E


def calibration_to_robot_pose(scene_js: dict, T_camopencv_to_robotbase_m) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the real-world calibrated camera to robot transform into simulation
    Inputs:
        scene_js: The scene JSON containing camera extrinsics
        T_camopencv_to_robotbase_m: The transform matrix from camera_optical (in opencv camera format) to robot_base
    Outputs:
        pos_w: The position of the robot base in world coordinates
        quat_w: The orientation of the robot base in world coordinates (as a quaternion)
        meta: A dictionary containing additional information about the calibration
    """

    T_camopencv_to_world = _as_homo(scene_js["camera"]["camera_opencv_to_world"])

    T_camopencv_to_robotbase = _as_homo(T_camopencv_to_robotbase_m)

    T_robotbase_to_world = T_camopencv_to_world @ np.linalg.inv(T_camopencv_to_robotbase)

    pos_w  = T_robotbase_to_world[:3, 3].copy()
    quat_w = _quat_wxyz(T_robotbase_to_world[:3, :3])

    return pos_w, quat_w

def calibration_to_robot_pose_deprecated(scene_js: dict, T_camisaac_to_robotbase_m) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the real-world calibrated camera to robot transform into simulation
    Inputs:
        scene_js: The scene JSON containing camera extrinsics
        T_camisaac_to_robotbase_m: The transform matrix from camera_isaac to robot_base
    Outputs:
        pos_w: The position of the robot base in world coordinates
        quat_w: The orientation of the robot base in world coordinates (as a quaternion)
        meta: A dictionary containing additional information about the calibration
    """

    T_camopencv_to_world = _as_homo(scene_js["camera"]["camera_opencv_to_world"])

    T_camisaac_to_robotbase = _as_homo(T_camisaac_to_robotbase_m)

    # This may be the true transformation
    # REP-103: camera_opencv(OpenCV: +x right, +y down, +z forward) -> camera_isaac(IsaacSim: +x forward, +y left, +z up)
    T_camopencv_to_camisaac = np.array([[ 0,  0,  1],
                                        [-1,  0,  0],
                                        [ 0, -1,  0]], dtype=np.float64)

    T_camopencv_to_robotbase = T_camisaac_to_robotbase @ _blk3(T_camopencv_to_camisaac)   # base -> camera_link

    T_robotbase_to_world = T_camopencv_to_world @ np.linalg.inv(T_camopencv_to_robotbase)

    pos_w  = T_robotbase_to_world[:3, 3].copy()
    quat_w = _quat_wxyz(T_robotbase_to_world[:3, :3])

    return pos_w, quat_w
