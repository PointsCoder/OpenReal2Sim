
from enum import Enum, auto
from typing import List, Dict, Optional
from dataclasses import dataclass, field

class SuccessMetricType(Enum):
    SIMPLE_LIFT = auto()
    TARGET_POINT = auto()
    TARGET_PLANE = auto()

class TaskType(Enum):
    SIMPLE_PICK = auto()
    SIMPLE_PICK_PLACE = auto()
    TARGETTED_PICK_PLACE = auto()

class RobotType(Enum):
    FRANKA = auto()
    UR5 = auto()


@dataclass
class BackgroundCfg:
    background_rgb_path: str
    background_mesh_path: str
    background_usd_path: str
    background_point: List[float]



@dataclass
class ObjectCfg:
    object_id: int
    object_name: str
    mesh_path: str
    usd_path: str


@dataclass
class SuccessMetric:
    success_metric_type: SuccessMetricType
    final_gripper_close:bool
    lift_height: Optional[float] = None
    ground_value: Optional[float] = None
    end_pose: Optional[List[float]] = None
 

@dataclass
class TrajectoryCfg:  
    robot_pose: List[float] # quat wxyz
    object_poses: Dict[int, List[float]] # quat
    object_trajectory: List[List[float]] # quat
    final_gripper_close: bool
    success_metric: SuccessMetric
    pregrasp_pose: Optional[List[float]] = None  ### This is eef in world frame.   quat
    grasp_pose: Optional[List[float]] = None  ### This is eef in world frame. quat
    robot_type: Optional[RobotType] = None
    init_manip_object_com: Optional[List[float]] = None # com in world frame



@dataclass
class CameraInfo:
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float
    camera_opencv_to_world: List[List[float]]
    camera_position: List[float]
    camera_heading_wxyz: List[float]

@dataclass
class TaskCfg:
    task_key: str
    task_id: int
    task_desc: List[str]
    task_type: TaskType
    background_cfg: BackgroundCfg
    camera_info: CameraInfo
    manipulated_oid: int  
    start_related: List[int]
    end_related: List[int]
    objects: List[ObjectCfg]
    reference_trajectory: List[TrajectoryCfg]
    generated_trajectories: List[TrajectoryCfg] = field(default_factory=list)








