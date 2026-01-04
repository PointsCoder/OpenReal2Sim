# OpenReal2Sim to LeRobot 数据转换指南

本指南介绍如何将 OpenReal2Sim 的仿真数据转换为 LeRobot 标准数据集格式。

## 数据格式映射

### OpenReal2Sim HDF5 数据结构
OpenReal2Sim 的 HDF5 文件包含以下数据：

```
episode.hdf5/
├── observation/
│   └── head_camera/
│       ├── rgb/           # (T, H, W, 3) - RGB 图像序列
│       ├── intrinsics/    # 相机内参
│       └── extrinsics/    # 相机外参
├── joint_action/
│   ├── joint_pos/         # (T, 7) - 关节位置
│   └── joint_vel/         # (T, 7) - 关节速度
├── endpose/
│   └── gripper_pos/       # (T, 1) - 夹爪位置
├── ee_pose/
│   └── ee_pose_l/         # (T, 7) - 本地坐标系下的末端执行器位姿
├── action/
│   └── actions/           # (T, action_dim) - 动作数据（可选）
└── meta/                  # 元数据
    ├── task_desc/         # 任务描述
    └── frame_count/       # 帧数
```

### LeRobot 数据格式映射

| OpenReal2Sim 字段 | LeRobot 字段 | 说明 |
|------------------|-------------|------|
| `rgb` | `observation.images.camera` | RGB 图像序列 |
| `ee_pose_cam` | `observation.state` | 状态向量 (7维: position + quaternion in camera frame) |
| `actions` 或 EEF 位姿差分 | `action` | 动作向量 (7维: delta position + delta quaternion) |
| `task_desc` | `task` | 任务描述字符串 |

## 安装依赖

确保安装了 LeRobot 和相关依赖：

```bash
# 安装 LeRobot
pip install lerobot

# 安装其他依赖
pip install h5py numpy torch torchvision
```

## 使用方法

### 1. 转换单个 HDF5 文件

```bash
python openreal2sim/simulation/openreal2sim_to_lerobot.py \
    --hdf5_path /path/to/episode.hdf5 \
    --output_repo my-robot-dataset \
    --episode_idx 0
```

### 2. 批量转换目录中的所有 HDF5 文件

```bash
python openreal2sim/simulation/openreal2sim_to_lerobot.py \
    --hdf5_dir /path/to/hdf5_directory \
    --output_repo my-robot-dataset
```

## 输出结果

转换脚本会生成 LeRobot 格式的数据集：

```
my-robot-dataset/
├── meta.json                    # 数据集元数据和统计信息
├── data/
│   ├── shard-000000.arrow      # 数据分片
│   └── ...
└── videos/
    ├── 000000.mp4              # 压缩的视频数据
    └── ...
```

## 数据集使用

转换完成后，可以使用标准的 LeRobot API 加载数据集：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset(
    repo_id="my-robot-dataset",
    root="/path/to/dataset"
)

# 访问数据
sample = dataset[0]
print(sample.keys())
# dict_keys(['observation.images.camera', 'observation.state', 'action', 'task', ...])
```

## 配置选项

### 动作类型选择

脚本支持两种动作表示方式：

1. **关节位置控制** (默认):
   ```python
   # 使用关节位置差分作为动作
   actions[t] = joint_pos[t+1] - joint_pos[t]
   ```

2. **EEF 位姿控制** (如果 HDF5 中包含 `actions` 字段):
   ```python
   # 直接使用 HDF5 中的 actions 字段
   actions = episode_data['actions']
   ```

### 状态向量组成

状态向量直接使用相机坐标系下的末端执行器位姿：
```python
state = ee_pose_cam  # 7维 - [x, y, z, qx, qy, qz, qw] 在相机坐标系下
```

## 故障排除

### 常见问题

1. **导入错误**: 确保已安装 LeRobot
   ```bash
   pip install lerobot
   ```

2. **HDF5 文件格式错误**: 检查 HDF5 文件是否包含所需的字段
   ```python
   import h5py
   with h5py.File('episode.hdf5', 'r') as f:
       print(list(f.keys()))  # 检查根级字段
   ```

3. **内存不足**: 对于大型数据集，考虑分批处理
   ```python
   # 在脚本中调整批量大小
   # 当前版本处理单集数据，可扩展为分批处理
   ```

4. **视频编码错误**: 确保安装了 FFmpeg
   ```bash
   # Ubuntu/Debian
   sudo apt install ffmpeg

   # macOS
   brew install ffmpeg
   ```

## 扩展功能

### 添加多视角相机

如果 HDF5 文件包含多个相机视角，可以扩展脚本来支持：

```python
features = {
    "observation.images.wrist": {...},
    "observation.images.front": {...},
    "observation.images.side": {...},
    # ...
}
```

### 添加时序信息

对于需要历史帧的模型，可以配置 `delta_timestamps`：

```python
delta_timestamps = {
    "observation.images.camera": [-0.1, 0.0],  # 当前帧和前一帧
    "observation.state": [0.0],                 # 当前状态
    "action": [0.0]                            # 当前动作
}
```

## 示例工作流

1. **运行仿真**:
   ```bash
   python openreal2sim/simulation/isaaclab/demo/sim_heuristic_manip.py
   ```

2. **导出 HDF5**:
   ```bash
   # 仿真脚本会自动调用 export_batch_data_to_hdf5()
   ```

3. **转换格式**:
   ```bash
   python openreal2sim/simulation/openreal2sim_to_lerobot.py \
       --hdf5_dir outputs/dataset/hdf5 \
       --output_repo lerobot_dataset
   ```

4. **训练模型**:
   ```python
   # 使用 we_learn 或其他 LeRobot 兼容的训练框架
   ```

## 技术细节

- **图像压缩**: 使用 JPEG 压缩减少存储空间
- **数据类型**: 自动转换为 float32 以确保兼容性
- **时间同步**: 假设所有字段都已正确时间同步
- **坐标系**: 保持原始坐标系定义

## 贡献

如果发现转换问题或需要新功能，请提交 issue 或 PR。
