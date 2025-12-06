#!/bin/bash
# Training script for DP on demo_video dataset (Franka, 8D)

task_name=${1:-demo_video}
task_config=${2:-config}
expert_data_num=${3:-15}
seed=${4:-1}
action_dim=${5:-8}  # Franka: 8D (7 joints + 1 gripper)
gpu_id=${6:-0}

head_camera_type=D435

DEBUG=False
save_ckpt=True

alg_name=robot_dp_$action_dim
config_name=${alg_name}
addition_info=train
exp_name=${task_name}-robot_dp-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33maction_dim: ${action_dim} (Franka: 8D)\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Data directory for Franka format
DATA_DIR="/home/peiqiduan/OpenReal2Sim/h5py/demo_video"

# Process data if zarr doesn't exist
if [ ! -d "./data/${task_name}-${task_config}-${expert_data_num}.zarr" ]; then
    echo "Processing data from ${DATA_DIR}..."
    bash process_data.sh ${task_name} ${task_config} ${expert_data_num} true ${DATA_DIR}
fi

python train.py --config-name=${config_name}.yaml \
                            task.name=${task_name} \
                            task.dataset.zarr_path="data/${task_name}-${task_config}-${expert_data_num}.zarr" \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            setting=${task_config} \
                            expert_data_num=${expert_data_num} \
                            head_camera_type=$head_camera_type

