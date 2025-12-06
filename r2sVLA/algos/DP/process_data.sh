#!/bin/bash

task_name=${1}
task_config=${2}
expert_data_num=${3}
use_franka_format=${4:-false}  # Default to false (ViperX format)
data_dir=${5}  # Optional custom data directory

# Build command
cmd="python process_data.py $task_name $task_config $expert_data_num"

if [ "$use_franka_format" = "true" ] || [ "$use_franka_format" = "True" ]; then
    cmd="$cmd --use_franka_format"
fi

if [ -n "$data_dir" ]; then
    cmd="$cmd --data_dir $data_dir"
fi

echo "Running: $cmd"
eval $cmd