#!/bin/bash
# Start background tmux session

chmod +x /data/openreal2sim/junsoo/OpenReal2Sim/update_gpu.sh

tmux new-session -d -s motion_docker '/data/openreal2sim/junsoo/OpenReal2Sim/update_gpu.sh'
tmux split-window -h -t motion_docker 'docker exec -u $(id -u junsookim):$(id -g junsookim) -it junsookim-openreal2sim-run-5100a6901617 bash -c "cd /app && ./openreal2sim/motion/motion_agent.sh --start 1"'
