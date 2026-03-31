#!/bin/bash
# Run on the host to update the freest GPU file periodically
while true; do
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t ',' -k2 -nr | head -n 1 | awk -F', ' '{print $1}' > /data/openreal2sim/junsoo/OpenReal2Sim/freest_gpu.txt
  sleep 5
done
