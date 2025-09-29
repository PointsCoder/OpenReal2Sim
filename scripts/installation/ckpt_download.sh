# mega-sam
mkdir -p third_party/mega-sam/Depth-Anything/checkpoints
wget -O third_party/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O third_party/mega-sam/cvd_opt/raft-things.pth