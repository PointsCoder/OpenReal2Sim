# mega-sam
mkdir -p third_party/mega-sam/Depth-Anything/checkpoints
wget -O third_party/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
gdown --id 1MqDajR89k-xLV0HIrmJ0k-n8ZpG6_suM -O third_party/mega-sam/cvd_opt/raft-things.pth

# segmentation
wget -O third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# foundation pose
mkdir -p third_party/FoundationPose/weights
gdown --folder 1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O third_party/FoundationPose/weights