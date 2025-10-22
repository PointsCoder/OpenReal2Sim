# mega-sam
mkdir -p third_party/mega-sam/Depth-Anything/checkpoints
wget -O third_party/mega-sam/Depth-Anything/checkpoints/depth_anything_vitl14.pth \
  https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth
wget -O third_party/mega-sam/cvd_opt/raft-things.pth \
  https://huggingface.co/datasets/licesma/raft_things/resolve/main/raft-things.pth

# segmentation
wget -O third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt


# wilor
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P /app/third_party/WiLoR/pretrained_models/
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt -P /app/third_party/WiLoR/pretrained_models/

# segmentation cuda extension
cd /app/third_party/Grounded-SAM-2 && \
  python setup.py build_ext --inplace -v && \
  cd /app

# foundation pose
mkdir -p third_party/FoundationPose/weights
hf download licesma/foundationpose_weights --repo-type dataset --local-dir third_party/FoundationPose/weights

# foundation pose compile
cd /app/third_party/FoundationPose/mycpp && \
  rm -rf build && \
  mkdir -p build && cd build && \
  cmake .. -DPYTHON_EXECUTABLE=$(which python) && make -j11 && \
  cd /app

cd /app/third_party/FoundationPose/bundlesdf/mycuda && \
  rm -rf build *egg* && \
  python -m pip install . --no-build-isolation && \
  cd /app 