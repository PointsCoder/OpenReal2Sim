# IsaacLab

## Installation

**Pull the pre-built docker image**

If you want to use a pre-built image from Docker Hub, you can pull it directly:
   ```bash
   docker pull ghcr.io/pointscoder/isaaclab:dev
   docker tag ghcr.io/pointscoder/isaaclab:dev isaaclab:dev
   ```

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then, launch the container:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml up -d isaaclab 
```
and enter it:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml exec isaaclab bash
```

## Preprocessing

We need to convert the textured meshes to USD format. This is done by running the following script:
```
python openreal2sim/simulation/isaaclab/sim_preprocess/usd_conversion.py
```

Then, we generate grasp proposals for each object in the scene:
```
python openreal2sim/simulation/isaaclab/sim_preprocess/grasp_generation.py
```

## Running the simulation

We provide heuristic policies using grasping and motion planning:
```
python openreal2sim/simulation/isaaclab/sim_heuristic_manip.py
```

## Replay Robot Trajectories

We can also replay the recorded robot trajectories in IsaacSim:
```
python openreal2sim/simulation/isaaclab/sim_replay_trajectories.py --demo_dir <path_to_demo_directory>
```