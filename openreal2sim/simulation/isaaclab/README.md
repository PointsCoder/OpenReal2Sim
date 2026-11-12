# IsaacLab

## Installation

**Pull the pre-built docker image**

Pull the pre-built image rather than building it; build only when you change a Dockerfile:
   ```bash
   docker compose -f docker/compose.yml pull simulation
   ```
   The packages are private until Esteban makes them public, so until then run
   `docker login ghcr.io` first with a GitHub token that has the `read:packages` scope.

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then, launch the container:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml up -d simulation 
```
and enter it:
```
HOST_UID=$(id -u) HOST_GID=$(id -g) docker compose -p "$USER" -f docker/compose.yml exec simulation bash
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
python openreal2sim/simulation/isaaclab/sim_trajectory_replay.py --demo_dir <path_to_demo_directory>
```