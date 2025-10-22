# IsaacLab

## Installation

**On the host machine and before entering the container**, run
```
xhost +local:
```
This is to allow the container to access the host's X server for IsaacSim GUI.

Then, launch the container:
```
docker compose -p "$USER" -f docker/compose.yml up -d isaaclab 
```
and enter it:
```
docker compose -p "$USER" -f docker/compose.yml exec isaaclab bash
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