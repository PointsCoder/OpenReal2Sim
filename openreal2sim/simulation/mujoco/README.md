# MuJoCo Simulation

Tools for converting reconstructed 3D assets to MuJoCo simulation format.

## Installation

### Docker Installation

Build the MuJoCo container locally:

```bash
docker build -t mujoco:dev -f docker/mujoco/Dockerfile .
```

Or using docker compose:

```bash
docker compose -f docker/compose.yml build mujoco
```

Before launching, enable X11 forwarding on the host:

```bash
xhost +local:
```

Launch the container:

```bash
docker compose -f docker/compose.yml up -d mujoco
docker compose -f docker/compose.yml exec mujoco bash
```
### Virtual Environment Installation

Alternatively, install dependencies in a virtual environment:
```bash
pip install -r docker/mujoco/requirements.docker.txt
```

## Workflow

### (Optional) Simplify Meshes

```bash
python openreal2sim/simulation/mujoco/tools/simplify_scene.py --scene-name demo_genvideo
```
**Warning**: This overwrites GLB files **in place**.

### GLB to MJCF Conversion

```bash
python openreal2sim/simulation/mujoco/tools/glb_to_mjcf.py --scene-name demo_genvideo
```

Converts GLB meshes to MJCF using CoACD for convex decomposition. Outputs to `outputs/<scene_name>/simulation/mujoco/mjcf/`.

### Scene Fusion

```bash
python openreal2sim/simulation/mujoco/tools/fuse_scene.py --scene-name demo_genvideo
```
Fuses MJCF assets with Franka Panda robot. Output: `outputs/<scene>/simulation/mujoco/scene.xml`

Options:
- `--object-mass`: Object masses (e.g., `object_1=1.0 object_2=2.0`)
- `--default-mass`: Default mass (default: 0.2 kg)

Simulation parameters are in `config/constants.yaml`.

### Visualization

```bash
python -m mujoco.viewer --mjcf outputs/demo_genvideo/simulation/mujoco/scene.xml
```

### Trajectory Replay

```bash
python openreal2sim/simulation/mujoco/tools/replay_trajectory.py \
  --demo-path outputs/demo_genvideo/demos/demo_0/env_000
```
Replays trajectories with PD control. Keyboard: Space (pause), R (restart), Q (quit).

You can override object masses with `--object-mass object_1=1.0 object_2=2.0`.

**Note:** Trajectories from Isaac Sim may not always succeed in MuJoCo due to dynamics differences, especially contact handling. You may need to tune the simulation parameters or modify the GLB to MJCF settings.

## License

The GLB to MJCF conversion code is adapted from [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) (MIT License).

