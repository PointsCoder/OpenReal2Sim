# MuJoCo Simulation

## Installation

The MuJoCo container can be built locally:

```bash
cd docker/mujoco
docker build -t mujoco:dev .
```

Or build via compose:

```bash
docker compose -f docker/compose.yml build mujoco
```

**Before launching the container**, enable X11 forwarding on the host:

```bash
xhost +local:
```

Launch and enter the container:

```bash
docker compose -f docker/compose.yml up -d mujoco
docker compose -f docker/compose.yml exec mujoco bash
```
