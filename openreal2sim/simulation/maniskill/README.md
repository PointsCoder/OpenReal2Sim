# ManiSkill Integration for OpenReal2Sim

This module provides a ManiSkill environment that loads reconstructed scenes from the OpenReal2Sim pipeline, enabling robotic manipulation simulation on real-world reconstructed scenes.

## Installation with Docker

The recommended way to set up the environment is by using the provided Dockerfile.

### Step 1: Build the Custom Docker Image

From the root of the OpenReal2Sim repository, run the following command to build the Docker image. The build context is set to the `docker` subdirectory where the Dockerfile and its resources are located.

```bash
docker build -t openreal2sim-maniskill openreal2sim/simulation/maniskill/docker
```

### Step 2: Run the Docker Container

Once the image is built, run the container to get an interactive shell. This will mount the current directory into the container at `/app`.

```bash
docker run --gpus all -it --rm -v $(pwd):/app -w /app openreal2sim-maniskill
```

You are now inside the container. All subsequent commands should be run from this shell.

## Optional: Verify Base ManiSkill Docker

If you encounter issues running the custom container, you can verify that your system is compatible with the base ManiSkill environment. This ensures that your Docker, GPU drivers, and NVIDIA Container Toolkit are set up correctly.

**1. Pull the base image:**

```bash
docker pull maniskill/base
```

**2. Run a simple demo:**
This command runs a demo with random actions. A viewer window should appear.

```bash
docker run --rm -it --gpus all --pid host maniskill/base python -m mani_skill.examples.demo_random_action
```

**3. Run a GPU simulation benchmark:**
This command tests the GPU simulation performance.

```bash
docker run --rm -it --gpus all --pid host maniskill/base python -m mani_skill.examples.benchmarking.gpu_sim
```

## Running Scripts

Here are some examples of how to run the provided scripts to test the environment.

### 1. Test the Environment

The `visualize_env.py` script loads a scene for inspection.

**Run with a demo scene:**

```bash
python openreal2sim/simulation/maniskill/scripts/visualize_env.py --scene outputs/demo_image/scene/json/scene.json --mode static
```

- Use `--mode static` to inspect the scene.
- Use `--mode random` to watch the robot perform random actions.

### 2. Visualize Grasp Poses

The `visualize_grasp_pose.py` script loads pre-computed grasps for an object and displays the best one.

**Run with a demo scene:**

```bash
python openreal2sim/simulation/maniskill/scripts/visualize_grasp_pose.py --scene outputs/demo_video/scene/json/scene.json
```

### 3. Run Motion Planning

The `run_motion_planning.py` script demonstrates a pick-and-place sequence.

**Run with visualization:**

```bash
python openreal2sim/simulation/maniskill/scripts/run_motion_planning.py --scene outputs/demo_video/scene/json/scene.json --vis
```

- The `--vis` flag is required to see the robot perform the task.
