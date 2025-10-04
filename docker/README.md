# Docker Instructions

Docker provides a sharable environment to run the code in this repository.

The name inside the brackets indicates which container is needed to run which script. These scripts are meant to be run in order.


## How to Run the Docker Container

Here are the steps to build and run the Docker container for this repository.

### Access to Docker Commands

Make sure you have Docker installed and running on your machine. 

You can check this by running:
   ```bash
   docker --version
   ```

Make sure you have been added to the `docker` group to run docker commands without `sudo`.

if you encountered permission issues when running `docker_build` or `docker_run`, you can ask your administrator to try:
   ```bash
   sudo usermod -aG docker $USER
   ```
to add you to the `docker` group. You may need to log out and log back in for this to take effect.

if sometimes the group change does not take effect, you can also try:
   ```bash
   newgrp docker
   ```
to switch to the `docker` group in the current terminal session.

### Use Pre-Built Docker Image

If you want to use a pre-built image from Docker Hub, you can pull it directly:
   ```bash
   docker pull ghcr.io/pointscoder/openreal2sim:dev
   docker tag ghcr.io/pointscoder/openreal2sim:dev openreal2sim:dev
   ```

### (Optional) Build the Docker Image

You can also build the docker image yourself. Once the container is running, you can execute any of the scripts above inside it.

**Build the container image**
   From the repository root:
   ```bash
   docker compose -f docker/compose.yml build openreal2sim
   ```

Optionally, you may want to push the image to a remote registry (e.g., GitHub Container Registry) for easier sharing.

If so, you can tag the image and push it:
   ```bash
   docker tag openreal2sim:dev ghcr.io/<username>/openreal2sim:dev
   docker push ghcr.io/<username>/openreal2sim:dev
   ```
Don't forget to change the visibility of the pushed image to public if you want others to access it.

### After Getting the Docker Image

Every time you want to run a script inside the repo, follow these steps:

**Run the container**
   ```bash
   docker compose -p "$USER" -f docker/compose.yml run openreal2sim
   ```

**Execute a script inside the container**
   
   Inside the container’s terminal and from the repository root:
   ```bash
   python <path_to_script>.py
   ```
