#!/usr/bin/env python3
"""Pull the images, prepare the host directories and run a pipeline stage."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMPOSE = ROOT / "docker" / "compose.yml"
SERVICES = ("reconstruction", "simulation")
MARKER = ROOT / "outputs" / ".weights_installed"
INSTALL = "scripts/installation/install.py"

# Directories the compose mounts need: Docker creates missing ones as root.
REPO_DIRS = ("outputs", "h5py", "tasks", ".cache")
CACHE_DIRS = ("torch", "huggingface", "hy3dgen", "warp", "matplotlib")

OVERRIDE = """services:
  {service}:
    deploy:
      resources:
        reservations:
          devices: !override
            - driver: nvidia
              device_ids: [{ids}]
              capabilities: [gpu]
"""


def fail(message):
    print("error: " + message, file=sys.stderr)
    sys.exit(1)


def docker_gid():
    """GID of the docker group, so the container can reach the socket."""
    try:
        line = subprocess.run(
            ["getent", "group", "docker"], stdout=subprocess.PIPE, text=True
        ).stdout
        return line.split(":")[2]
    except (IndexError, OSError):
        return "999"


def compose_env():
    env = dict(os.environ)
    env["HOST_UID"] = str(os.getuid())
    env["HOST_GID"] = str(os.getgid())
    env["DOCKER_GID"] = docker_gid()
    env["PWD"] = str(ROOT)
    # The simulation service mounts $XAUTHORITY; point it somewhere that exists.
    if not env.get("XAUTHORITY") or not Path(env["XAUTHORITY"]).exists():
        env["XAUTHORITY"] = "/dev/null"
    return env


def gpu_override(service, gpus, keep):
    """Write a compose override pinning the service to the given GPUs."""
    ids = ", ".join('"%s"' % g.strip() for g in gpus.split(",") if g.strip())
    if not ids:
        fail("--gpus takes a comma separated list of indices, for example 0,1")
    handle, path = tempfile.mkstemp(prefix="openreal2sim-gpus-", suffix=".yml")
    with os.fdopen(handle, "w") as stream:
        stream.write(OVERRIDE.format(service=service, ids=ids))
    keep.append(path)
    return path


def compose(args, env, overrides=()):
    cmd = ["docker", "compose", "-p", os.environ.get("USER", "openreal2sim")]
    cmd += ["-f", str(COMPOSE)]
    for path in overrides:
        cmd += ["-f", path]
    return subprocess.call(cmd + args, cwd=str(ROOT), env=env)


def check_host():
    if shutil.which("docker") is None:
        fail("docker is not installed or not on PATH")
    probe = subprocess.run(
        ["docker", "info", "--format", "{{json .Runtimes}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if probe.returncode != 0:
        fail(
            "cannot talk to the docker daemon. Is it running, and are you in the "
            "docker group (sudo usermod -aG docker $USER)?"
        )
    try:
        runtimes = json.loads(probe.stdout)
    except ValueError:
        runtimes = {}
    if "nvidia" not in runtimes:
        fail(
            "the nvidia container runtime is missing. Install the NVIDIA Container "
            "Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
        )
    if subprocess.call(
        ["docker", "compose", "version"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ):
        fail("docker compose (v2 plugin) is missing")
    if not COMPOSE.is_file():
        fail("compose file not found at %s" % COMPOSE)
    print("docker, the nvidia runtime and %s are in place" % COMPOSE)


def make_dirs():
    made = []
    for name in REPO_DIRS:
        path = ROOT / name
        if not path.exists():
            path.mkdir(parents=True)
            made.append(str(path))
    for name in CACHE_DIRS:
        path = Path.home() / ".cache" / name
        if not path.exists():
            path.mkdir(parents=True)
            made.append(str(path))
    print("created %d host directories" % len(made))
    for path in made:
        print("  " + path)


def setup(args):
    check_host()
    make_dirs()
    env = compose_env()
    if args.build:
        print("building %s (hours, only needed when a Dockerfile changes)" % ", ".join(SERVICES))
        if compose(["build"] + list(SERVICES), env):
            fail("build failed")
    else:
        print("pulling %s" % ", ".join(SERVICES))
        if compose(["pull"] + list(SERVICES), env):
            fail(
                "pull failed. The packages are private: if the error above says "
                "denied or unauthorized, run docker login ghcr.io with a token "
                "that has the read:packages scope, or pass --build"
            )
    if args.skip_weights:
        print("skipping the weights download")
        return
    if MARKER.exists() and not args.force_weights:
        print("weights already installed (%s); pass --force-weights to redo it" % MARKER)
        return
    print("downloading the weights with %s (~22 GB, around 20 minutes)" % INSTALL)
    keep = []
    overrides = [gpu_override("reconstruction", args.gpus, keep)] if args.gpus else []
    code = compose(
        ["run", "--rm", "reconstruction", "python", INSTALL], env, overrides
    )
    for path in keep:
        os.unlink(path)
    if code:
        fail("%s failed with exit code %d; the weights are incomplete" % (INSTALL, code))
    MARKER.parent.mkdir(parents=True, exist_ok=True)
    MARKER.write_text("written by init.py after %s exited 0\n" % INSTALL)
    print("weights installed; marker written to %s" % MARKER)


def run(args, command):
    """docker compose run --rm the service with the given command."""
    keep = []
    overrides = [gpu_override(args.service, args.gpus, keep)] if args.gpus else []
    compose_args = ["run", "--rm"]
    for port in args.port:
        compose_args += ["--publish", port]
    code = compose(compose_args + [args.service] + command, compose_env(), overrides)
    for path in keep:
        os.unlink(path)
    return code


def split_command(argv):
    """Everything after a bare -- is the command to run in the container."""
    if "--" in argv:
        index = argv.index("--")
        return argv[:index], argv[index + 1:]
    return argv, []


def add_run_flags(parser):
    parser.add_argument("service", choices=SERVICES)
    parser.add_argument("--gpus", help="comma separated GPU indices, default all")
    parser.add_argument(
        "--port", action="append", default=[], metavar="HOST:CONTAINER",
        help="publish a port, for example 7860:7860 or 8000:5000",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand")

    prepare = subparsers.add_parser("setup", help="check the host, pull the images, download the weights")
    prepare.add_argument("--build", action="store_true", help="build the images instead of pulling them")
    prepare.add_argument("--skip-weights", action="store_true", help="do not run " + INSTALL)
    prepare.add_argument("--force-weights", action="store_true", help="run " + INSTALL + " again")
    prepare.add_argument("--gpus", help="comma separated GPU indices, default all")

    stage = subparsers.add_parser("run", help="run a command in a service container")
    add_run_flags(stage)
    stage.add_argument("command", nargs="*", help="-- then the command to run")

    interactive = subparsers.add_parser("shell", help="open bash in a service container")
    add_run_flags(interactive)

    head, tail = split_command(sys.argv[1:])
    args = parser.parse_args(head)
    if args.subcommand is None:
        parser.print_help()
        return 1
    if args.subcommand == "setup":
        setup(args)
        return 0
    if args.subcommand == "shell":
        return run(args, ["bash"])
    command = args.command + tail
    if not command:
        fail("nothing to run: put the command after --")
    return run(args, command)


if __name__ == "__main__":
    sys.exit(main())
