# Real-to-Sim Robotic Simulation

This directory contains the code and resources necessary to set up a robotic simulation environment that mirrors real-world scenarios, using the reconstructed assets from the [reconstruction](../reconstruction/README.md) step.

Current supported physics simulators:

- [x] IssacLab ([IssacLab v2.0.2](https://isaac-sim.github.io/IsaacLab/v2.0.2/source/setup/ecosystem.html) & [IssacSim v4.5.0](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/index.html))
- [x] Maniskills
- [x] Mujoco

## Running the IsaacLab stage

```
python openreal2sim/steps/simulation/simulation.py [--config config/config.yaml] [--key KEY] [--stage STEP]
```

Steps, in order: `usd_conversion`, `sim_heuristic_manip`, `sim_randomize_rollout`.
`--stage` starts the run from that step, as in the other stage runners. See
[isaaclab/README.md](isaaclab/README.md) for the backend details. The maniskill
and mujoco backends keep their own entry points.
