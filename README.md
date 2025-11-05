# OpenReal2Sim
A toolbox for real-to-sim reconstruction and robotic simulation

## What We Can Do

- [x] Real-to-sim assets reconstruction from images and (generated) videos
- [x] IsaacLab support for scenario import, camera setup, and rendering from the same input viewpoint 
- [x] IsaacLab support for robotic trajectory generation by cross-embodiment transfer from videos
- [x] Preliminary Maniskills support

## Installation

Clone this repository recursively:
```
git clone git@github.com:PointsCoder/OpenReal2Sim.git --recurse-submodules
```

Next, we need to set up the Python environment. We recommend using `docker` for managing dependencies.

Please refer to [docker installation](docker/README.md) for launching the docker environment.

## How to Use

Please follow the step-by-step user guide [here](docs/HowToUse.md).

## Citation
If you find this repository useful in your research, please consider citing:
```
@misc{openreal2sim,
  title={OpenReal2Sim: A Toolbox for Real-to-Sim Reconstruction and Robotic Simulation},
  author={OpenReal2Sim Development Team},
  year={2025}
}
```
```
@inproceedings{rola,
  title={Robot learning from any images},
  author={Zhao, Siheng and Mao, Jiageng and Chow, Wei and Shangguan, Zeyu and Shi, Tianheng and Xue, Rong and Zheng, Yuxi and Weng, Yijia and You, Yang and Seita, Daniel and others},
  booktitle={Conference on Robot Learning},
  pages={4226--4245},
  year={2025},
  organization={PMLR}
}
```