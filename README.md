# OpenReal2Sim
OpenReal2Sim is a toolbox for real-to-sim reconstruction and robotic simulation. 

It supports our multiple research projects, including [PhysWorld](https://pointscoder.github.io/PhysWorld_Web/) and [RoLA](https://sihengz02.github.io/RoLA/).

![image](./assets/real_1x4.jpg)
![image](./assets/sim_1x4.jpg)

## What We Can Do

- [x] Real-to-sim assets reconstruction from images and (generated) videos
- [x] IsaacLab support for scenario import, camera setup, and rendering from the same input viewpoint 
- [x] Preliminary IsaacLab support for robotic trajectory generation by cross-embodiment transfer from videos
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
@inproceedings{physworld,
  title={Robot Learning from a Physical World Model},
  author = {Mao, Jiageng and He, Sicheng and Wu, Hao-Ning and You, Yang and Sun, Shuyang and Wang, Zhicheng and Bao, Yanan and Chen, Huizhong and Guibas, Leonidas and Guizilini, Vitor and Zhou, Howard and Wang, Yue},
  year={2025},
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