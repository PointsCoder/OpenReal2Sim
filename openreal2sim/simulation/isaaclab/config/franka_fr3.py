# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka FR3 robot.

The following configurations are available:

* :obj:`FRANKA_FR3_CFG`: Franka FR3 robot with FR3 hand
* :obj:`FRANKA_FR3_HIGH_PD_CFG`: Franka FR3 robot with FR3 hand with stiffer PD control
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FRANKA_FR3_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ISAACLAB_NUCLEUS_DIR.replace("http://", "https://").replace("IsaacLab", "Robots/Franka/FR3/fr3.usd"),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "fr3_joint1": 0.0,
            "fr3_joint2": -0.569,
            "fr3_joint3": 0.0,
            "fr3_joint4": -2.810,
            "fr3_joint5": 0.0,
            "fr3_joint6": 3.037,
            "fr3_joint7": 0.741,
            "fr3_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka FR3 robot."""


FRANKA_FR3_HIGH_PD_CFG = FRANKA_FR3_CFG.copy()
FRANKA_FR3_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_shoulder"].damping = 80.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].stiffness = 400.0
FRANKA_FR3_HIGH_PD_CFG.actuators["fr3_forearm"].damping = 80.0
"""Configuration of Franka FR3 robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
