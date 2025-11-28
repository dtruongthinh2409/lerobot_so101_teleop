# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
here = os.path.dirname(os.path.abspath(__file__))


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{here}/usd/SO-ARM101-USD.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": 0.0,
            "Pitch": 0,
            "Elbow": 0,
            "Wrist_Pitch": 0,
            "Wrist_Roll": 0,
            "Jaw": 0.0,
        },
    ),
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=[
                "Rotation",
                "Pitch"
            ],
            effort_limit_sim=30,
            # velocity_limit_sim=2,
            stiffness=8,
            damping=0.8,
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch", "Elbow", "Wrist_Pitch"],
            effort_limit_sim=30,
            # velocity_limit_sim=2,
            stiffness=8,
            damping=0.8,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            # velocity_limit_sim=2,
            stiffness=8,
            damping=0.8,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30,
            # velocity_limit_sim=2,
            stiffness=3,
            damping=.3,
        ),
    },
)


SO101_REAL_CFG = SO101_CFG.copy()
# observing at the table
SO101_REAL_CFG.init_state.joint_pos = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745, 
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}

# amperical testing against real recordings
SO101_REAL_CFG.actuators["rotation"].stiffness = 50
SO101_REAL_CFG.actuators["rotation"].damping = .6
SO101_REAL_CFG.actuators["pitch"].stiffness = 25
SO101_REAL_CFG.actuators["pitch"].damping = .6
SO101_REAL_CFG.actuators["wrist"].stiffness = 7
SO101_REAL_CFG.actuators["wrist"].damping = .5
SO101_REAL_CFG.actuators["gripper"].stiffness = 4
SO101_REAL_CFG.actuators["gripper"].damping = 0.3

