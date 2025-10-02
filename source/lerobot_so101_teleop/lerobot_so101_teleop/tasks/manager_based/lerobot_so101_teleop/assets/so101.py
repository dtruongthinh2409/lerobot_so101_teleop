# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
##
import os
here = os.path.dirname(os.path.abspath(__file__))


SO101_CFG = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
        usd_path=f"{here}/props/SO-ARM101-USD.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=1
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
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "Rotation", 
                "Pitch",
                "Elbow", 
                "Wrist_Pitch", 
                "Wrist_Roll",
                # "Jaw"
                ],
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
            damping=0.3,
        ),
    },
)
