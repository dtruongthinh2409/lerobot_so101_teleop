# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
SO-ARM101 Optimized Configuration - Based on Real Servo Specs

Tuned based on:
- Feetech STS3215 servo specifications
- Actual gear ratios per joint
- Empirical analysis from real robot data
- Episode 1/2 performance analysis
"""

import os
import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

here = os.path.dirname(os.path.abspath(__file__))


SO101_OPTIMIZED_CFG = ArticulationCfg(
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
            "Rotation": -0.2736,     # Real robot starting pose
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
    ),
    actuators={
        # ROTATION (Gear: 1/191, Torque: 34.4 N-m)
        # Analysis: MAE 0.09-0.11° (very tight) vs Real 0.65-0.85°
        # Could be slightly looser to match real robot
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,        # Moderate increase from 50 (high torque available)
            damping=0.7,         # Increase from 0.6 (reduce overshoot)
        ),
        
        # PITCH (Gear: 1/345, Torque: 62.1 N-m - HIGHEST)
        # Analysis: MAE 0.78-0.85° vs Real 1.10-1.51° (good match!)
        # Fights gravity, needs stability
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,        # Increase from 25 (highest torque ratio)
            damping=0.8,         # Increase from 0.6 (heavy load, prevent sag/oscillation)
        ),
        
        # ELBOW (Gear: 1/191, Torque: 34.4 N-m)
        # Analysis: MAE 0.75-1.08° vs Real 1.81-2.90°
        # Could be looser to match real robot dynamics
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,        # Keep moderate (allow some natural tracking error)
            damping=0.7,         # Increase from 0.6 (better response)
        ),
        
        # WRIST PITCH (Gear: 1/147, Torque: 26.5 N-m)
        # Analysis: MAE 0.53-0.54° vs Real 0.64-0.83° (excellent match!)
        # Keep current tuning - working very well
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,        # Moderate (lighter load than main arm)
            damping=0.5,         # Good balance
        ),
        
        # WRIST ROLL (Gear: 1/147, Torque: 26.5 N-m)
        # Analysis: MAE 0.31-0.35° vs Real 0.22-0.30° (PERFECT!)
        # Don't change - this is ideal
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,         # Perfect as-is (matches real robot exactly)
            damping=0.5,         # Perfect as-is
        ),
        
        # GRIPPER/JAW (Gear: 1/147, Torque: 26.5 N-m)
        # Analysis: MAE 2.84-3.69° vs Real 2.59-3.43° (very good!)
        # Keep compliant for grasping
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30,
            stiffness=4,         # Low for compliant grasping
            damping=0.3,         # Good as-is
        ),
    },
)


# Conservative version (if optimized feels too stiff)
SO101_CONSERVATIVE_CFG = SO101_OPTIMIZED_CFG.copy()
SO101_CONSERVATIVE_CFG.actuators = {
    "rotation": ImplicitActuatorCfg(
        joint_names_expr=["Rotation"],
        effort_limit_sim=30,
        stiffness=50,        # Original
        damping=0.6,         # Original
    ),
    "pitch": ImplicitActuatorCfg(
        joint_names_expr=["Pitch"],
        effort_limit_sim=30,
        stiffness=25,        # Original
        damping=0.7,         # Slight increase for stability
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["Elbow"],
        effort_limit_sim=30,
        stiffness=25,        # Original
        damping=0.6,         # Original
    ),
    "wrist_pitch": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Pitch"],
        effort_limit_sim=30,
        stiffness=12,
        damping=0.5,
    ),
    "wrist_roll": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Roll"],
        effort_limit_sim=30,
        stiffness=7,
        damping=0.5,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["Jaw"],
        effort_limit_sim=30,
        stiffness=4,
        damping=0.3,
    ),
}


##
# USAGE
##
"""
To use:

1. In base_env_cfg.py:
   from lerobot_so101_teleop.assets.so101_optimized import SO101_OPTIMIZED_CFG
   robot: ArticulationCfg = SO101_OPTIMIZED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

2. Record test episode

3. Analyze:
   python analysis/scripts/analyze_datasets.py --episode <N>

4. Compare with Episode 1:
   python analysis/scripts/analyze_datasets.py --episode 1

KEY CHANGES from current:
- Separated all joints (no more grouping)
- Rotation: 50→55, damping 0.6→0.7
- Pitch: 25→30, damping 0.6→0.8 (more stable under gravity)
- Elbow: 25→25, damping 0.6→0.7
- Wrist/Gripper: Unchanged (already optimal)

EXPECTED RESULTS:
- Overall MAE: 1.0-1.2° (slightly tighter than Episode 1's 1.09°)
- Better pitch stability (reduced oscillation)
- Maintained excellent wrist/gripper performance

If this feels too stiff, use SO101_CONSERVATIVE_CFG instead.
"""

