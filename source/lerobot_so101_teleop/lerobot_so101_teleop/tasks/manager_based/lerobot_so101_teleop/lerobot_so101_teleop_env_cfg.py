# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


from . import mdp

##
# Pre-defined configs
##
from .assets.so101 import SO101_CFG


##
# Scene definition
##
@configclass
class LerobotSo101TeleopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    world_environment = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"source/lerobot_so101_teleop/lerobot_so101_teleop/tasks/manager_based/lerobot_so101_teleop/assets/stage.usda",
        ),
    )

    # Position values are grabbed from the USDA file, maybe there is a cleaner way to do this
    base_plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Base",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.263018, -0.0750343, 0.0427306),
            rot=(0.98098075, 0.0, 0.0, 0.19410504),
        ),
    )

    robot_pad = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment/table/Static/Cube_002"
    )

    # Lights
    rect_01 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment/room/lights/RectLight"
    )
    rect_02 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment/room/lights/RectLight_01"
    )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    def __post_init__(self) -> None:
        rings = [
            ("Ring01", (0.2778369, 0.0842460, 0.1237873)),
            ("Ring02", (0.2712965, 0.01669392, 0.0968627)),
            ("Ring03", (0.256586, 0.1236745, 0.0997186)),
            ("Ring04", (0.297391, 0.1040611, 0.0650642)),
            ("Ring05", (0.1968696, 0.1239567, 0.0698426)),
        ]
        for ring in rings:
            ring_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/" + ring[0],
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=ring[1],
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            setattr(self, ring[0], ring_cfg)


##
# MDP settings
##

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_positions = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"],
        scale=1,
        use_default_offset=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Reset robot position
    reset_robot_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "Rotation",
                    "Pitch",
                    "Elbow",
                    "Wrist_Pitch",
                    "Wrist_Roll",
                    "Jaw",
                ],
            ),
            "position_range": (0, 0),
            "velocity_range": (0, 0),
        },
    )

    # Randomize light exposure
    reset_light_01_exposure = EventTerm(
        func=mdp.randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("rect_01"),
        },
    )

    reset_light_02_exposure = EventTerm(
        func=mdp.randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("rect_02"),
        },
    )

    reset_physics_material_props = EventTerm(
        func=mdp.randomize_physics_material,
        mode="reset",
        params={},
    )

    # Reset physics objects to their initial positions
    reset_base_plate = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0, 0),
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("base_plate"),
        },
    )

    reset_robot_pad_orientation = EventTerm(
        func=mdp.randomize_static_asset_orientation,
        mode="reset",
        params={
            "pose_range": {"yaw": (math.pi-0.2, math.pi+0.2)},
            "asset_cfg": SceneEntityCfg("robot_pad"),
        },
    )

    def __post_init__(self) -> None:
        # Handle rings transform and physics material
        for ring in ["Ring01", "Ring02", "Ring03", "Ring04", "Ring05"]:
          
            reset_ring_transform = EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg(ring),
                },
            )
            setattr(self, f"reset_{ring}_transform", reset_ring_transform)
     
            reset_ring_physics_material = EventTerm(
                func=mdp.apply_physics_material,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(ring),
                },
            )
            setattr(self, f"reset_{ring}_phyiscs_material", reset_ring_physics_material)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    pass


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    pass


##
# Environment configuration
##

@configclass
class LerobotSo101TeleopEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: LerobotSo101TeleopSceneCfg = LerobotSo101TeleopSceneCfg(
        num_envs=4096, env_spacing=4.0
    )
    # Basic settings
    observations = None
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards = None  # No rewards for teleoperation
    terminations = None  # No terminations for teleoperation



    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5

        self.scene.num_envs = 1  # Always 1 env for teleoperation
        # viewer settings
        self.viewer.eye = (-0.25, -0.4, 0.22)
        self.viewer.lookat = (0.15, 0.0, 0.12)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.render.rendering_mode = "quality"
        self.sim.render.enable_translucency = False

        # create physics material
        physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        )
        self.physics_material_path = "/Looks/physicsMaterial"
        physics_material_cfg.func(self.physics_material_path, physics_material_cfg)
