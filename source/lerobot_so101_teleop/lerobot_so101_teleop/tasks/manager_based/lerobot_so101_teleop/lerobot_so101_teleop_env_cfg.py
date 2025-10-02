# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.sim import get_current_stage
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
            pos=(0.2630187963533502, -0.07503431658766903, 0.04273062786311794),
            rot=(0.98098075, 0.0, 0.0, 0.19410504),
        ),
    )

    ring_01 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Ring01",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.27783690762421165, 0.08424602121311484, 0.12378733765868938),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ring_02 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Ring02",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.27129651520577125, 0.016693925458027874, 0.09686271112669181),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ring_03 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Ring03",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2565863897049621, 0.12367459118708185, 0.09971861729194248),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ring_04 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Ring04",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2973915909379369, 0.10406110833155013, 0.06506421666657526),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    ring_05 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Environment/rock_a_stack/Ring05",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.19686964137123242, 0.12395675122242616, 0.06984264096164769),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Lights
    rect_01 = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Environment/room/lights/RectLight")
    rect_02 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Environment/room/lights/RectLight_01"
    )

    # robot
    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


##
# MDP settings
##


def randomize_light_exposure(
    env,
    env_ids: torch.Tensor | None,
    exposure_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = None,
):

    stage = get_current_stage()
    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    exposure = math_utils.sample_uniform(*exposure_range, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        prim = stage.GetPrimAtPath(asset_prim_path)
        if prim.IsValid():
            prim.GetAttribute("inputs:exposure").Set(exposure)
            print(
                f"[INFO]: Exposure set on light prim {asset_cfg.name} to {exposure:.2f}"
            )


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
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("rect_01"),
        },
    )

    reset_light_02_exposure = EventTerm(
        func=randomize_light_exposure,
        mode="reset",
        params={
            "exposure_range": (-3.0, 1.0),
            "asset_cfg": SceneEntityCfg("rect_02"),
        },
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
    reset_ring_01 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring_01"),
        },
    )
    reset_ring_02 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring_02"),
        },
    )
    reset_ring_03 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring_03"),
        },
    )
    reset_ring_04 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring_04"),
        },
    )
    reset_ring_05 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0, 0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ring_05"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass

    # # (1) Constant running reward
    # alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # # (2) Failure penalty
    # terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
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
        self.viewer.eye = (0.6, 0.6, 0.3)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
