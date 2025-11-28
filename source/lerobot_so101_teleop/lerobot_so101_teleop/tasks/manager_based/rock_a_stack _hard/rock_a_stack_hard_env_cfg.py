import os
import math
import numpy as np

import isaaclab.sim as sim_utils

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.sensors import TiledCameraCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaacsim.core.utils.rotations import euler_angles_to_quat

import isaaclab.envs.mdp as mdp_lib

from . import mdp


# import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.utils import configclass

from lerobot_so101_teleop import assets
from lerobot_so101_teleop.mdp import (
    randomize_physics_material,
    reset_root_state_uniform,
    reset_rings_transform_state_uniform,
)

# from lerobot_so101_teleop.assets.so101 import SO101_EVAL_CFG
from ..base.base_env_cfg import BaseEnvCfg, LerobotSo101BaseSceneCfg, EventCfg

assets_path = os.path.dirname(os.path.abspath(assets.__file__))


@configclass
class LerobotSo101RockAStackHardSceneCfg(LerobotSo101BaseSceneCfg):

    camera_external = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/external_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            fisheye_nominal_height=480,
            fisheye_nominal_width=640,
            fisheye_optical_centre_x=320,
            fisheye_optical_centre_y=120, # hmmm
            fisheye_max_fov=170,
            fisheye_polynomial_a=0,
            fisheye_polynomial_b=0.0015,
            fisheye_polynomial_c=0,
            fisheye_polynomial_d=0,
            fisheye_polynomial_e=0,
            fisheye_polynomial_f=0,
        ),
        # place the camera at the mount point
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.57, 0.0, 0.055),
            rot=euler_angles_to_quat(np.array([115, 0, 90]), degrees=True),
            convention="opengl",
        ),
    )

    rock_a_stack = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RockAStack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assets_path}/usd/real-rock-a-stack.usd",
        ),
        # to deconflict with the robot during initialization
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.2, 0, 0.1),
            rot=(0.0, 0.0, 0.0, 1),
        ),
    )

    base_plate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/RockAStack/base",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.2, 0, 0.1),
            rot=(0.98098075, 0.0, 0.0, 0.19410504),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # rings
        for idx in range(1, 5):
            far_pos = (10, 0, 0.03)
            near_pos = (0.2, 0, 0.03) 
            pos = near_pos if idx == 4 else far_pos # keep only the last ring
            ring_cfg = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RockAStack/" + f"ring{idx:02d}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=pos,  # intial pose to not conflict with the robot
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            setattr(self, f"Ring{idx:02d}", ring_cfg)
            print(f"ring{idx:02d}")


@configclass
class RockAStackEventCfg(EventCfg):
    """Configuration for events."""

    reset_physics_material_props = EventTerm(
        func=randomize_physics_material,
        mode="reset",
        params={
            "asset_names": [
                "Ring01",
                "Ring02",
                "Ring03",
                "Ring04",
                "base_plate",
            ]
        },
    )

    # Reset physics objects to their initial positions
    reset_base_plate = EventTerm(
        func=reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.2 - 0.19, 0.2 - 0.21),
                # offset by -2 to not conflict with the robot
                "y": (-0.03, -0.05),
                # only scatter on right side of the robot
                "z": (0.04, 0.04),
                # no need to scatter on z
                "roll": (0, 0),
                "pitch": (0, 0),
                "yaw": (-0.5, 0.5),
                # randomize yaw
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("base_plate"),
        },
    )

    reset_ring_transform = EventTerm(
        func=reset_rings_transform_state_uniform,
        mode="reset",
        params={
            "asset_names": ["Ring01", "Ring02", "Ring03", "Ring04"],
            "x_y_pose_range": {
                "x": (-0.02, 0.02),
                # offset from the initial position to not conflict with the robot
                "y": (0.05, 0.08),
                # only scatter on left side of the robot
            },
        },
    )



@configclass
class RockAStackHardEnvCfg(BaseEnvCfg):
    """Configuration for the rock-a-stack environment."""

    scene: LerobotSo101RockAStackHardSceneCfg = LerobotSo101RockAStackHardSceneCfg()
    events: RockAStackEventCfg = RockAStackEventCfg()

    # robot: ArticulationCfg = SO101_EVAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


    def __post_init__(self) -> None:
        """Post initialization."""
        super().__post_init__()

        # self.scene.robot = self.robot
        

        # create physics material
        physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.5,
        )
        self.physics_material_path = "/Looks/physicsMaterial"
        physics_material_cfg.func(self.physics_material_path, physics_material_cfg)



# @configclass
# class RockAStackEvalEventCfg(RockAStackEventCfg):
#     reset_ring_transform = EventTerm(
#         func=reset_rings_transform_state_uniform,
#         mode="reset",
#         params={
#             "asset_names": ["Ring01", "Ring02", "Ring03", "Ring04"],
#             "x_y_pose_range": {
#                 "x": (-0.1*0.2, 0.1*0.2),
#                 "y": (0.0, 0.1*0.2),
#             },
#         },
#     )


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # Sparse reward: binary success when ring is aligned with pole in x-y plane
#     ring_xy_aligned = RewTerm(
#         func=mdp.ring_on_pole_xy_alignment,
#         weight=1.0,
#         params={
#             "tolerance": 0.005,  # 5mm 
#             "ring_cfg": SceneEntityCfg("Ring04"),
#             "pole_cfg": SceneEntityCfg("base_plate"),
#         },
#     )
    
#     # Dense shaped reward: encourages moving ring closer to pole
#     ring_xy_distance = RewTerm(
#         func=mdp.ring_on_pole_xy_distance,
#         weight=0.5,
#         params={
#             "std": 0.1,  # Standard deviation for smooth reward shaping
#             "ring_cfg": SceneEntityCfg("Ring04"),
#             "pole_cfg": SceneEntityCfg("base_plate"),
#         },
#     )
    
#     # Complete success reward: ring is aligned AND at correct height (inserted)
#     ring_insertion = RewTerm(
#         func=mdp.ring_insertion_success,
#         weight=10.0,  # Large bonus for complete success
#         params={
#             "xy_tolerance": 0.004,  # 4mm tolerance in x-y plane (5x stricter)
#             "z_threshold": 0.06,  # Ring must be at ~6cm height to be considered inserted (slid down onto pole)
#             "ring_cfg": SceneEntityCfg("Ring04"),
#             "pole_cfg": SceneEntityCfg("base_plate"),
#         },
#     )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Dynamic timeout: starts at 200 steps, extends by 100 when ring gets close to pole
    time_out = DoneTerm(
        func=mdp.DynamicTimeout, 
        time_out=True,
        params={
            "base_timeout": 250,  # Initial timeout in steps
            "extension": 150,     # Additional steps when condition is met
            "tolerance": 0.02,    # Distance threshold (2cm) to trigger extension
            "ring_cfg": SceneEntityCfg("Ring04"),
            "pole_cfg": SceneEntityCfg("base_plate"),
        },
    )
    
    # Terminate on successful insertion (task success)
    success = DoneTerm(
        func=mdp.ring_insertion_success,
        time_out=False,  # This is a termination (task completion), not a timeout
        params={
            "xy_tolerance": 0.005,  # 5mm tolerance in x-y plane
            "z_threshold": 0.06,  # Ring must be at ~6cm height to be considered inserted
            "ring_cfg": SceneEntityCfg("Ring04"),
            "pole_cfg": SceneEntityCfg("base_plate"),
        },
    )


@configclass
class RockAStackHardEnvEvalCfg(RockAStackHardEnvCfg):

    # robot: ArticulationCfg = SO101_EVAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # events: RockAStackEvalEventCfg = RockAStackEvalEventCfg()
    # rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()  


    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 400 / 60.0 
        # self.scene.robot = self.robot