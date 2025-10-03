import torch

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

from isaaclab.sim import get_current_stage
from isaaclab.managers import SceneEntityCfg

from isaacsim.core.prims import XFormPrim

def randomize_static_asset_orientation(
    env,
    env_ids: torch.Tensor | None,
    pose_range: dict | None,
    asset_cfg: SceneEntityCfg = None,
):

    asset = env.scene[asset_cfg.name]
    asset_prim_path = asset.prim_paths[0]

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.unwrapped.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=env.unwrapped.device)
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    asset_xform = XFormPrim(prim_paths_expr=asset_prim_path)

    with Sdf.ChangeBlock():
        asset_xform.set_local_poses(orientations=orientations)


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


def apply_physics_material(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = None,
):

    asset = env.scene[asset_cfg.name]
    asset_prims = sim_utils.find_matching_prims(asset.cfg.prim_path)

    with Sdf.ChangeBlock():
        for asset_prim in asset_prims:
            sim_utils.bind_physics_material(
                asset_prim.GetPath(), env.cfg.physics_material_path
            )


def randomize_physics_material(env, env_ids: torch.Tensor | None):

    stage = get_current_stage()
    physics_material_prim = stage.GetPrimAtPath(env.cfg.physics_material_path)

    static_friction = math_utils.sample_uniform(0.0, 0.5, (1,), device="cpu").item()
    dynamic_friction = math_utils.sample_uniform(0.0, 0.5, (1,), device="cpu").item()
    restitution = math_utils.sample_uniform(0.0, 0.3, (1,), device="cpu").item()

    with Sdf.ChangeBlock():
        physics_material_prim.GetAttribute("physics:staticFriction").Set(
            static_friction
        )
        physics_material_prim.GetAttribute("physics:dynamicFriction").Set(
            dynamic_friction
        )
        physics_material_prim.GetAttribute("physics:restitution").Set(restitution)
        print(
            f"[INFO]: Physics material: Static Friction: {static_friction:.2f}, Dynamic Friction: {dynamic_friction:.2f}, Restitution: {restitution:.2f}"
        )
