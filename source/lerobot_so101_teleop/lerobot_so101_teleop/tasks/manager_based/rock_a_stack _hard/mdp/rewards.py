# Copyright (c) 2025
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Debug flag - set to True to enable detailed logging from reward functions
DEBUG_REWARDS = False


def ring_on_pole_xy_alignment(
    env: ManagerBasedRLEnv,
    tolerance: float,
    ring_cfg: SceneEntityCfg = SceneEntityCfg("Ring04"),
    pole_cfg: SceneEntityCfg = SceneEntityCfg("base_plate"),
) -> torch.Tensor:
    """Reward the agent when the ring is aligned with the pole in x-y plane.
    
    This function checks if the ring (Ring04) and pole (base_plate) are within a specified
    tolerance in the x and y world coordinates. Returns 1.0 if aligned, 0.0 otherwise.
    
    Args:
        env: The RL environment instance.
        tolerance: The maximum distance (in meters) in the x-y plane to consider as aligned.
        ring_cfg: Configuration for the ring object (default: Ring04).
        pole_cfg: Configuration for the pole object (default: base_plate).
        
    Returns:
        A tensor of shape (num_envs,) with 1.0 for aligned environments and 0.0 otherwise.
    """
    # Extract the ring and pole objects from the scene
    ring: RigidObject = env.scene[ring_cfg.name]
    pole: RigidObject = env.scene[pole_cfg.name]
    
    # Get world positions: (num_envs, 3) where columns are [x, y, z]
    ring_pos_w = ring.data.root_pos_w
    pole_pos_w = pole.data.root_pos_w
    
    # Calculate distance in x-y plane only (ignoring z)
    xy_distance = torch.norm(ring_pos_w[:, :2] - pole_pos_w[:, :2], dim=1)
    
    # Debug logging
    if DEBUG_REWARDS and torch.any(xy_distance <= tolerance):
        print(f"[DEBUG] ring_xy_aligned: distance={xy_distance[0].item():.4f}m, tolerance={tolerance}m, ALIGNED!")
    
    # Return 1.0 if within tolerance, 0.0 otherwise
    return torch.where(xy_distance <= tolerance, 1.0, 0.0)


def ring_on_pole_xy_distance(
    env: ManagerBasedRLEnv,
    std: float,
    ring_cfg: SceneEntityCfg = SceneEntityCfg("Ring04"),
    pole_cfg: SceneEntityCfg = SceneEntityCfg("base_plate"),
) -> torch.Tensor:
    """Shaped reward based on the distance between ring and pole in x-y plane using tanh-kernel.
    
    This function provides a smooth reward that encourages the agent to move the ring closer
    to the pole. The reward is highest (1.0) when the ring is directly over the pole and
    decreases smoothly as the distance increases.
    
    Args:
        env: The RL environment instance.
        std: Standard deviation for the tanh kernel (controls reward smoothness).
        ring_cfg: Configuration for the ring object (default: Ring04).
        pole_cfg: Configuration for the pole object (default: base_plate).
        
    Returns:
        A tensor of shape (num_envs,) with shaped rewards based on x-y distance.
    """
    # Extract the ring and pole objects from the scene
    ring: RigidObject = env.scene[ring_cfg.name]
    pole: RigidObject = env.scene[pole_cfg.name]
    
    # Get world positions: (num_envs, 3) where columns are [x, y, z]
    ring_pos_w = ring.data.root_pos_w
    pole_pos_w = pole.data.root_pos_w
    
    # Calculate distance in x-y plane only (ignoring z)
    xy_distance = torch.norm(ring_pos_w[:, :2] - pole_pos_w[:, :2], dim=1)
    
    # Return shaped reward using tanh kernel
    # Returns 1.0 when distance is 0, approaches 0 as distance increases
    return 1.0 - torch.tanh(xy_distance / std)


def ring_insertion_success(
    env: ManagerBasedRLEnv,
    xy_tolerance: float,
    z_threshold: float,
    ring_cfg: SceneEntityCfg = SceneEntityCfg("Ring04"),
    pole_cfg: SceneEntityCfg = SceneEntityCfg("base_plate"),
) -> torch.Tensor:
    """Complete success reward: ring is aligned with pole in x-y AND at appropriate height.
    
    NOTE: This logic is mirrored in terminations.py/ring_insertion_success() which uses
    the same conditions to terminate the episode. Keep parameters in sync!
    
    This function checks both horizontal alignment (x-y) and vertical position (z) to determine
    if the ring has been successfully inserted onto the pole. The ring must be:
    1. Within xy_tolerance of the pole's x-y position
    2. Below z_threshold height (indicating it's been lowered onto the pole)
    
    Args:
        env: The RL environment instance.
        xy_tolerance: Maximum distance (in meters) in x-y plane to consider as aligned.
        z_threshold: Maximum z-height (in meters) for the ring to be considered inserted.
        ring_cfg: Configuration for the ring object (default: Ring04).
        pole_cfg: Configuration for the pole object (default: base_plate).
        
    Returns:
        A tensor of shape (num_envs,) with 1.0 for successful insertion, 0.0 otherwise.
    """
    # Extract the ring and pole objects from the scene
    ring: RigidObject = env.scene[ring_cfg.name]
    pole: RigidObject = env.scene[pole_cfg.name]
    
    # Get world positions: (num_envs, 3) where columns are [x, y, z]
    ring_pos_w = ring.data.root_pos_w
    pole_pos_w = pole.data.root_pos_w
    
    # Calculate distance in x-y plane only
    xy_distance = torch.norm(ring_pos_w[:, :2] - pole_pos_w[:, :2], dim=1)
    
    # Check if ring is aligned in x-y
    is_aligned = xy_distance <= xy_tolerance
    
    # Check if ring is at appropriate height (inserted onto pole)
    is_at_height = ring_pos_w[:, 2] <= z_threshold
    
    # Debug logging
    if DEBUG_REWARDS:
        ring_z = ring_pos_w[0, 2].item()
        dist = xy_distance[0].item()
        if is_aligned[0] and is_at_height[0]:
            print(f"[DEBUG] 🎉 INSERTION SUCCESS! xy_dist={dist:.4f}m, ring_z={ring_z:.4f}m")
        elif is_aligned[0]:
            print(f"[DEBUG] Ring aligned but too high: xy_dist={dist:.4f}m, ring_z={ring_z:.4f}m (need <{z_threshold}m)")
        elif is_at_height[0]:
            print(f"[DEBUG] Ring at correct height but not aligned: xy_dist={dist:.4f}m, ring_z={ring_z:.4f}m")
    
    # Return 1.0 only if both conditions are met
    return torch.where(is_aligned & is_at_height, 1.0, 0.0)

