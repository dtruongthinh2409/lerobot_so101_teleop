# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch


import carb

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import lerobot_so101_teleop.tasks  # noqa: F401
from lerobot_so101_teleop.keyboard import KeyboardControl
from lerobot_so101_teleop.lerobot_interface import LeRobotSO101Interface


def main():

    keyboard_control = KeyboardControl()

    lerobot_cfg = {"port": "/dev/ttyACM0", "id": "leader_arm_1"}
    lerobot_interface = LeRobotSO101Interface(cfg=lerobot_cfg)

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Click 'R' to reset the world")
    # reset environment
    env.reset()

    # Allocate action tensor
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # disable translucency
    # carb_settings = carb.settings.get_settings()
    # carb_settings.set("/rtx/translucency/enabled", False)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            real_action = lerobot_interface.teleop_dev.get_action()
            actions[:] = lerobot_interface.get_mapped_actions_vectorized(
                real_action
            ).to(env.unwrapped.device)

            # apply actions
            env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

    env.close()


if __name__ == "__main__":

    main()

    simulation_app.close()
