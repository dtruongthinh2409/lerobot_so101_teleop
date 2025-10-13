# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Isaac Lab SO-101 Teleop agent.")
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

parser.add_argument("--repo_id", type=str, default=None, help="Repository ID to store the dataset.")
parser.add_argument("--repo_root", type=str, default=None, help="Repository root to store the dataset.")
parser.add_argument("--task_name", type=str, default=None, help="Name of the task.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# always enable cameras to record video
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch
import os
import time


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import lerobot_so101_teleop.tasks  # noqa: F401
from lerobot_so101_teleop.keyboard import KeyboardControl
from lerobot_so101_teleop.lerobot_interface import LeRobotSO101Interface
from lerobot_so101_teleop.lerobot_recorder import LeRobotRecorder

import omni.replicator.core as rep
from isaaclab.utils import convert_dict_to_backend


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
    print(f"[INFO]: Click 'S' to start/stop recording; 'R' will also stop recording")
    # reset environment
    env.reset()

    # Allocate action tensor
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)

    # simulate environment

    # camera
    camera = env.unwrapped.scene["gripper_cam"]

    # Recording dataset
    if all([args_cli.repo_id, args_cli.repo_root, args_cli.task_name]):
        recording_mode = True
    else:
        recording_mode = False

    if recording_mode:
        recorder = LeRobotRecorder(
            task_name=args_cli.task_name,
            repo_id=args_cli.repo_id,
            dataset_root=args_cli.repo_root,
            fps=30,
            device=env.unwrapped.device,
            rgb_height=env.scene.cfg.gripper_cam.height,
            rgb_width=env.scene.cfg.gripper_cam.width,
        )
        try:
            recorder.init_dataset() 
        except ValueError:
            print(f"[ERROR]: Failed to initialize dataset. folder already exists")
            env.close()
            simulation_app.close()

    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            real_action = lerobot_interface.teleop_dev.get_action()
            actions[:] = lerobot_interface.get_mapped_actions_vectorized(
                real_action
            ).to(env.unwrapped.device)

            # apply actions
            obs, _, _, _, _ = env.step(actions)

            if keyboard_control.reset_world:
                keyboard_control.reset_world = False
                env.reset()
                continue

            if recording_mode and keyboard_control.recording:
                recorder.push_frame_to_buffer(
                    real_action, 
                    obs["policy"][0], 
                    camera.data.output["rgb"][0]
                )
   
    env.close()


if __name__ == "__main__":

    main()

    while True:
        simulation_app.update()

    simulation_app.close()
