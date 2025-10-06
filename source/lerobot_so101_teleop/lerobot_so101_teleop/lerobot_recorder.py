import os

import torch
import numpy as np


import omni.kit.app
from carb.eventdispatcher import get_eventdispatcher, Event
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class LeRobotRecorder:

    STOP_RECORDING_EVENT: str = "lerobot_so101_teleop.stop_recording"

    def __init__(
        self, 
        task_name: str, 
        repo_id: str, 
        dataset_root: str, 
        fps: int, 
        device: str,
        rgb_height: int,
        rgb_width: int,
    ):

        self.fps = fps
        self.dt = 1 / self.fps
        self.rgb_height = rgb_height
        self.rgb_width = rgb_width

        self.FOLLOWER_OBS_FEATURES = {
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            },
            "observation.camera.gripper": {
                "dtype": "video",
                "shape": (self.rgb_height, self.rgb_width, 3),
                "names": ["height", "width", "channels"],
            },
        }

        self.LEADER_ACTION_FEATURES = {
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": [
                    "shoulder_pan.pos",
                    "shoulder_lift.pos",
                    "elbow_flex.pos",
                    "wrist_flex.pos",
                    "wrist_roll.pos",
                    "gripper.pos",
                ],
            }
        }

        self.SO101_ACTION_NAMES = [
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        ]

        self.repo_id = repo_id
        self.dataset_root = dataset_root
        self.task_name = task_name
        self.dataset_features = {
            **self.FOLLOWER_OBS_FEATURES,
            **self.LEADER_ACTION_FEATURES,
        }


        self.device = device
        self.capcity = 3 * 60 * self.fps
        self.current_frame = 0

        self.action_buffers = []
        self.observation_buffer_tensor = None
        self.rgb_buffer_tensor = None

        self.stop_recording_sub = get_eventdispatcher().observe_event(
            observer_name="stop_recording_observer",
            event_name=self.STOP_RECORDING_EVENT,
            on_event=self.save_episode,
        )

    def check_dataset_exists(self):
        if os.path.exists(self.dataset_root):
            return True
        else:
            return False

    def init_dataset(self):
        if self.check_dataset_exists():
            try:
                self.dataset = LeRobotDataset(
                    self.repo_id,
                    root=self.dataset_root,
                )
                print(f"[INFO]: Existing dataset initialized - {self.dataset.root}")
                return
            except:
                raise ValueError(
                    f"[ERROR]: Dataset folder exists but cannot be initialized at {self.dataset_root}"
                )

        self.dataset = LeRobotDataset.create(
            self.repo_id,
            fps=self.fps,
            features=self.dataset_features,
            root=self.dataset_root,
            robot_type="so101_follower",
        )

        print(f"[INFO]: New dataset initialized - {self.dataset.root}")

    def allocate_buffers(self):
        # self.action_buffer_tensor = torch.zeros((self.capcity, 6), dtype=torch.float32, device=self.device)
        self.observation_buffer_tensor = torch.zeros(
            (self.capcity, 6), dtype=torch.float32, device=self.device
        )
        self.rgb_buffer_tensor = torch.zeros(
            (self.capcity, self.rgb_height, self.rgb_width, 3), dtype=torch.uint8, device=self.device
        )

    def push_frame_to_buffer(self, action, observation, rgb):
        if self.current_frame >= self.capcity:
            # TODO: extand tensors to increase the buffer capacity if reached
            print(
                f"[INFO]: Reached the maximum capacity of the buffer. Skipping frame {self.current_frame}"
            )
            return

        if self.rgb_buffer_tensor is None:
            self.allocate_buffers()

        self.action_buffers.append(action.copy())
        self.observation_buffer_tensor[self.current_frame] = observation.clone()
        self.rgb_buffer_tensor[self.current_frame] = rgb.clone()

        self.current_frame += 1

    def add_dataset_frame(self, action, observation, rgb, frame_index):
        action = np.array(
            [action[name] for name in self.SO101_ACTION_NAMES], dtype=np.float32
        )
        frame = {
            "action": action,
            "observation.state": observation,
            "observation.camera.gripper": rgb,
        }
        frame_timestamp = frame_index * self.dt
        self.dataset.add_frame(frame, task=self.task_name, timestamp=frame_timestamp)

    def save_episode(self, event: Event):
        if event.event_name == self.STOP_RECORDING_EVENT:
            total_frames = len(self.action_buffers)
            print(f"[INFO]: Processing {total_frames} frames...")

            # batch copy to cpu
            cpu_rgb_buffer_tensor = self.rgb_buffer_tensor.to("cpu").numpy()
            cpu_obs_buffer_tensor = self.observation_buffer_tensor.to("cpu").numpy()

            for frame_index in tqdm(
                range(total_frames), desc="Processing frames", unit="frame"
            ):
                self.add_dataset_frame(
                    self.action_buffers[frame_index],
                    cpu_obs_buffer_tensor[frame_index],
                    cpu_rgb_buffer_tensor[frame_index],
                    frame_index,
                )

            self.dataset.save_episode()
            # encode the episode videos
            # print(f"[INFO]: Encoding episode to video...")
            # self.dataset.encode_episode_videos(self.dataset.num_episodes - 1)

            self.action_buffers = []
            self.observation_buffer_tensor = None
            self.rgb_buffer_tensor = None

            self.current_frame = 0
            print("[INFO]: Cleared buffers")

            print(f"[INFO]: Episode saved.")


# if __name__ == "__main__":
#     dataset = LeRobotDataset(
#                 "/lbenhorin/lerobot_so101_teleop",
#                 root="/home/lbenhorin/workspaces/github-isaac/lerobot_so101_teleop/datasets/recording_so101_teleop",
#             )
#     print(dataset)
