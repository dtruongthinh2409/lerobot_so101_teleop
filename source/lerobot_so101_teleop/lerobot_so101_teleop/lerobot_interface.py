import torch

from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig


class LeRobotSO101Interface:

    # USD Robot has joint ranges that are not ranging from -100 to 100
    SO101_USD_MAPPING = {
        "shoulder_pan": {"joint_min": -110, "joint_max": 110},
        "shoulder_lift": {"joint_min": -100, "joint_max": 100},
        "elbow_flex": {"joint_min": -100, "joint_max": 90},
        "wrist_flex": {"joint_min": -95, "joint_max": 95},
        "wrist_roll": {"joint_min": -160, "joint_max": 160},
        "gripper": {"joint_min": -10, "joint_max": 100},
    }

    # Joint order is the order of the joints in the USD articulation
    SO101_JOINT_ORDER = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]

    def __init__(self, cfg: dict):
        config = SO101LeaderConfig(port=cfg["port"], id=cfg["id"])
        self.teleop_dev = SO101Leader(config)
        self.teleop_dev.connect()

        self.joint_names = [joint.split(".")[0] for joint in self.SO101_JOINT_ORDER]
        self.joint_mins = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_min"] for name in self.joint_names],
            dtype=torch.float32,
        )
        self.joint_maxs = torch.tensor(
            [self.SO101_USD_MAPPING[name]["joint_max"] for name in self.joint_names],
            dtype=torch.float32,
        )

        print(f"[INFO]: Connected to the Arm at {cfg['port']} with id {cfg['id']}")

    def get_mapped_actions_vectorized(self, real_action):
        # Extract raw values
        raw_values = torch.tensor(
            [real_action[joint] for joint in self.SO101_JOINT_ORDER],
            dtype=torch.float32,
        )

        # Normalize: gripper (0-100) vs others (-100-100)
        # Gripper is always the last element
        normalized = torch.zeros_like(raw_values)
        normalized[:-1] = (
            raw_values[:-1] + 100
        ) / 200.0  # first 5 joints: -100-100 -> 0-1
        normalized[-1] = raw_values[-1] / 100.0  # gripper: 0-100 -> 0-1

        # Map to joint ranges (degrees)
        mapped_deg = self.joint_mins + normalized * (self.joint_maxs - self.joint_mins)

        # Convert to radians
        return mapped_deg * torch.pi / 180
