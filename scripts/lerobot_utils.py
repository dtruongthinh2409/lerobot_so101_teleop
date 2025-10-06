import argparse


from lerobot.datasets.lerobot_dataset import LeRobotDataset

parser = argparse.ArgumentParser(description="LeRobot Utils.")


parser.add_argument(
    "--repo-id", type=str, default=None, help="Repository ID to store the dataset."
)
parser.add_argument(
    "--root", type=str, default=None, help="Repository root to store the dataset."
)
parser.add_argument(
    "--push",
    action="store_true",
    default=False,
    help="Push the dataset to HuggingFace Hub.",
)
parser.add_argument(
    "--private", action="store_true", default=False, help="Private dataset."
)
parser.add_argument(
    "--tags",
    type=str,
    nargs="+",
    default=None,
    help="Tags to add to the dataset. Can specify multiple tags.",
)

args_cli = parser.parse_args()


def push_dataset_to_hub(
    repo_id: str, root: str, private: bool = True, tags: list[str] = None
):
    try:
        dataset = LeRobotDataset(repo_id=repo_id, root=root)
    except Exception as e:
        print(f"[ERROR]: Failed to initialize dataset: {e}")
        return

    print(f"[INFO]: Pushing dataset to HuggingFace Hub...")
    dataset.push_to_hub(private=private, tags=tags)


if __name__ == "__main__":
    if args_cli.push:
        push_dataset_to_hub(
            repo_id=args_cli.repo_id,
            root=args_cli.root,
            private=args_cli.private,
            tags=args_cli.tags,
        )
