# Lerobot SO-101 Teleop Isaac Lab environment


[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)


![](./banner.jpg)


[](https://github.com/user-attachments/assets/39d67d5d-b2f9-4184-9aa8-4c874476afd0)


[example.webm](https://github.com/user-attachments/assets/f66d9502-2cda-4ac9-b0ff-f750d6b19e2d)

Sample Environment for the LeRobot SO-101 Robot in Isaac Lab to collect demonstrations in a simulation. This can be utilised to later procedurally scale up datasets using various methods of domain randomization and style transfer techniques ✨ 🤖




## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
 Use Conda to be synced with the Lerobot installation guide.

- In your isaac lab conda environment, Install Lerobot
    ```
    pip install 'lerobot[feetech]==0.3.3'
    ```
- Make sure your Lerobot _Leader_ arm has been [calibrated](https://huggingface.co/docs/lerobot/en/so101#calibrate).

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    ```bash
    git clone https://gitlab-master.nvidia.com/lbenhorin/lerobot_so101_teleop.git
    git lfs pull # assets
    ```

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # Make sure your isaac lab + lerbot conda env is activated
    python -m pip install -e source/lerobot_so101_teleop
    ```

- Verify that the extension is correctly installed by listing the available tasks:

    ```bash
    python scripts/list_envs.py
    ```
    You should see a single environment called `Lerobot-So101-Teleop-v0`
- Edit `scripts/lerobot_agent.py` by changing `lerobot_cfg` to match your port and id of your setup. (You have this information from the calibration step)

- Run the environment

    ```bash
    python scripts/lerobot_agent.py --task Lerobot-So101-Teleop-v0
    ```
- Get familiar with the teleop feeling

## Record a dataset

_Currently only local recording without pushing to HuggingFace hub_

- Run the environment and include dataset repo-id and repo-root (They will get created if not already exists)

    ```bash
    python scripts/lerobot_agent.py --task Lerobot-So101-Teleop-v0 \
    --repo_id lerobot_so101_teleop \
    --repo_root $(pwd)/datasets/lerobot_so101_teleop \
    --task_name "Pick up the blue ring and put it on the pole"
    ```
- Click `S` to start/stop the recording. Reset the environment `R` will also stop the recording

- After each episode there will be a short processing time, during that the simulator will be paused and a progress bar will show in the console.

- When done, exit the simulation with `Ctrl+C`

- To playback dataset episodes, use lerobot rerun visualizer
    ```bash
    python -m lerobot.scripts.visualize_dataset \
    --repo-id lerobot_so101_teleop \
    --root $(pwd)/datasets/lerobot_so101_teleop --episode-index 0
    ```

## Contributors

- Thank you [LycheeAI](https://lycheeai-hub.com/) for making the SO101 arm available in USD format  💚 [https://github.com/MuammerBay/so-arm101-ros2-bridge/tree/main/IsaacSim_USD](https://github.com/MuammerBay/so-arm101-ros2-bridge/tree/main/IsaacSim_USD)

- Contributions are welcome via pull requests

## Limitations

- Simulated teleoperation can be hard without VR headset.


