# Adding New Tasks

This guide explains how to create new tasks for the SO-101 robot.

This repo provide a base configuration for the robot and the environment, and new tasks can be created by inheriting from this base configuration.

## File Structure for Tasks

All tasks are defined in `source/lerobot_so101_teleop/lerobot_so101_teleop/tasks/manager_based/`.

The structure is as follows:

```
manager_based/
├── base/
│   └── base_env_cfg.py       # <-- Base configuration. Do not modify.
├── rock_a_stack/
│   └── rock_a_stack_env_cfg.py # <-- Example task: Rock a Stack
└── <your_new_task>/
    └── <your_new_task>_env_cfg.py # <-- Your new task will be defined here.
```

-   `base_env_cfg.py`: Contains the common settings for the robot and its environment. This file should not be modified.
-   `rock_a_stack_env_cfg.py`: This is an example of a task-specific configuration. It inherits settings from the base and adds the specific objects and logic for the "Rock a Stack" task.

## Creating a New Task (e.g., "Table Cleanup")

To create a new task such as "Table Cleanup," follow these steps:

1.  **Create a New Directory:**
    Inside the `manager_based` directory, create a new directory for your task (e.g., `table_cleanup`).

2.  **Create a Configuration File:**
    Inside your new `table_cleanup` directory, create a Python file named `table_cleanup_env_cfg.py`. In this file, you will define what is unique to your task, such as new objects in the scene and the task's goal. Your new task configuration will automatically inherit the base robot and room settings from `base_env_cfg.py`.

3.  **Register the Environment:**
    Create an `__init__.py` file in your task's directory. In this file, you must register the new environment with Gymnasium. look at how its done in the base task.

