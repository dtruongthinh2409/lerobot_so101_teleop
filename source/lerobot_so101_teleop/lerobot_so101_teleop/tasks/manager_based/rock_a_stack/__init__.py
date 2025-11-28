import gymnasium as gym


##
# Register Gym environments.
##


gym.register(
    id="Lerobot-So101-Teleop-Rock-A-Stack",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rock_a_stack_env_cfg:RockAStackEnvCfg",
    },
)

gym.register(
    id="Lerobot-So101-Teleop-Rock-A-Stack-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rock_a_stack_env_cfg:RockAStackEnvEvalCfg",
    },
)