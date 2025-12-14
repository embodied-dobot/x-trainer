import gymnasium as gym

gym.register(
    id='LeIsaac-XTrainer-PickCube-v0',
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.xtrainer_pick_cube_env_cfg:PickCubeEnvCfg",
    },
)