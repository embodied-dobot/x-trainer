import gymnasium as gym
from .xtrainer_pickup_recognition_env_cfg import Task2EnvCfg

gym.register(
    id="task2",  # 新任务ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": Task2EnvCfg,
    },
)
