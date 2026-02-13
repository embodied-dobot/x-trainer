import gymnasium as gym

# 导入你刚才重命名的配置类
from .xtrainer_pickup_recognition_env_cfg import PickupRecognitionEnvCfg

gym.register(
    id="task1",  # 新任务ID
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PickupRecognitionEnvCfg,
    },
)