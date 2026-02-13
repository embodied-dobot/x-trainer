import torch

from dataclasses import MISSING
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from leisaac.assets.robots.xtrainer import XTRAINER_FOLLOWER_CFG
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
import numpy as np

from . import mdp

def euler_to_quat(roll_deg, pitch_deg, yaw_deg):
    r = torch.tensor([np.radians(roll_deg)], dtype=torch.float32)
    p = torch.tensor([np.radians(pitch_deg)], dtype=torch.float32)
    y = torch.tensor([np.radians(yaw_deg)], dtype=torch.float32)
    
    quat = math_utils.quat_from_euler_xyz(r, p, y)
    
    return tuple(quat[0].tolist())

@configclass
class XTrainerArmTaskSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the bi arm task."""

    scene: AssetBaseCfg = MISSING

    robot: ArticulationCfg = XTRAINER_FOLLOWER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    left_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6/left_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.065, 0.03), 
                                        rot=euler_to_quat(-15.0, 0.0, 0.0),
                                        convention="ros"),  # wxyz --> x -10°
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # HFOV 69°
            focal_length=26.8,
            horizontal_aperture=36.83,
            focus_distance=400.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1/30.0, # 30FPS
    )

    right_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6/right_wrist_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.065, 0.03), 
                                        rot=euler_to_quat(-15.0, 0.0, 0.0),
                                        convention="ros"),  # wxyz --> x -10°
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # HFOV 69°
            focal_length=26.8,
            horizontal_aperture=36.83,
            focus_distance=400.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1/30.0, # 30FPS
    )

    top: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/base_link/top_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.53, -0.55, 1.0), 
                                        # rot=(0.276, -0.961, 0.0, 0.0), 
                                        rot=euler_to_quat(-148.0, 0.0, 0.0),
                                        convention="ros"),  # wxyz  x  -148°
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # HFOV 69°
            focal_length=26.8,
            horizontal_aperture=36.83,
            focus_distance=400.0,
            clipping_range=(0.01, 50.0),
            lock_camera=True
        ),
        width=640,
        height=480,
        update_period=1/30.0, # 30FPS
    )

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

@configclass
class XTrainerArmActionsCfg:
    """Configuration for the actions."""
    left_arm_action: mdp.ActionTermCfg = MISSING
    left_gripper_action: mdp.ActionTermCfg = MISSING
    right_arm_action: mdp.ActionTermCfg = MISSING
    right_gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class XTrainerArmEventCfg:
    """Configuration for the events."""

    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class XTrainerArmObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # left_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J1_.*"])})
        # left_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J1_.*"])})
        left_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J1_.*"])})
        left_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J1_.*"])})
        left_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J1_.*"])})

        # right_joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J2_.*"])})
        # right_joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J2_.*"])})
        right_joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J2_.*"])})
        right_joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J2_.*"])})
        right_joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J2_.*"])})

        actions = ObsTerm(func=mdp.last_action)
        left_wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("left_wrist"), "data_type": "rgb", "normalize": False})
        right_wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("right_wrist"), "data_type": "rgb", "normalize": False})
        top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("top"), "data_type": "rgb", "normalize": False})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class XTrainerArmRewardsCfg:
    """Configuration for the rewards"""


@configclass
class XTrainerArmTerminationsCfg:
    """Configuration for the termination"""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class XTrainerArmTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the X-Trainer arm task template environment."""

    scene: XTrainerArmTaskSceneCfg = MISSING

    observations: XTrainerArmObservationsCfg = MISSING
    actions: XTrainerArmActionsCfg = XTrainerArmActionsCfg()
    events: XTrainerArmEventCfg = XTrainerArmEventCfg()

    rewards: XTrainerArmRewardsCfg = XTrainerArmRewardsCfg()
    terminations: XTrainerArmTerminationsCfg = MISSING

    recorders: RecordTerm = RecordTerm()

    dynamic_reset_gripper_effort_limit: bool = True
    """Whether to dynamically reset the gripper effort limit."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 2
        self.episode_length_s = 8.0
        self.viewer.eye = (0.5, -2.6, 1.8)
        self.viewer.lookat = (0.6, 11.8, -8.1)

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

    def use_teleop_device(self, teleop_device) -> None:
        self.task_type = teleop_device
        self.actions = init_action_cfg(self.actions, device=teleop_device)

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)
