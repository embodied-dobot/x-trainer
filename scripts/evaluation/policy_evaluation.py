"""Script to run a leisaac evaluation with scoring for competition tasks."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac evaluation with scoring for competition tasks.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument("--eval_rounds", type=int, default=0, help="Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual reset.")
parser.add_argument("--policy_type", type=str, default="gr00tn1.5", help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi, xtrainer_act.")
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument("--enable_visualization", action="store_true", help="Enable task1 detection visualization (green/red boxes and detection points). Default: False.")
# 评分相关参数
parser.add_argument("--task1_time_limit_minutes", type=float, default=10.0, help="Time limit for task1 in minutes (default: 10 minutes).")
parser.add_argument("--task2_time_limit_minutes", type=float, default=10.0, help="Time limit for task2 in minutes (default: 10 minutes).")
parser.add_argument("--task3_time_limit_minutes", type=float, default=10.0, help="Time limit for task3 in minutes (default: 10 minutes).")

"""
python scripts/evaluation/policy_evaluation.py \
    --task=task1 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Sort objects into correct bins" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="./checkpoints/last/pretrained_model" \
    --task1_time_limit_minutes=10.0
"""

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import time
import torch
import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import SceneEntityCfg

import leisaac  # noqa: F401
from leisaac.utils.env_utils import get_task_type, dynamic_reset_gripper_effort_limit_sim

import carb
import omni

import leisaac.tasks


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        """Release the keyboard interface."""
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def preprocess_obs_dict(obs_dict: dict, model_type: str, language_instruction: str):
    """Preprocess the observation dictionary to the format expected by the policy."""
    if model_type in ["gr00tn1.5", "lerobot", "openpi", "xtrainer_act"]:
        obs_dict["task_description"] = language_instruction
        return obs_dict
    else:
        raise ValueError(f"Model type {model_type} not supported")


def evaluate_task1(env, episode_start_time, time_limit_minutes):
    """
    评估task1的得分
    
    评分标准：
    - 完成性得分 20分：放入n个物体获得 n/6 × 20分
    - 成功率得分 60分：正确分类放置n个获得 n/6 × 60分
    - 工作效率得分 20分：在规定时间T内完成得20分，超时1分钟扣5分，最大超时4分钟
    
    Args:
        env: 环境实例
        episode_start_time: 回合开始时间（秒）
        time_limit_minutes: 规定时间限制（分钟）
    
    Returns:
        dict: 包含各项得分和总分的字典
    """
    from leisaac.tasks.task1.mdp.observations import object_in_container
    
    # Task1的物体和容器配置
    good_objects = [
        SceneEntityCfg("Nova_J1_good_split"),
        SceneEntityCfg("Nova_J4_good_split"),
        SceneEntityCfg("Nova_J5_good_split"),
    ]
    bad_objects = [
        SceneEntityCfg("Nova_J1_bigbad_split"),
        SceneEntityCfg("Nova_J4_bigbad_split"),
        SceneEntityCfg("Nova_J5_bigbad_split"),
    ]
    klt_good = SceneEntityCfg("KLT_good")
    klt_bad = SceneEntityCfg("KLT_bad")
    
    # 物体偏移值（从terminations配置中获取）
    object_offsets = {
        "Nova_J1_good_split": (0.12309, 0.17763, -0.13233),
        "Nova_J4_good_split": (-0.19401, -0.41001, 0.23779),
        "Nova_J5_good_split": (-0.11345, -0.00266, -0.29996),
        "Nova_J1_bigbad_split": (-0.05194, 0.5316, -0.01087),
        "Nova_J4_bigbad_split": (0.1216, 0.10313, -0.04597),
        "Nova_J5_bigbad_split": (0.35327, 1.2108, -0.06794),
    }
    
    x_range = (-0.12, 0.12)
    y_range = (-0.155, 0.155)
    z_range = (-0.10, 0.10)
    
    # 检查每个物体是否在容器中
    placed_count = 0  # 放入任何容器的物体数量
    correct_count = 0  # 正确分类放置的物体数量
    
    # 检查good物体
    for obj_cfg in good_objects:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        # 检查是否在正确的容器（KLT_good）中
        in_good_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_good,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=False,
        )
        
        # 检查是否在错误的容器（KLT_bad）中
        in_bad_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_bad,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=False,
        )
        
        if in_good_klt[0].item():
            placed_count += 1
            correct_count += 1
        elif in_bad_klt[0].item():
            placed_count += 1  # 放入容器了，但不正确
    
    # 检查bad物体
    for obj_cfg in bad_objects:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        # 检查是否在正确的容器（KLT_bad）中
        in_bad_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_bad,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=False,
        )
        
        # 检查是否在错误的容器（KLT_good）中
        in_good_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_good,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=False,
        )
        
        if in_bad_klt[0].item():
            placed_count += 1
            correct_count += 1
        elif in_good_klt[0].item():
            placed_count += 1  # 放入容器了，但不正确
    
    # 计算完成性得分（20分）
    completion_score = (placed_count / 6.0) * 20.0
    
    # 计算成功率得分（60分）
    accuracy_score = (correct_count / 6.0) * 60.0
    
    # 计算工作效率得分（20分）
    # 注意：只有完成所有6个物体才能获得工作效率得分
    elapsed_time_minutes = (time.time() - episode_start_time) / 60.0
    
    if placed_count < 6:
        # 如果没完成所有6个物体，工作效率得分为0
        efficiency_score = 0.0
    elif elapsed_time_minutes <= time_limit_minutes:
        # 在规定时间内完成所有6个物体，得20分
        efficiency_score = 20.0
    else:
        # 超时了，每超时1分钟扣5分，最大超时4分钟
        overtime_minutes = elapsed_time_minutes - time_limit_minutes
        if overtime_minutes >= 4.0:
            efficiency_score = 0.0
        else:
            # 每超时1分钟扣5分
            efficiency_score = max(0.0, 20.0 - overtime_minutes * 5.0)
    
    # 总分
    total_score = completion_score + accuracy_score + efficiency_score
    
    return {
        "completion_score": completion_score,
        "accuracy_score": accuracy_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "placed_count": placed_count,
        "correct_count": correct_count,
        "elapsed_time_minutes": elapsed_time_minutes,
    }


def evaluate_task2(env, episode_start_time, time_limit_minutes):
    """
    评估task2的得分
    
    评分标准：
    - 收集完整性得分 40分：三个物件置于绿色判定区（固定矩形区域），得分 = 40 × s/3
    - 倾倒准确性得分 40分：三个物件同时置于绿色和蓝色判定区交集内，得分 = 40 × n/3
    - 工作效率得分 20分：在规定时间T内完成（交集内）任务得满分20分，超时1分钟扣5分，最大超时4分钟
    
    Args:
        env: 环境实例
        episode_start_time: 回合开始时间（秒）
        time_limit_minutes: 规定时间限制（分钟）
    
    Returns:
        dict: 包含各项得分和总分的字典
    """
    from isaaclab.utils.math import quat_apply
    
    # Task2的物体配置
    objects = [
        SceneEntityCfg("Emergency_Stop_Button"),
        SceneEntityCfg("factory_nut_loose"),  # nut
        SceneEntityCfg("chip"),
    ]
    protectlid = SceneEntityCfg("protectlid")
    
    # 物体偏移值（从terminations配置中获取）
    object_offsets = {
        "Emergency_Stop_Button": (0, 0.05166, 0.31674),
        "factory_nut_loose": (0.0, 0.0, 0.02),
        "chip": (0.0, 0.0, 0.0),
    }
    
    # 绿色判定区（固定矩形区域）
    rect_position = (0.689, -0.262, 0.113)
    rect_size = (0.144, 0.144, 0.050)
    
    # 蓝色判定区（protectlid下方区域）大小
    protectlid_zone_size = (0.18, 0.24, 0.040)
    
    # 计算矩形区域的边界
    rect_x, rect_y, rect_z = rect_position
    rect_size_x, rect_size_y, rect_size_z = rect_size
    rect_x_min = rect_x - rect_size_x / 2.0
    rect_x_max = rect_x + rect_size_x / 2.0
    rect_y_min = rect_y - rect_size_y / 2.0
    rect_y_max = rect_y + rect_size_y / 2.0
    rect_z_min = rect_z - rect_size_z / 2.0
    rect_z_max = rect_z + rect_size_z / 2.0
    
    # 获取protectlid的位置
    protectlid_entity: RigidObject = env.scene[protectlid.name]
    protectlid_pos = protectlid_entity.data.root_pos_w  # (num_envs, 3)
    
    # 检查每个物体
    collected_count = 0  # 在绿色判定区内的物体数量
    intersection_count = 0  # 在交集内的物体数量
    
    for obj_cfg in objects:
        obj_name = obj_cfg.name
        obj_entity: RigidObject = env.scene[obj_cfg.name]
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        # 获取物体位置和旋转，应用偏移补偿
        object_pos = obj_entity.data.root_pos_w.clone()
        object_quat = obj_entity.data.root_quat_w.clone()
        
        # 将偏移值从局部坐标系转换到世界坐标系（考虑物体旋转）
        if offset != (0.0, 0.0, 0.0):
            offset_local = torch.tensor(offset, device=env.device, dtype=object_pos.dtype)
            offset_world = quat_apply(object_quat, offset_local.unsqueeze(0).repeat(env.num_envs, 1))
            detection_point_pos = object_pos + offset_world
        else:
            detection_point_pos = object_pos
        
        # 检查判定点是否在绿色判定区（矩形区域）内
        in_rect_x = torch.logical_and(
            detection_point_pos[:, 0] >= rect_x_min,
            detection_point_pos[:, 0] <= rect_x_max
        )
        in_rect_y = torch.logical_and(
            detection_point_pos[:, 1] >= rect_y_min,
            detection_point_pos[:, 1] <= rect_y_max
        )
        in_rect_z = torch.logical_and(
            detection_point_pos[:, 2] >= rect_z_min,
            detection_point_pos[:, 2] <= rect_z_max
        )
        in_green_zone = torch.logical_and(in_rect_x, in_rect_y)
        in_green_zone = torch.logical_and(in_green_zone, in_rect_z)
        
        # 计算蓝色判定区（protectlid下方区域）的边界
        protectlid_zone_size_x, protectlid_zone_size_y, protectlid_zone_size_z = protectlid_zone_size
        protectlid_zone_x_min = protectlid_pos[:, 0] - protectlid_zone_size_x / 2.0
        protectlid_zone_x_max = protectlid_pos[:, 0] + protectlid_zone_size_x / 2.0
        protectlid_zone_y_min = protectlid_pos[:, 1] - protectlid_zone_size_y / 2.0
        protectlid_zone_y_max = protectlid_pos[:, 1] + protectlid_zone_size_y / 2.0
        # 蓝色判定区向上调整0.02米（与terminations.py保持一致）
        protectlid_zone_z_min = protectlid_pos[:, 2] - protectlid_zone_size_z + 0.02
        protectlid_zone_z_max = protectlid_pos[:, 2] + 0.02
        
        # 检查判定点是否在蓝色判定区内
        in_protectlid_x = torch.logical_and(
            detection_point_pos[:, 0] >= protectlid_zone_x_min,
            detection_point_pos[:, 0] <= protectlid_zone_x_max
        )
        in_protectlid_y = torch.logical_and(
            detection_point_pos[:, 1] >= protectlid_zone_y_min,
            detection_point_pos[:, 1] <= protectlid_zone_y_max
        )
        in_protectlid_z = torch.logical_and(
            detection_point_pos[:, 2] >= protectlid_zone_z_min,
            detection_point_pos[:, 2] <= protectlid_zone_z_max
        )
        in_blue_zone = torch.logical_and(in_protectlid_x, in_protectlid_y)
        in_blue_zone = torch.logical_and(in_blue_zone, in_protectlid_z)
        
        # 检查是否在交集内（同时满足绿色和蓝色判定区）
        in_intersection = torch.logical_and(in_green_zone, in_blue_zone)
        
        # 统计（使用第一个环境的结果）
        env_id = 0  # 使用第一个环境
        if in_green_zone[env_id].item():
            collected_count += 1
        if in_intersection[env_id].item():
            intersection_count += 1
    
    # 计算收集完整性得分（40分）
    collection_score = (collected_count / 3.0) * 40.0
    
    # 计算倾倒准确性得分（40分）
    accuracy_score = (intersection_count / 3.0) * 40.0
    
    # 计算工作效率得分（20分）
    # 注意：只有完成所有3个物体在交集内才能获得工作效率得分
    elapsed_time_minutes = (time.time() - episode_start_time) / 60.0
    
    if intersection_count < 3:
        # 如果没完成所有3个物体在交集内，工作效率得分为0
        efficiency_score = 0.0
    elif elapsed_time_minutes <= time_limit_minutes:
        # 在规定时间内完成所有3个物体在交集内，得20分
        efficiency_score = 20.0
    else:
        # 超时了，每超时1分钟扣5分，最大超时4分钟
        overtime_minutes = elapsed_time_minutes - time_limit_minutes
        if overtime_minutes >= 4.0:
            efficiency_score = 0.0
        else:
            # 每超时1分钟扣5分
            efficiency_score = max(0.0, 20.0 - overtime_minutes * 5.0)
    
    # 总分
    total_score = collection_score + accuracy_score + efficiency_score
    
    return {
        "collection_score": collection_score,
        "accuracy_score": accuracy_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "collected_count": collected_count,
        "intersection_count": intersection_count,
        "elapsed_time_minutes": elapsed_time_minutes,
    }


def evaluate_task3(env, episode_start_time, time_limit_minutes):
    """
    评估task3的得分
    
    评分标准：
    - 试管插入得分 80分：成功将3个试管全部置于试管架孔位(绿色判定区）得满分80分。部分完成：根据成功插入的试管数量按比例给分，插入n个试管得n/3 × 80分
    - 工作效率得分 20分：在规定时间T内完成全部3个试管插入得满分20分。超时惩罚：每超时1分钟扣减5分。最大允许超时4分钟，超过4分钟则工作效率得分为0分
    
    Args:
        env: 环境实例
        episode_start_time: 回合开始时间（秒）
        time_limit_minutes: 规定时间限制（分钟）
    
    Returns:
        dict: 包含各项得分和总分的字典
    """
    from leisaac.tasks.task3.mdp.observations import object_in_container
    
    # Task3的物体配置
    tube_objects = [
        SceneEntityCfg("newTube_01"),
        SceneEntityCfg("newTube_02"),
        SceneEntityCfg("newTube_03"),
    ]
    rack = SceneEntityCfg("rack")
    
    # 物体偏移值（从terminations配置中获取）
    object_offsets = {
        "newTube_01": (-0.08114, 0.13, -0.01572),
        "newTube_02": (-0.08114, 0.13, -0.01572),
        "newTube_03": (-0.08114, 0.13, -0.01572),
    }
    
    # 判定区范围
    x_range = (-0.20, 0.20)
    y_range = (-0.02, 0.02)
    z_range = (-0.05, 0)
    
    # 检查每个试管是否在rack的绿色判定区内
    inserted_count = 0  # 成功插入的试管数量
    
    for obj_cfg in tube_objects:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        in_rack_zone = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=rack,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=False,
        )
        
        if in_rack_zone[0].item():
            inserted_count += 1
    
    # 计算试管插入得分（80分）
    insertion_score = (inserted_count / 3.0) * 80.0
    
    # 计算工作效率得分（20分）
    # 注意：只有完成所有3个试管插入才能获得工作效率得分
    elapsed_time_minutes = (time.time() - episode_start_time) / 60.0
    
    if inserted_count < 3:
        # 如果没完成所有3个试管插入，工作效率得分为0
        efficiency_score = 0.0
    elif elapsed_time_minutes <= time_limit_minutes:
        # 在规定时间内完成所有3个试管插入，得20分
        efficiency_score = 20.0
    else:
        # 超时了，每超时1分钟扣5分，最大超时4分钟
        overtime_minutes = elapsed_time_minutes - time_limit_minutes
        if overtime_minutes >= 4.0:
            efficiency_score = 0.0
        else:
            # 每超时1分钟扣5分
            efficiency_score = max(0.0, 20.0 - overtime_minutes * 5.0)
    
    # 总分
    total_score = insertion_score + efficiency_score
    
    return {
        "insertion_score": insertion_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "inserted_count": inserted_count,
        "elapsed_time_minutes": elapsed_time_minutes,
    }


def main():
    """Running lerobot evaluation with scoring for competition tasks."""

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    # task1, task2, task3都使用XTrainer机器人，需要使用xtrainerleader设备类型
    if args_cli.task in ["task1", "task2", "task3"]:
        task_type = "xtrainerleader"
    else:
        task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    # 根据命令行参数设置task1的判定可视化开关
    if hasattr(env_cfg, 'enable_visualization'):
        env_cfg.enable_visualization = args_cli.enable_visualization

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    # create policy
    model_type = args_cli.policy_type
    if args_cli.policy_type == "gr00tn1.5":
        from leisaac.policy import Gr00tServicePolicyClient
        from isaaclab.sensors import Camera

        if task_type == "so101leader":
            modality_keys = ["single_arm", "gripper"]
        else:
            raise ValueError(f"Task type {task_type} not supported when using GR00T N1.5 policy yet.")

        policy = Gr00tServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            modality_keys=modality_keys,
        )
    elif "lerobot" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient
        from isaaclab.sensors import Camera

        model_type = 'lerobot'

        policy_type = args_cli.policy_type.split("-")[1]
        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)},
            task_type=task_type,
            policy_type=policy_type,
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif "xtrainer_act" in args_cli.policy_type:
        from leisaac.policy import LeRobotServicePolicyClient
        from isaaclab.sensors import Camera

        model_type = 'xtrainer_act'

        policy = LeRobotServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            timeout_ms=args_cli.policy_timeout_ms,
            camera_infos={key: sensor.image_shape for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)},
            task_type=task_type,
            policy_type='act',
            pretrained_name_or_path=args_cli.policy_checkpoint_path,
            actions_per_chunk=args_cli.policy_action_horizon,
            device=args_cli.device,
        )
    elif args_cli.policy_type == "openpi":
        from leisaac.policy import OpenPIServicePolicyClient
        from isaaclab.sensors import Camera

        policy = OpenPIServicePolicyClient(
            host=args_cli.policy_host,
            port=args_cli.policy_port,
            camera_keys=[key for key, sensor in env.scene.sensors.items() if isinstance(sensor, Camera)],
            task_type=task_type,
        )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    episode_count = 1
    all_scores = []  # 存储所有回合的得分

    # simulate environment
    while max_episode_count <= 0 or episode_count <= max_episode_count:
        print(f"\n{'='*60}")
        print(f"[Evaluation] 开始评估回合 {episode_count}...")
        print(f"{'='*60}")
        
        episode_start_time = time.time()
        success, time_out = False, False
        
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                if controller.reset_state:
                    controller.reset()
                    obs_dict, _ = env.reset()
                    episode_count += 1
                    break

                # 每次循环都获取新的动作块
                obs_dict = preprocess_obs_dict(obs_dict['policy'], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    if reset_terminated[0]:
                        success = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
                
                # 如果在执行动作时成功或超时，跳出外层循环
                if success or time_out:
                    break
        
        # 计算得分
        episode_end_time = time.time()
        scores = None
        
        if args_cli.task == "task1":
            scores = evaluate_task1(env, episode_start_time, args_cli.task1_time_limit_minutes)
            all_scores.append(scores)
            
            print(f"\n[评分] 回合 {episode_count} 得分详情:")
            print(f"  - 完成性得分: {scores['completion_score']:.2f} / 20.0 (放入 {scores['placed_count']}/6 个物体)")
            print(f"  - 成功率得分: {scores['accuracy_score']:.2f} / 60.0 (正确分类 {scores['correct_count']}/6 个物体)")
            print(f"  - 工作效率得分: {scores['efficiency_score']:.2f} / 20.0 (用时 {scores['elapsed_time_minutes']:.2f} 分钟, 限制 {args_cli.task1_time_limit_minutes} 分钟)")
            print(f"  - 总分: {scores['total_score']:.2f} / 100.0")
        elif args_cli.task == "task2":
            scores = evaluate_task2(env, episode_start_time, args_cli.task2_time_limit_minutes)
            all_scores.append(scores)
            
            print(f"\n[评分] 回合 {episode_count} 得分详情:")
            print(f"  - 收集完整性得分: {scores['collection_score']:.2f} / 40.0 (收集 {scores['collected_count']}/3 个物体到绿色判定区)")
            print(f"  - 倾倒准确性得分: {scores['accuracy_score']:.2f} / 40.0 (放置 {scores['intersection_count']}/3 个物体到交集区域)")
            print(f"  - 工作效率得分: {scores['efficiency_score']:.2f} / 20.0 (用时 {scores['elapsed_time_minutes']:.2f} 分钟, 限制 {args_cli.task2_time_limit_minutes} 分钟)")
            print(f"  - 总分: {scores['total_score']:.2f} / 100.0")
        elif args_cli.task == "task3":
            scores = evaluate_task3(env, episode_start_time, args_cli.task3_time_limit_minutes)
            all_scores.append(scores)
            
            print(f"\n[评分] 回合 {episode_count} 得分详情:")
            print(f"  - 试管插入得分: {scores['insertion_score']:.2f} / 80.0 (插入 {scores['inserted_count']}/3 个试管)")
            print(f"  - 工作效率得分: {scores['efficiency_score']:.2f} / 20.0 (用时 {scores['elapsed_time_minutes']:.2f} 分钟, 限制 {args_cli.task3_time_limit_minutes} 分钟)")
            print(f"  - 总分: {scores['total_score']:.2f} / 100.0")
        else:
            # 其他任务只显示成功/失败
            if success:
                print(f"[Evaluation] 回合 {episode_count} 成功完成!")
            elif time_out:
                print(f"[Evaluation] 回合 {episode_count} 超时!")
            else:
                print(f"[Evaluation] 回合 {episode_count} 未完成")
        
        episode_count += 1
        
        # 重置环境准备下一回合
        if max_episode_count <= 0 or episode_count <= max_episode_count:
            obs_dict, _ = env.reset()
            controller.reset()
    
    # 输出最终统计
    print(f"\n{'='*60}")
    print(f"[Evaluation] 评估完成!")
    print(f"{'='*60}")
    
    if args_cli.task == "task1" and len(all_scores) > 0:
        avg_completion = sum(s['completion_score'] for s in all_scores) / len(all_scores)
        avg_accuracy = sum(s['accuracy_score'] for s in all_scores) / len(all_scores)
        avg_efficiency = sum(s['efficiency_score'] for s in all_scores) / len(all_scores)
        avg_total = sum(s['total_score'] for s in all_scores) / len(all_scores)
        
        print(f"\n[最终统计] 共完成 {len(all_scores)} 个回合:")
        print(f"  - 平均完成性得分: {avg_completion:.2f} / 20.0")
        print(f"  - 平均成功率得分: {avg_accuracy:.2f} / 60.0")
        print(f"  - 平均工作效率得分: {avg_efficiency:.2f} / 20.0")
        print(f"  - 平均总分: {avg_total:.2f} / 100.0")
        print(f"  - 最高分: {max(s['total_score'] for s in all_scores):.2f} / 100.0")
        print(f"  - 最低分: {min(s['total_score'] for s in all_scores):.2f} / 100.0")
    elif args_cli.task == "task2" and len(all_scores) > 0:
        avg_collection = sum(s['collection_score'] for s in all_scores) / len(all_scores)
        avg_accuracy = sum(s['accuracy_score'] for s in all_scores) / len(all_scores)
        avg_efficiency = sum(s['efficiency_score'] for s in all_scores) / len(all_scores)
        avg_total = sum(s['total_score'] for s in all_scores) / len(all_scores)
        
        print(f"\n[最终统计] 共完成 {len(all_scores)} 个回合:")
        print(f"  - 平均收集完整性得分: {avg_collection:.2f} / 40.0")
        print(f"  - 平均倾倒准确性得分: {avg_accuracy:.2f} / 40.0")
        print(f"  - 平均工作效率得分: {avg_efficiency:.2f} / 20.0")
        print(f"  - 平均总分: {avg_total:.2f} / 100.0")
        print(f"  - 最高分: {max(s['total_score'] for s in all_scores):.2f} / 100.0")
        print(f"  - 最低分: {min(s['total_score'] for s in all_scores):.2f} / 100.0")
    elif args_cli.task == "task3" and len(all_scores) > 0:
        avg_insertion = sum(s['insertion_score'] for s in all_scores) / len(all_scores)
        avg_efficiency = sum(s['efficiency_score'] for s in all_scores) / len(all_scores)
        avg_total = sum(s['total_score'] for s in all_scores) / len(all_scores)
        
        print(f"\n[最终统计] 共完成 {len(all_scores)} 个回合:")
        print(f"  - 平均试管插入得分: {avg_insertion:.2f} / 80.0")
        print(f"  - 平均工作效率得分: {avg_efficiency:.2f} / 20.0")
        print(f"  - 平均总分: {avg_total:.2f} / 100.0")
        print(f"  - 最高分: {max(s['total_score'] for s in all_scores):.2f} / 100.0")
        print(f"  - 最低分: {min(s['total_score'] for s in all_scores):.2f} / 100.0")

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
