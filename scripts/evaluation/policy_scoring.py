"""Script to run a leisaac inference with leisaac in the simulation, with scoring."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import math

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac inference for leisaac in the simulation.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument("--episode_length_s", type=float, default=60.0, help="Episode length in seconds.")
parser.add_argument("--eval_rounds", type=int, default=0, help="Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual reset.")
parser.add_argument("--score_t0_s", type=float, default=None, help="Scoring time baseline t0 in seconds (efficiency).")
parser.add_argument("--policy_type", type=str, default="gr00tn1.5", help="Type of policy to use. support gr00tn1.5, lerobot-<model_type>, openpi, xtrainer_act.")
parser.add_argument("--policy_host", type=str, default="localhost", help="Host of the policy server.")
parser.add_argument("--policy_port", type=int, default=5555, help="Port of the policy server.")
parser.add_argument("--policy_timeout_ms", type=int, default=15000, help="Timeout of the policy server.")
parser.add_argument("--policy_action_horizon", type=int, default=16, help="Action horizon of the policy.")
parser.add_argument("--policy_language_instruction", type=str, default=None, help="Language instruction of the policy.")
parser.add_argument("--policy_checkpoint_path", type=str, default=None, help="Checkpoint path of the policy.")
parser.add_argument("--enable_visualization", action="store_true", help="Enable task1 detection visualization (green/red boxes and detection points). Default: False.")

"""
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="/home/wyt/yc/lerobot/outputs/train/act_xtrainer_lift_cube/checkpoints/last/pretrained_model"
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

import leisaac  # noqa: F401
from isaaclab.utils.math import quat_apply
from leisaac.utils.env_utils import get_task_type, dynamic_reset_gripper_effort_limit_sim
from leisaac.tasks.task1.mdp.observations import object_in_container
from leisaac.tasks.task2.mdp.observations import object_in_task2_intersection

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


def cleanup_policy(policy) -> None:
    """Best-effort cleanup for policy clients."""
    if policy is None:
        return
    if hasattr(policy, "channel"):
        try:
            policy.channel.close()
        except Exception:
            pass


def create_policy(env: ManagerBasedRLEnv, task_type: str):
    """Create a new policy client for each episode to clear hidden state."""
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
            verbose=False,
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
            verbose=False,
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
    else:
        raise ValueError(f"Policy type {args_cli.policy_type} not supported")

    return policy, model_type


def get_step_dt_seconds(env: ManagerBasedRLEnv, fallback_hz: int) -> float:
    """Best-effort step dt in seconds for scoring time usage."""
    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    if hasattr(env, "physics_dt"):
        return float(env.physics_dt) * getattr(env.cfg, "decimation", 1)
    if hasattr(env, "cfg") and hasattr(env.cfg, "sim") and hasattr(env.cfg.sim, "dt"):
        return float(env.cfg.sim.dt) * getattr(env.cfg, "decimation", 1)
    return 1.0 / float(fallback_hz)


def get_task1_scoring_params(env: ManagerBasedRLEnv) -> dict | None:
    """Fetch task1 scoring params from env config if available."""
    if not hasattr(env, "cfg"):
        return None
    terminations = getattr(env.cfg, "terminations", None)
    if terminations is None or not hasattr(terminations, "success"):
        return None
    params = getattr(terminations.success, "params", None)
    if not params:
        return None
    required_keys = ["good_objects_cfg", "bad_objects_cfg", "klt_good_cfg", "klt_bad_cfg"]
    if any(key not in params for key in required_keys):
        return None
    return params


def get_task2_scoring_params(env: ManagerBasedRLEnv) -> dict | None:
    """Fetch task2 scoring params from env config if available."""
    if not hasattr(env, "cfg"):
        return None
    terminations = getattr(env.cfg, "terminations", None)
    if terminations is None or not hasattr(terminations, "success"):
        return None
    params = getattr(terminations.success, "params", None)
    if not params:
        return None
    required_keys = ["objects_cfg", "protectlid_cfg", "rect_position", "rect_size"]
    if any(key not in params for key in required_keys):
        return None
    return params


def get_task3_scoring_params(env: ManagerBasedRLEnv) -> dict | None:
    """Fetch task3 scoring params from env config if available."""
    if not hasattr(env, "cfg"):
        return None
    terminations = getattr(env.cfg, "terminations", None)
    if terminations is None or not hasattr(terminations, "success"):
        return None
    params = getattr(terminations.success, "params", None)
    if not params:
        return None
    required_keys = ["tube_objects_cfg", "rack_cfg"]
    if any(key not in params for key in required_keys):
        return None
    return params


def compute_task2_counts(env: ManagerBasedRLEnv, params: dict) -> dict:
    """Compute task2 placement counts using existing success判定逻辑."""
    objects_cfg = params["objects_cfg"]
    protectlid_cfg = params["protectlid_cfg"]
    rect_position = params["rect_position"]
    rect_size = params["rect_size"]
    object_offsets = params.get("object_offsets", {})
    protectlid_zone_size = params.get("protectlid_zone_size", (0.18, 0.24, 0.040))

    rect_x, rect_y, rect_z = rect_position
    rect_size_x, rect_size_y, rect_size_z = rect_size
    rect_x_min = rect_x - rect_size_x / 2.0
    rect_x_max = rect_x + rect_size_x / 2.0
    rect_y_min = rect_y - rect_size_y / 2.0
    rect_y_max = rect_y + rect_size_y / 2.0
    rect_z_min = rect_z - rect_size_z / 2.0
    rect_z_max = rect_z + rect_size_z / 2.0

    collection_count = 0
    intersection_count = 0

    rect_cache = getattr(env, "_task2_last_in_rect_state", None)
    intersection_cache = getattr(env, "_task2_last_in_intersection_state", None)
    use_cache = rect_cache is not None and intersection_cache is not None
    if use_cache:
        required_keys = [obj_cfg.name for obj_cfg in objects_cfg]
        use_cache = all(key in rect_cache and key in intersection_cache for key in required_keys)
    if not use_cache:
        rect_cache = getattr(env, "_task2_in_rect_state", None)
        intersection_cache = getattr(env, "_task2_in_intersection_state", None)
        use_cache = rect_cache is not None and intersection_cache is not None
        if use_cache:
            required_keys = [obj_cfg.name for obj_cfg in objects_cfg]
            use_cache = all(key in rect_cache and key in intersection_cache for key in required_keys)

    for obj_cfg in objects_cfg:
        obj_name = obj_cfg.name
        obj_entity = env.scene[obj_cfg.name]
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))

        if use_cache:
            in_rect = rect_cache[obj_name]
            in_intersection = intersection_cache[obj_name]
        else:
            object_pos = obj_entity.data.root_pos_w.clone()
            object_quat = obj_entity.data.root_quat_w.clone()
            if offset != (0.0, 0.0, 0.0):
                offset_local = torch.tensor(offset, device=env.device, dtype=object_pos.dtype)
                offset_world = quat_apply(object_quat, offset_local.unsqueeze(0).repeat(env.num_envs, 1))
                detection_point_pos = object_pos + offset_world
            else:
                detection_point_pos = object_pos

            in_rect_x = torch.logical_and(
                detection_point_pos[:, 0] >= rect_x_min,
                detection_point_pos[:, 0] <= rect_x_max,
            )
            in_rect_y = torch.logical_and(
                detection_point_pos[:, 1] >= rect_y_min,
                detection_point_pos[:, 1] <= rect_y_max,
            )
            in_rect_z = torch.logical_and(
                detection_point_pos[:, 2] >= rect_z_min,
                detection_point_pos[:, 2] <= rect_z_max,
            )
            in_rect = torch.logical_and(in_rect_x, in_rect_y)
            in_rect = torch.logical_and(in_rect, in_rect_z)

            in_intersection = object_in_task2_intersection(
                env=env,
                object_cfg=obj_cfg,
                protectlid_cfg=protectlid_cfg,
                rect_position=rect_position,
                rect_size=rect_size,
                protectlid_zone_size=protectlid_zone_size,
                object_offset=offset,
                verbose=False,
            )

        collection_count += int(in_rect[0].item())
        intersection_count += int(in_intersection[0].item())

    total_objects = len(objects_cfg)
    return {
        "collection_count": collection_count,
        "intersection_count": intersection_count,
        "total_objects": total_objects,
    }


def compute_task3_counts(env: ManagerBasedRLEnv, params: dict) -> dict:
    """Compute task3 placement counts using existing success判定逻辑."""
    tube_objects_cfg = params["tube_objects_cfg"]
    rack_cfg = params.get("rack_cfg")
    x_range = params.get("x_range", (-0.20, 0.20))
    y_range = params.get("y_range", (-0.02, 0.02))
    z_range = params.get("z_range", (0.10, 0.15))
    object_offsets = params.get("object_offsets", {})

    state_cache = getattr(env, "_task3_last_in_rack_state", None)
    use_cache = state_cache is not None
    if use_cache:
        required_keys = [f"{obj_cfg.name}_{rack_cfg.name}" for obj_cfg in tube_objects_cfg]
        use_cache = all(key in state_cache for key in required_keys)
    if not use_cache:
        state_cache = getattr(env, "_object_in_container_state", None)
        use_cache = state_cache is not None
        if use_cache:
            required_keys = [f"{obj_cfg.name}_{rack_cfg.name}" for obj_cfg in tube_objects_cfg]
            use_cache = all(key in state_cache for key in required_keys)

    inserted_count = 0
    for obj_cfg in tube_objects_cfg:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        if use_cache:
            in_rack = state_cache[f"{obj_name}_{rack_cfg.name}"]
        else:
            in_rack = object_in_container(
                env=env,
                object_cfg=obj_cfg,
                container_cfg=rack_cfg,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                object_offset=offset,
                verbose=False,
            )
        inserted_count += int(in_rack[0].item())

    total_objects = len(tube_objects_cfg)
    return {
        "inserted_count": inserted_count,
        "total_objects": total_objects,
    }


def compute_task2_scores(
    collection_count: int,
    intersection_count: int,
    total_objects: int,
    elapsed_s: float,
    t0_s: float,
) -> dict:
    """Compute task2 score components and total score."""
    if total_objects <= 0:
        return {
            "collection_score": 0.0,
            "accuracy_score": 0.0,
            "efficiency_score": 0.0,
            "total_score": 0.0,
            "overtime_minutes": None,
        }

    collection_score = (collection_count / total_objects) * 40.0
    accuracy_score = (intersection_count / total_objects) * 40.0

    efficiency_score = 0.0
    if intersection_count == total_objects and t0_s > 0.0:
        efficiency_score = max((t0_s - elapsed_s) / t0_s * 20.0, 0.0)

    total_score = collection_score + accuracy_score + efficiency_score
    return {
        "collection_score": collection_score,
        "accuracy_score": accuracy_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "overtime_minutes": None,
    }


def compute_task3_scores(
    inserted_count: int,
    total_objects: int,
    elapsed_s: float,
    t0_s: float,
) -> dict:
    """Compute task3 score components and total score."""
    if total_objects <= 0:
        return {
            "insert_score": 0.0,
            "efficiency_score": 0.0,
            "total_score": 0.0,
            "overtime_minutes": None,
        }

    insert_score = (inserted_count / total_objects) * 80.0
    efficiency_score = 0.0
    if inserted_count == total_objects and t0_s > 0.0:
        efficiency_score = max((t0_s - elapsed_s) / t0_s * 20.0, 0.0)

    total_score = insert_score + efficiency_score
    return {
        "insert_score": insert_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "overtime_minutes": None,
    }


def compute_task1_counts(env: ManagerBasedRLEnv, params: dict) -> dict:
    """Compute task1 placement counts using existing success判定逻辑."""
    good_objects_cfg = params["good_objects_cfg"]
    bad_objects_cfg = params["bad_objects_cfg"]
    klt_good_cfg = params.get("klt_good_cfg")
    klt_bad_cfg = params.get("klt_bad_cfg")
    x_range = params.get("x_range", (-0.12, 0.12))
    y_range = params.get("y_range", (-0.155, 0.155))
    z_range = params.get("z_range", (-0.10, 0.10))
    object_offsets = params.get("object_offsets", {})

    state_cache = getattr(env, "_task1_last_in_container_state", None)
    use_cache = state_cache is not None
    if use_cache:
        required_keys = []
        for obj_cfg in good_objects_cfg + bad_objects_cfg:
            required_keys.append(f"{obj_cfg.name}_KLT_good")
            required_keys.append(f"{obj_cfg.name}_KLT_bad")
        use_cache = all(key in state_cache for key in required_keys)
    if not use_cache:
        state_cache = getattr(env, "_object_in_container_state", None)
        use_cache = state_cache is not None
        if use_cache:
            required_keys = []
            for obj_cfg in good_objects_cfg + bad_objects_cfg:
                required_keys.append(f"{obj_cfg.name}_{klt_good_cfg.name}")
                required_keys.append(f"{obj_cfg.name}_{klt_bad_cfg.name}")
            use_cache = all(key in state_cache for key in required_keys)

    completion_count = 0
    correct_count = 0
    for obj_cfg in good_objects_cfg:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        if use_cache:
            if f"{obj_name}_KLT_good" in state_cache:
                in_good = state_cache[f"{obj_name}_KLT_good"]
                in_bad = state_cache[f"{obj_name}_KLT_bad"]
            else:
                in_good = state_cache[f"{obj_name}_{klt_good_cfg.name}"]
                in_bad = state_cache[f"{obj_name}_{klt_bad_cfg.name}"]
        else:
            in_good = object_in_container(
                env=env,
                object_cfg=obj_cfg,
                container_cfg=klt_good_cfg,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                object_offset=offset,
                verbose=False,
            )
            in_bad = object_in_container(
                env=env,
                object_cfg=obj_cfg,
                container_cfg=klt_bad_cfg,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                object_offset=offset,
                verbose=False,
            )
        completion_count += int(torch.logical_or(in_good, in_bad)[0].item())
        correct_count += int(in_good[0].item())

    for obj_cfg in bad_objects_cfg:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        if use_cache:
            if f"{obj_name}_KLT_good" in state_cache:
                in_good = state_cache[f"{obj_name}_KLT_good"]
                in_bad = state_cache[f"{obj_name}_KLT_bad"]
            else:
                in_good = state_cache[f"{obj_name}_{klt_good_cfg.name}"]
                in_bad = state_cache[f"{obj_name}_{klt_bad_cfg.name}"]
        else:
            in_good = object_in_container(
                env=env,
                object_cfg=obj_cfg,
                container_cfg=klt_good_cfg,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                object_offset=offset,
                verbose=False,
            )
            in_bad = object_in_container(
                env=env,
                object_cfg=obj_cfg,
                container_cfg=klt_bad_cfg,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                object_offset=offset,
                verbose=False,
            )
        completion_count += int(torch.logical_or(in_good, in_bad)[0].item())
        correct_count += int(in_bad[0].item())

    total_objects = len(good_objects_cfg) + len(bad_objects_cfg)
    return {
        "completion_count": completion_count,
        "correct_count": correct_count,
        "total_objects": total_objects,
    }


def compute_task1_scores(
    completion_count: int,
    correct_count: int,
    total_objects: int,
    elapsed_s: float,
    t0_s: float,
) -> dict:
    """Compute task1 score components and total score."""
    if total_objects <= 0:
        return {
            "completion_score": 0.0,
            "success_score": 0.0,
            "efficiency_score": 0.0,
            "total_score": 0.0,
            "overtime_minutes": None,
        }

    completion_score = (completion_count / total_objects) * 20.0
    success_score = (correct_count / total_objects) * 60.0

    efficiency_score = 0.0
    if completion_count == total_objects and t0_s > 0.0:
        efficiency_score = max((t0_s - elapsed_s) / t0_s * 20.0, 0.0)

    total_score = completion_score + success_score + efficiency_score
    return {
        "completion_score": completion_score,
        "success_score": success_score,
        "efficiency_score": efficiency_score,
        "total_score": total_score,
        "overtime_minutes": None,
    }


def main():
    """Running lerobot teleoperation with leisaac manipulation environment."""

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

    # 关闭判定与观测中的verbose输出，避免刷屏
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "success"):
        if hasattr(env_cfg.terminations.success, "params") and "verbose" in env_cfg.terminations.success.params:
            env_cfg.terminations.success.params["verbose"] = False
        if hasattr(env_cfg.terminations.success, "params") and "visualize" in env_cfg.terminations.success.params:
            env_cfg.terminations.success.params["visualize"] = False
    if hasattr(env_cfg, "observations") and hasattr(env_cfg.observations, "subtask_terms"):
        for _, term in vars(env_cfg.observations.subtask_terms).items():
            if hasattr(term, "params") and "verbose" in term.params:
                term.params["verbose"] = False

    # modify configuration
    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    # create environment
    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()

    # reset environment
    obs_dict, _ = env.reset()
    controller.reset()

    # record the results
    success_count = 0
    completed_episodes = 0
    episode_scores = []
    episode_scores_task2 = []
    episode_scores_task3 = []
    step_dt_s = get_step_dt_seconds(env, args_cli.step_hz)
    task1_params = get_task1_scoring_params(env) if args_cli.task == "task1" else None
    if args_cli.task == "task1" and task1_params is None:
        print("[WARNING] 未能获取task1评分参数，后续不输出task1评分。")
    task2_params = get_task2_scoring_params(env) if args_cli.task == "task2" else None
    if args_cli.task == "task2" and task2_params is None:
        print("[WARNING] 未能获取task2评分参数，后续不输出task2评分。")
    task3_params = get_task3_scoring_params(env) if args_cli.task == "task3" else None
    if args_cli.task == "task3" and task3_params is None:
        print("[WARNING] 未能获取task3评分参数，后续不输出task3评分。")

    score_t0_s = args_cli.score_t0_s if args_cli.score_t0_s is not None else args_cli.episode_length_s

    # simulate environment
    policy = None
    model_type = args_cli.policy_type
    while max_episode_count <= 0 or completed_episodes < max_episode_count:
        cleanup_policy(policy)
        policy, model_type = create_policy(env, task_type)
        print(f"[Evaluation] Evaluating episode {completed_episodes + 1}...")
        success, time_out = False, False
        manual_reset = False
        episode_done = False
        steps_in_episode = 0
        while simulation_app.is_running():
            if controller.reset_state:
                controller.reset()
                obs_dict, _ = env.reset()
                manual_reset = True
                break

            # run everything in inference mode
            with torch.inference_mode():
                # 每次循环都获取新的动作块
                obs_dict = preprocess_obs_dict(obs_dict['policy'], model_type, args_cli.policy_language_instruction)
                actions = policy.get_action(obs_dict).to(env.device)
                
                for i in range(min(args_cli.policy_action_horizon, actions.shape[0])):
                    action = actions[i, :, :]
                    if env.cfg.dynamic_reset_gripper_effort_limit:
                        dynamic_reset_gripper_effort_limit_sim(env, task_type)
                    obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                    steps_in_episode += 1
                    if args_cli.episode_length_s > 0.0 and (steps_in_episode * step_dt_s) >= args_cli.episode_length_s:
                        time_out = True
                        episode_done = True
                        break
                    if reset_terminated[0]:
                        success = True
                        episode_done = True
                        break
                    if reset_time_outs[0]:
                        time_out = True
                        episode_done = True
                        break
                    if rate_limiter:
                        rate_limiter.sleep(env)
                
                # 如果在执行动作时成功或超时，跳出外层循环
                if success or time_out:
                    break
                    
            if success:
                print(f"[Evaluation] Episode {completed_episodes + 1} is successful!")
                break
            if time_out:
                print(f"[Evaluation] Episode {completed_episodes + 1} timed out!")
                break

        if manual_reset:
            continue
        if not episode_done and not success and not time_out:
            # Safety: if simulator stops without a terminal condition, exit loop.
            break

        completed_episodes += 1
        if success:
            success_count += 1

        if args_cli.task == "task1" and task1_params is not None:
            counts = compute_task1_counts(env, task1_params)
            elapsed_s = steps_in_episode * step_dt_s
            scores = compute_task1_scores(
                completion_count=counts["completion_count"],
                correct_count=counts["correct_count"],
                total_objects=counts["total_objects"],
                elapsed_s=elapsed_s,
                t0_s=score_t0_s,
            )
            scores.update({
                "episode": completed_episodes,
                "completion_count": counts["completion_count"],
                "correct_count": counts["correct_count"],
                "elapsed_s": elapsed_s,
                "time_limit_s": score_t0_s,
            })
            episode_scores.append(scores)
            print(
                "[Score][Task1] "
                f"Episode {scores['episode']} | "
                f"完成性 {scores['completion_score']:.2f} "
                f"({scores['completion_count']}/{counts['total_objects']}), "
                f"成功率 {scores['success_score']:.2f} "
                f"({scores['correct_count']}/{counts['total_objects']}), "
                f"效率 {scores['efficiency_score']:.2f} "
                f"(耗时 {elapsed_s:.1f}s / t0 {score_t0_s:.1f}s), "
                f"总分 {scores['total_score']:.2f}"
            )
        if args_cli.task == "task2" and task2_params is not None:
            counts = compute_task2_counts(env, task2_params)
            elapsed_s = steps_in_episode * step_dt_s
            scores = compute_task2_scores(
                collection_count=counts["collection_count"],
                intersection_count=counts["intersection_count"],
                total_objects=counts["total_objects"],
                elapsed_s=elapsed_s,
                t0_s=score_t0_s,
            )
            scores.update({
                "episode": completed_episodes,
                "collection_count": counts["collection_count"],
                "intersection_count": counts["intersection_count"],
                "elapsed_s": elapsed_s,
                "time_limit_s": score_t0_s,
            })
            episode_scores_task2.append(scores)
            print(
                "[Score][Task2] "
                f"Episode {scores['episode']} | "
                f"收集完整性 {scores['collection_score']:.2f} "
                f"({scores['collection_count']}/{counts['total_objects']}), "
                f"倾倒准确性 {scores['accuracy_score']:.2f} "
                f"({scores['intersection_count']}/{counts['total_objects']}), "
                f"效率 {scores['efficiency_score']:.2f} "
                f"(耗时 {elapsed_s:.1f}s / t0 {score_t0_s:.1f}s), "
                f"总分 {scores['total_score']:.2f}"
            )
        if args_cli.task == "task3" and task3_params is not None:
            counts = compute_task3_counts(env, task3_params)
            elapsed_s = steps_in_episode * step_dt_s
            scores = compute_task3_scores(
                inserted_count=counts["inserted_count"],
                total_objects=counts["total_objects"],
                elapsed_s=elapsed_s,
                t0_s=score_t0_s,
            )
            scores.update({
                "episode": completed_episodes,
                "inserted_count": counts["inserted_count"],
                "elapsed_s": elapsed_s,
                "time_limit_s": score_t0_s,
            })
            episode_scores_task3.append(scores)
            print(
                "[Score][Task3] "
                f"Episode {scores['episode']} | "
                f"试管插入 {scores['insert_score']:.2f} "
                f"({scores['inserted_count']}/{counts['total_objects']}), "
                f"效率 {scores['efficiency_score']:.2f} "
                f"(耗时 {elapsed_s:.1f}s / t0 {score_t0_s:.1f}s), "
                f"总分 {scores['total_score']:.2f}"
            )
        if completed_episodes > 0:
            print(f"[Evaluation] now success rate: {success_count / completed_episodes}  [{success_count}/{completed_episodes}]")
        else:
            print("[Evaluation] now success rate: 0.0  [0/0]")
    if completed_episodes > 0:
        print(f"[Evaluation] Final success rate: {success_count / completed_episodes:.3f}  [{success_count}/{completed_episodes}]")
    else:
        print("[Evaluation] Final success rate: 0.000  [0/0]")
    if args_cli.task == "task1" and episode_scores:
        avg_completion = sum(s["completion_score"] for s in episode_scores) / len(episode_scores)
        avg_success = sum(s["success_score"] for s in episode_scores) / len(episode_scores)
        avg_efficiency = sum(s["efficiency_score"] for s in episode_scores) / len(episode_scores)
        avg_total = sum(s["total_score"] for s in episode_scores) / len(episode_scores)
        print(
            "[Score][Task1] "
            f"平均分 | 完成性 {avg_completion:.2f}, "
            f"成功率 {avg_success:.2f}, "
            f"效率 {avg_efficiency:.2f}, "
            f"总分 {avg_total:.2f}"
        )
    if args_cli.task == "task2" and episode_scores_task2:
        avg_collection = sum(s["collection_score"] for s in episode_scores_task2) / len(episode_scores_task2)
        avg_accuracy = sum(s["accuracy_score"] for s in episode_scores_task2) / len(episode_scores_task2)
        avg_efficiency = sum(s["efficiency_score"] for s in episode_scores_task2) / len(episode_scores_task2)
        avg_total = sum(s["total_score"] for s in episode_scores_task2) / len(episode_scores_task2)
        print(
            "[Score][Task2] "
            f"平均分 | 收集完整性 {avg_collection:.2f}, "
            f"倾倒准确性 {avg_accuracy:.2f}, "
            f"效率 {avg_efficiency:.2f}, "
            f"总分 {avg_total:.2f}"
        )
    if args_cli.task == "task3" and episode_scores_task3:
        avg_insert = sum(s["insert_score"] for s in episode_scores_task3) / len(episode_scores_task3)
        avg_efficiency = sum(s["efficiency_score"] for s in episode_scores_task3) / len(episode_scores_task3)
        avg_total = sum(s["total_score"] for s in episode_scores_task3) / len(episode_scores_task3)
        print(
            "[Score][Task3] "
            f"平均分 | 试管插入 {avg_insert:.2f}, "
            f"效率 {avg_efficiency:.2f}, "
            f"总分 {avg_total:.2f}"
        )

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
