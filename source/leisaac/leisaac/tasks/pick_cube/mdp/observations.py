import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.utils.math import quat_apply

def object_grasped(
        env: ManagerBasedRLEnv | DirectRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        diff_threshold: float = 0.05,
        grasp_threshold: float = 0.01) -> torch.Tensor:
    """Check if an object(orange) is grasped by the specified robot."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_entity: RigidObject = env.scene[object_cfg.name]

    object_pos = object_entity.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 1, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 1, :]
    offset_local = torch.tensor([0.0, 0.0, 0.16], device=env.device).repeat(env.num_envs, 1)
    offset_world = quat_apply(ee_quat_w, offset_local)  # 0.16 is the offset between J2_6 and gripper_center
    grasp_center_pos = ee_pos_w + offset_world

    pos_diff = torch.linalg.vector_norm(object_pos - grasp_center_pos, dim=1)

    joint_ids, _ = robot.find_joints("J2_8")
    # print(pos_diff)
    grasped = torch.logical_and(pos_diff < diff_threshold, robot.data.joint_pos[:, joint_ids[0]] > grasp_threshold)

    return grasped


def put_cube_to_plate(
        env: ManagerBasedRLEnv | DirectRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
        plate_cfg: SceneEntityCfg = SceneEntityCfg("Plate"),
        x_range: tuple[float, float] = (-0.10, 0.10),
        y_range: tuple[float, float] = (-0.10, 0.10),
        diff_threshold: float = 0.05,
        grasp_threshold: float = 0.01,
) -> torch.Tensor:
    """Check if an object(orange) is placed on the specified plate."""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object_entity: RigidObject = env.scene[object_cfg.name]
    plate: RigidObject = env.scene[plate_cfg.name]

    plate_x, plate_y = plate.data.root_pos_w[:, 0], plate.data.root_pos_w[:, 1]
    object_x, object_y = object_entity.data.root_pos_w[:, 0], object_entity.data.root_pos_w[:, 1]
    object_in_plate_x = torch.logical_and(object_x < plate_x + x_range[1], object_x > plate_x + x_range[0])
    object_in_plate_y = torch.logical_and(object_y < plate_y + y_range[1], object_y > plate_y + y_range[0])
    object_in_plate = torch.logical_and(object_in_plate_x, object_in_plate_y)

    ee_pos_w = ee_frame.data.target_pos_w[:, 1, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 1, :]
    offset_local = torch.tensor([0.0, 0.0, 0.16], device=env.device).repeat(env.num_envs, 1)
    offset_world = quat_apply(ee_quat_w, offset_local)
    grasp_center_pos = ee_pos_w + offset_world
    object_pos = object_entity.data.root_pos_w
    pos_diff = torch.linalg.vector_norm(object_pos - grasp_center_pos, dim=1)
    ee_near_to_object = pos_diff < diff_threshold

    joint_ids, _ = robot.find_joints("J2_8")
    gripper_joint_id = joint_ids[0]
    gripper_open = robot.data.joint_pos[:, gripper_joint_id] < grasp_threshold

    placed = torch.logical_and(object_in_plate, ee_near_to_object)
    placed = torch.logical_and(placed, gripper_open)

    return placed