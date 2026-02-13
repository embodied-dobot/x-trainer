from __future__ import annotations

import torch
from typing import List, Dict, Optional

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.utils.math import quat_apply

from leisaac.utils.robot_utils import is_xtrainer_at_rest_pose

# 导入可视化函数
from .observations import (
    create_detection_zone_visualization, 
    create_object_detection_point_visualization,
    update_object_detection_point_visualization
)


def task_done(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: List[SceneEntityCfg],
    plate_cfg: SceneEntityCfg,
    x_range: tuple[float, float] = (-0.10, 0.10),
    y_range: tuple[float, float] = (-0.10, 0.10),
    height_range: tuple[float, float] = (-0.05, 0.05),
) -> torch.Tensor:
    """Determine if the object picking task is complete.

    This function checks whether all success conditions for the task have been met:
    1. object is within the target x/y range
    2. object is below a minimum height
    3. robot come back to the rest pose

    Args:
        env: The RL environment instance.
        objects_cfg: Configuration for the object entities.
        plate_cfg: Configuration for the plate entity.
        x_range: Range of x positions of the object for task completion.
        y_range: Range of y positions of the object for task completion.
        height_range: Range of height (z position) of the object for task completion.
    Returns:
        Boolean tensor indicating which environments have completed the task.
    """
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    plate: RigidObject = env.scene[plate_cfg.name]
    plate_x = plate.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    plate_y = plate.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    plate_height = plate.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    for object_cfg in objects_cfg:
        object_entity: RigidObject = env.scene[object_cfg.name]
        object_x = object_entity.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
        object_y = object_entity.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
        object_height = object_entity.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

        done = torch.logical_and(done, object_x < plate_x + x_range[1])
        done = torch.logical_and(done, object_x > plate_x + x_range[0])
        done = torch.logical_and(done, object_y < plate_y + y_range[1])
        done = torch.logical_and(done, object_y > plate_y + y_range[0])
        done = torch.logical_and(done, object_height < plate_height + height_range[1])
        done = torch.logical_and(done, object_height > plate_height + height_range[0])

    joint_pos = env.scene["robot"].data.joint_pos
    joint_names = env.scene["robot"].data.joint_names
    done = torch.logical_and(done, is_xtrainer_at_rest_pose(joint_pos, joint_names))

    return done


def task2_success(
    env: ManagerBasedRLEnv | DirectRLEnv,
    objects_cfg: List[SceneEntityCfg],
    protectlid_cfg: SceneEntityCfg = SceneEntityCfg("protectlid"),
    # 矩形区域参数（绝对坐标）
    rect_position: tuple[float, float, float] = (0.689, -0.262, 0.113),
    rect_size: tuple[float, float, float] = (0.144, 0.144, 0.050),
    # 物体位置偏移补偿值（用于补偿物体坐标原点不是真实中心的情况）
    object_offsets: Optional[Dict[str, tuple[float, float, float]]] = None,
    verbose: bool = True,
    visualize: bool = True,
) -> torch.Tensor:
    """判定task2任务是否成功完成。
    
    成功条件：
    1. Emergency_Stop_Button、nut（factory_nut_loose）、chip的判定点都被放置在指定矩形区域
       位置: (0.689, -0.262, 0.113)，尺寸: (0.144, 0.144, 0.050)
    2. 这些判定点同时被protectlid下方的区域盖住（判定点同时满足被这两段判定区域覆盖）
    
    Args:
        env: 环境实例
        objects_cfg: 需要判定的物体配置列表（Emergency_Stop_Button、factory_nut_loose、chip）
        protectlid_cfg: protectlid配置
        rect_position: 矩形区域中心位置 (x, y, z)
        rect_size: 矩形区域尺寸 (x, y, z)
        object_offsets: 物体位置偏移补偿值字典，格式为 {"物体名称": (x, y, z)}
        verbose: 是否输出调试信息
        visualize: 是否创建可视化（仅在第一次调用时创建）
    
    Returns:
        布尔张量，表示每个环境是否完成任务
    """
    # 添加调试输出，确认函数被调用
    if not hasattr(env, '_task2_success_called'):
        print("[DEBUG] ========== task2_success函数被调用 ==========")
        env._task2_success_called = True
    
    if object_offsets is None:
        object_offsets = {}
    
    # 创建可视化（仅在第一次调用时，且配置中启用了可视化）
    enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
    if visualize and enable_vis and not hasattr(env, '_task2_visualization_created'):
        print("[DEBUG] ========== 开始创建Task2判定区域可视化 ==========")
        try:
            # 创建矩形区域可视化（绿色）
            print(f"[DEBUG] 创建矩形区域可视化...")
            create_detection_zone_visualization(
                env=env,
                container_cfg=None,  # 使用绝对位置
                container_name="target_rect",
                position=rect_position,
                size=rect_size,
                color=(0.0, 1.0, 0.0),  # 绿色
                line_width=3.0,
            )
            
            # 获取protectlid的位置和尺寸（用于创建protectlid下方区域可视化）
            protectlid: RigidObject = env.scene[protectlid_cfg.name]
            # 假设protectlid是一个平面，我们需要获取它的位置和尺寸
            # 这里我们创建一个protectlid下方的区域可视化（蓝色，表示被盖住的区域）
            # 注意：protectlid下方区域需要根据protectlid的实际位置和尺寸来确定
            # 为了简化，我们假设protectlid下方区域是一个较大的区域，覆盖矩形区域
            protectlid_pos = protectlid.data.root_pos_w[0].cpu().numpy()  # 使用第一个环境的位置作为参考
            
            # 创建protectlid下方区域可视化（蓝色，半透明）
            # 区域大小
            protectlid_zone_size = (0.18, 0.24, 0.040)
            # 确保所有值都是Python float类型（对应C++ double）
            protectlid_zone_pos = (
                float(protectlid_pos[0]),
                float(protectlid_pos[1]),
                float(protectlid_pos[2] - protectlid_zone_size[2] / 2.0 + 0.02),  # protectlid下方，向上偏移0.02
            )
            print(f"[DEBUG] 创建protectlid下方区域可视化...")
            create_detection_zone_visualization(
                env=env,
                container_cfg=None,
                container_name="protectlid_zone",
                position=protectlid_zone_pos,
                size=protectlid_zone_size,
                color=(0.0, 0.0, 1.0),  # 蓝色
                line_width=3.0,
            )
            
            # 保存protectlid区域可视化的prim路径，以便后续更新位置
            try:
                stage = env.sim.stage
                protectlid_prim_path = protectlid.root_physx_view.prim_paths[0]
                path_parts = protectlid_prim_path.split("/")
                if len(path_parts) >= 4:
                    env_base_path = "/".join(path_parts[:4])
                else:
                    env_base_path = "/World/envs/env_0"
                
                env._protectlid_zone_prim_paths = [
                    f"{env_base_path.replace('env_0', f'env_{i}')}/Scene/DetectionZone_protectlid_zone"
                    for i in range(env.num_envs)
                ]
            except Exception as e:
                print(f"[WARNING] 无法保存protectlid区域prim路径: {e}")
                env._protectlid_zone_prim_paths = []
            
            # 为所有物体创建判定点可视化
            for obj_cfg in objects_cfg:
                obj_name = obj_cfg.name
                obj_offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
                print(f"[DEBUG] 开始创建{obj_name}判定点可视化...")
                create_object_detection_point_visualization(
                    env=env,
                    object_cfg=obj_cfg,
                    object_name=obj_name,
                    object_offset=obj_offset,
                    color=(1.0, 1.0, 0.0),  # 黄色
                    radius=0.015,  # 1.5cm半径
                )
            
            env._task2_visualization_created = True
            print("[INFO] ✓ Task2判定区域可视化已创建")
            print("      - 绿色框=目标矩形区域")
            print("      - 蓝色框=protectlid下方区域")
            print("      - 黄色球体=物体判定点")
        except Exception as e:
            print(f"[ERROR] 创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 初始化所有环境都成功
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # 缓存每个物体在判定区的状态，便于评分脚本读取
    if not hasattr(env, "_task2_in_rect_state"):
        env._task2_in_rect_state = {}
    if not hasattr(env, "_task2_in_intersection_state"):
        env._task2_in_intersection_state = {}
    # 保存最近一次判定快照，避免env重置后评分为0
    if not hasattr(env, "_task2_last_in_rect_state"):
        env._task2_last_in_rect_state = {}
    if not hasattr(env, "_task2_last_in_intersection_state"):
        env._task2_last_in_intersection_state = {}
    
    # 计算矩形区域的边界
    rect_x, rect_y, rect_z = rect_position
    rect_size_x, rect_size_y, rect_size_z = rect_size
    rect_x_min = rect_x - rect_size_x / 2.0
    rect_x_max = rect_x + rect_size_x / 2.0
    rect_y_min = rect_y - rect_size_y / 2.0
    rect_y_max = rect_y + rect_size_y / 2.0
    rect_z_min = rect_z - rect_size_z / 2.0
    rect_z_max = rect_z + rect_size_z / 2.0
    
    # 获取protectlid的位置（用于检查判定点是否在protectlid下方）
    protectlid: RigidObject = env.scene[protectlid_cfg.name]
    protectlid_pos = protectlid.data.root_pos_w  # (num_envs, 3)
    
    # 检查所有物体的判定点
    for obj_cfg in objects_cfg:
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
        
        # 检查判定点是否在矩形区域内（绝对坐标）
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
        in_rect = torch.logical_and(in_rect_x, in_rect_y)
        in_rect = torch.logical_and(in_rect, in_rect_z)
        
        # 计算protectlid下方区域的边界
        protectlid_zone_size = (0.18, 0.24, 0.040)
        protectlid_zone_size_x, protectlid_zone_size_y, protectlid_zone_size_z = protectlid_zone_size
        protectlid_zone_x_min = protectlid_pos[:, 0] - protectlid_zone_size_x / 2.0
        protectlid_zone_x_max = protectlid_pos[:, 0] + protectlid_zone_size_x / 2.0
        protectlid_zone_y_min = protectlid_pos[:, 1] - protectlid_zone_size_y / 2.0
        protectlid_zone_y_max = protectlid_pos[:, 1] + protectlid_zone_size_y / 2.0
        # 蓝色区域向上偏移0.02米
        protectlid_zone_z_min = protectlid_pos[:, 2] - protectlid_zone_size_z + 0.02
        protectlid_zone_z_max = protectlid_pos[:, 2] + 0.02
        
        # 检查判定点是否在protectlid下方区域内
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
        in_protectlid_zone = torch.logical_and(in_protectlid_x, in_protectlid_y)
        in_protectlid_zone = torch.logical_and(in_protectlid_zone, in_protectlid_z)
        
        # 两个区域的交集
        obj_success = torch.logical_and(in_rect, in_protectlid_zone)

        # 缓存当前判定结果
        env._task2_in_rect_state[obj_name] = in_rect.clone()
        env._task2_in_intersection_state[obj_name] = obj_success.clone()
        # 保存最近一次判定快照
        env._task2_last_in_rect_state[obj_name] = in_rect.clone()
        env._task2_last_in_intersection_state[obj_name] = obj_success.clone()
        
        done = torch.logical_and(done, obj_success)
        
        # 更新protectlid区域可视化位置（跟随protectlid移动）
        enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
        if enable_vis and hasattr(env, '_protectlid_zone_prim_paths'):
            try:
                from .observations import update_protectlid_zone_visualization
                protectlid_zone_size = (0.18, 0.24, 0.040)
                update_protectlid_zone_visualization(env, protectlid_cfg, protectlid_zone_size)
            except Exception as e:
                if not hasattr(env, '_protectlid_zone_update_error_shown'):
                    print(f"[WARNING] 更新protectlid区域可视化时出错: {e}")
        
        # 更新判定点可视化位置（如果存在且启用了可视化）
        if enable_vis and hasattr(env, '_detection_point_visualizations') and obj_name in env._detection_point_visualizations:
            try:
                # 尝试从配置中获取最新的偏移值并更新
                try:
                    if (hasattr(env, 'cfg') and 
                        hasattr(env.cfg, 'terminations') and 
                        env.cfg.terminations.success is not None and
                        hasattr(env.cfg.terminations.success, 'params')):
                        object_offsets_from_cfg = env.cfg.terminations.success.params.get("object_offsets", {})
                        if obj_name in object_offsets_from_cfg:
                            # 更新存储的偏移值
                            old_offset = env._detection_point_visualizations[obj_name]['object_offset']
                            new_offset = object_offsets_from_cfg[obj_name]
                            if old_offset != new_offset:
                                if not hasattr(env, '_offset_update_count'):
                                    env._offset_update_count = {}
                                if obj_name not in env._offset_update_count:
                                    env._offset_update_count[obj_name] = 0
                                env._offset_update_count[obj_name] += 1
                                if env._offset_update_count[obj_name] == 1:  # 只输出第一次
                                    print(f"[DEBUG] 更新偏移值 ({obj_name}): {old_offset} -> {new_offset}")
                                env._detection_point_visualizations[obj_name]['object_offset'] = new_offset
                except:
                    # 如果无法从配置读取，使用存储的值
                    pass
                
                # 更新可视化球体位置
                update_object_detection_point_visualization(env, obj_name)
            except Exception as e:
                if not hasattr(env, '_update_vis_error_count'):
                    env._update_vis_error_count = {}
                if obj_name not in env._update_vis_error_count:
                    env._update_vis_error_count[obj_name] = 0
                env._update_vis_error_count[obj_name] += 1
                if env._update_vis_error_count[obj_name] <= 3:  # 只输出前3次错误
                    print(f"[WARNING] 更新可视化时出错 ({obj_name}): {e}")
        
        # 输出调试信息
        if verbose:
            for env_id in range(env.num_envs):
                if not obj_success[env_id]:
                    dp_pos = detection_point_pos[env_id].cpu().numpy()
                    pl_pos = protectlid_pos[env_id].cpu().numpy()
                    print(f"[Env {env_id}] {obj_name} 判定点位置: ({dp_pos[0]:.3f}, {dp_pos[1]:.3f}, {dp_pos[2]:.3f})")
                    print(f"          矩形区域: X[{rect_x_min:.3f}, {rect_x_max:.3f}], Y[{rect_y_min:.3f}, {rect_y_max:.3f}], Z[{rect_z_min:.3f}, {rect_z_max:.3f}]")
                    print(f"          protectlid位置: ({pl_pos[0]:.3f}, {pl_pos[1]:.3f}, {pl_pos[2]:.3f})")
                    print(f"          在矩形内: {in_rect[env_id].item()}, 在protectlid下方: {in_protectlid_zone[env_id].item()}")
    
    # 输出任务完成信息
    if verbose:
        for env_id in range(env.num_envs):
            if done[env_id]:
                print(f"[Env {env_id}] ✓ Task2 任务成功完成！所有物体判定点已正确放置在目标区域且被protectlid覆盖。")
    
    return done