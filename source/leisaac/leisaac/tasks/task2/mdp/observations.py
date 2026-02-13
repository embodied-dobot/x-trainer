import torch
import math
from typing import TYPE_CHECKING, Optional

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

try:
    from pxr import UsdGeom, Gf, Usd, UsdShade
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    UsdShade = None

# ==============================================================================
# 通用抓取检测 (检查 nova2 是否被抓起)
# ==============================================================================
def object_grasped(
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"), # 这里泛指 nova2
        diff_threshold: float = 0.05,
        grasp_threshold: float = 0.01
) -> torch.Tensor:
    """检查指定的物体是否被机器人抓取。"""
    robot: Articulation = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    object_entity: RigidObject = env.scene[object_cfg.name]

    # 获取数据
    object_pos = object_entity.data.root_pos_w
    # 注意：根据你的Frame配置，这里可能需要调整索引，假设只有一个末端
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    # 计算抓取中心 (假设 Z 轴偏移 0.16m，根据你的夹爪实际情况调整)
    offset_local = torch.tensor([0.0, 0.0, 0.16], device=env.device).repeat(env.num_envs, 1)
    offset_world = quat_apply(ee_quat_w, offset_local)  
    grasp_center_pos = ee_pos_w + offset_world

    # 计算距离
    pos_diff = torch.linalg.vector_norm(object_pos - grasp_center_pos, dim=1)

    # 检查夹爪闭合力度 (假设 J2_8 是夹爪关节)
    # 注意：如果你的夹爪关节名变了，这里要改
    joint_ids, _ = robot.find_joints("J2_8")
    is_gripper_closed = robot.data.joint_pos[:, joint_ids[0]] > grasp_threshold
    
    # 判定：距离近 且 夹爪用力
    grasped = torch.logical_and(pos_diff < diff_threshold, is_gripper_closed)

    return grasped

# ==============================================================================
# 通用放置检测 (检查物体是否在容器内)
# ==============================================================================
def object_in_container(
        env: ManagerBasedRLEnv | DirectRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
        # 容器的有效范围 (Box check)
        x_range: tuple[float, float] = (-0.10, 0.10), 
        y_range: tuple[float, float] = (-0.10, 0.10),
        z_range: tuple[float, float] = (0.0, 0.20),
        # 物体位置偏移补偿值 (用于补偿物体坐标原点不是真实中心的情况)
        object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        # 是否输出调试信息
        verbose: bool = False,
) -> torch.Tensor:
    """检查物体是否位于容器的范围内。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        container_cfg: 容器配置
        x_range: X轴范围
        y_range: Y轴范围
        z_range: Z轴范围
        object_offset: 物体位置偏移补偿值 (x, y, z)
        verbose: 是否输出调试信息（仅在状态变化时输出）
    """
    object_entity: RigidObject = env.scene[object_cfg.name]
    container: RigidObject = env.scene[container_cfg.name]

    # 获取物体位置和旋转，应用偏移补偿（考虑旋转）
    object_pos = object_entity.data.root_pos_w.clone()
    object_quat = object_entity.data.root_quat_w.clone()
    
    # 将偏移值从局部坐标系转换到世界坐标系（考虑物体旋转）
    if object_offset != (0.0, 0.0, 0.0):
        offset_local = torch.tensor(object_offset, device=env.device, dtype=object_pos.dtype)
        # 为每个环境应用旋转
        offset_world = quat_apply(object_quat, offset_local.unsqueeze(0).repeat(env.num_envs, 1))
        object_pos = object_pos + offset_world
    else:
        offset_world = torch.zeros_like(object_pos)

    # 获取相对位置： 物体 - 容器
    rel_pos = object_pos - container.data.root_pos_w
    
    # 检查 X, Y, Z 是否都在范围内
    in_x = torch.logical_and(rel_pos[:, 0] > x_range[0], rel_pos[:, 0] < x_range[1])
    in_y = torch.logical_and(rel_pos[:, 1] > y_range[0], rel_pos[:, 1] < y_range[1])
    # Z轴检查：物体必须比容器底部高一点，但不能飞太高
    in_z = torch.logical_and(rel_pos[:, 2] > z_range[0], rel_pos[:, 2] < z_range[1])

    in_container = torch.logical_and(in_x, in_y)
    in_container = torch.logical_and(in_container, in_z)

    # 更新判定点可视化位置（如果存在且启用了可视化）
    enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
    if enable_vis and hasattr(env, '_detection_point_visualizations') and object_cfg.name in env._detection_point_visualizations:
        try:
            # 尝试从配置中获取最新的偏移值并更新
            try:
                if (hasattr(env, 'cfg') and 
                    hasattr(env.cfg, 'terminations') and 
                    env.cfg.terminations.success is not None and
                    hasattr(env.cfg.terminations.success, 'params')):
                    object_offsets = env.cfg.terminations.success.params.get("object_offsets", {})
                    if object_cfg.name in object_offsets:
                        # 更新存储的偏移值
                        old_offset = env._detection_point_visualizations[object_cfg.name]['object_offset']
                        new_offset = object_offsets[object_cfg.name]
                        if old_offset != new_offset:
                            if not hasattr(env, '_offset_update_count'):
                                env._offset_update_count = {}
                            if object_cfg.name not in env._offset_update_count:
                                env._offset_update_count[object_cfg.name] = 0
                            env._offset_update_count[object_cfg.name] += 1
                            if env._offset_update_count[object_cfg.name] == 1:  # 只输出第一次
                                print(f"[DEBUG] 更新偏移值 ({object_cfg.name}): {old_offset} -> {new_offset}")
                            env._detection_point_visualizations[object_cfg.name]['object_offset'] = new_offset
            except:
                # 如果无法从配置读取，使用存储的值
                pass
            
            update_object_detection_point_visualization(env, object_cfg.name)
        except Exception as e:
            if not hasattr(env, '_update_vis_error_count'):
                env._update_vis_error_count = {}
            if object_cfg.name not in env._update_vis_error_count:
                env._update_vis_error_count[object_cfg.name] = 0
            env._update_vis_error_count[object_cfg.name] += 1
            if env._update_vis_error_count[object_cfg.name] <= 3:  # 只输出前3次错误
                print(f"[WARNING] 更新可视化时出错 ({object_cfg.name}): {e}")
    
    # 输出调试信息（使用环境实例存储状态，避免全局状态问题）
    if verbose:
        # 使用环境实例的属性来存储状态
        if not hasattr(env, '_object_in_container_state'):
            env._object_in_container_state = {}
        
        # 添加调用计数，用于调试
        if not hasattr(env, '_object_in_container_call_count'):
            env._object_in_container_call_count = {}
        
        state_key = f"{object_cfg.name}_{container_cfg.name}"
        if state_key not in env._object_in_container_state:
            env._object_in_container_state[state_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            env._object_in_container_call_count[state_key] = 0
        
        # 记录调用次数（每100次输出一次，避免刷屏）
        env._object_in_container_call_count[state_key] += 1
        if env._object_in_container_call_count[state_key] == 1:
            print(f"[DEBUG] ObsTerm被调用: {object_cfg.name} -> {container_cfg.name}")
        
        prev_state = env._object_in_container_state[state_key]
        # 检测状态变化：从False变为True
        state_changed = torch.logical_and(in_container, ~prev_state)
        
        for env_id in range(env.num_envs):
            if state_changed[env_id]:
                print(f"[Env {env_id}] ✓ 物体 {object_cfg.name} 已放入容器 {container_cfg.name}")
        
        # 更新状态
        env._object_in_container_state[state_key] = in_container.clone()

    return in_container


# ==============================================================================
# Task2专用：更新物体判定点可视化（不进行判定，只更新可视化）
# ==============================================================================
def update_task2_detection_point(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    verbose: bool = False,
) -> torch.Tensor:
    """Task2专用：更新物体判定点的可视化位置。
    
    这个函数不进行任何判定，只负责更新可视化球体的位置。
    用于在ObservationsCfg中配置，确保每个步骤都更新可视化。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        object_offset: 物体位置偏移补偿值 (x, y, z)
        verbose: 是否输出调试信息
    
    Returns:
        总是返回False（因为不进行判定）
    """
    # 更新判定点可视化位置（如果存在且启用了可视化）
    enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
    if enable_vis and hasattr(env, '_detection_point_visualizations') and object_cfg.name in env._detection_point_visualizations:
        try:
            # 尝试从配置中获取最新的偏移值并更新
            try:
                if (hasattr(env, 'cfg') and 
                    hasattr(env.cfg, 'terminations') and 
                    env.cfg.terminations.success is not None and
                    hasattr(env.cfg.terminations.success, 'params')):
                    object_offsets = env.cfg.terminations.success.params.get("object_offsets", {})
                    if object_cfg.name in object_offsets:
                        # 更新存储的偏移值
                        old_offset = env._detection_point_visualizations[object_cfg.name]['object_offset']
                        new_offset = object_offsets[object_cfg.name]
                        if old_offset != new_offset:
                            if not hasattr(env, '_offset_update_count'):
                                env._offset_update_count = {}
                            if object_cfg.name not in env._offset_update_count:
                                env._offset_update_count[object_cfg.name] = 0
                            env._offset_update_count[object_cfg.name] += 1
                            if env._offset_update_count[object_cfg.name] == 1:  # 只输出第一次
                                print(f"[DEBUG] 更新偏移值 ({object_cfg.name}): {old_offset} -> {new_offset}")
                            env._detection_point_visualizations[object_cfg.name]['object_offset'] = new_offset
            except:
                # 如果无法从配置读取，使用存储的值
                pass
            
            # 更新可视化球体位置
            update_object_detection_point_visualization(env, object_cfg.name)
        except Exception as e:
            if not hasattr(env, '_update_vis_error_count'):
                env._update_vis_error_count = {}
            if object_cfg.name not in env._update_vis_error_count:
                env._update_vis_error_count[object_cfg.name] = 0
            env._update_vis_error_count[object_cfg.name] += 1
            if env._update_vis_error_count[object_cfg.name] <= 3:  # 只输出前3次错误
                print(f"[WARNING] 更新可视化时出错 ({object_cfg.name}): {e}")
    
    # 返回False，因为这只是用于更新可视化，不进行判定
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# ==============================================================================
# Task2专用：更新protectlid下方区域可视化位置
# ==============================================================================
def update_protectlid_zone_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    protectlid_cfg: SceneEntityCfg,
    zone_size: tuple[float, float, float] = (0.18, 0.24, 0.040),  # 蓝色判定区大小
    color: tuple[float, float, float] = (0.0, 0.0, 1.0),  # 蓝色
) -> None:
    """更新protectlid下方区域的可视化位置和旋转，使其跟随protectlid移动和旋转。
    
    旋转只考虑XY平面的旋转（绕Z轴旋转）。
    
    Args:
        env: 环境实例
        protectlid_cfg: protectlid配置
        zone_size: 区域尺寸 (x, y, z)
        color: RGB颜色 (0-1范围)
    """
    if not USD_AVAILABLE:
        return
    
    if not hasattr(env, '_protectlid_zone_prim_paths'):
        return
    
    try:
        stage = env.sim.stage
        protectlid: RigidObject = env.scene[protectlid_cfg.name]
        
        # 更新每个环境的区域位置和旋转
        for env_id, prim_path in enumerate(env._protectlid_zone_prim_paths):
            # 获取protectlid当前位置和旋转
            protectlid_pos = protectlid.data.root_pos_w[env_id].cpu().numpy()
            protectlid_quat = protectlid.data.root_quat_w[env_id].cpu().numpy()  # (w, x, y, z)
            
            # 计算区域中心位置（protectlid下方）
            zone_center_z = protectlid_pos[2] - zone_size[2] / 2.0 + 0.05
            zone_pos = (
                float(protectlid_pos[0]),
                float(protectlid_pos[1]),
                float(zone_center_z),
            )
            
            # 将四元数转换为欧拉角（XYZ顺序）
            # 四元数格式：(w, x, y, z)
            # 使用标准公式将四元数转换为欧拉角（XYZ顺序）
            w, x, y, z = protectlid_quat[0], protectlid_quat[1], protectlid_quat[2], protectlid_quat[3]
            
            # 计算欧拉角（XYZ顺序，即roll, pitch, yaw）
            # Roll (X轴旋转)
            sin_roll = 2.0 * (w * x + y * z)
            cos_roll = 1.0 - 2.0 * (x * x + y * y)
            roll_rad = math.atan2(sin_roll, cos_roll)
            
            # Pitch (Y轴旋转)
            sin_pitch = 2.0 * (w * y - z * x)
            if abs(sin_pitch) >= 1.0:
                # 万向锁情况，使用备用计算
                pitch_rad = math.copysign(math.pi / 2.0, sin_pitch)
            else:
                pitch_rad = math.asin(sin_pitch)
            
            # Yaw (Z轴旋转)
            sin_yaw = 2.0 * (w * z + x * y)
            cos_yaw = 1.0 - 2.0 * (y * y + z * z)
            yaw_rad = math.atan2(sin_yaw, cos_yaw)
            
            # 转换为度数（USD使用度数）
            x_rotation_deg = float(roll_rad * 180.0 / math.pi)   # roll (绕X轴)
            y_rotation_deg = float(pitch_rad * 180.0 / math.pi)  # pitch (绕Y轴)
            z_rotation_deg = float(yaw_rad * 180.0 / math.pi)    # yaw (绕Z轴)
            
            # 获取prim并更新位置和旋转
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                continue
                
            try:
                xformable = UsdGeom.Xformable(prim)
                xform_ops = xformable.GetOrderedXformOps()
                
                # 查找或创建translate和rotateXYZ操作
                translate_op = None
                rotate_xyz_op = None
                
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                    elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        rotate_xyz_op = op
                
                # 如果缺少操作，清除现有的xform操作并创建新的
                if translate_op is None or rotate_xyz_op is None:
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                    rotate_xyz_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
                
                # 更新位置和旋转（XYZ欧拉角）
                translate_op.Set(Gf.Vec3d(*zone_pos))
                rotate_xyz_op.Set(Gf.Vec3d(x_rotation_deg, y_rotation_deg, z_rotation_deg))
                
            except Exception as update_e:
                if not hasattr(env, '_protectlid_zone_update_error_shown'):
                    print(f"[ERROR] 更新protectlid区域位置和旋转时出错 ({prim_path}): {update_e}")
                    import traceback
                    traceback.print_exc()
                    env._protectlid_zone_update_error_shown = True
        
    except Exception as e:
        if not hasattr(env, '_protectlid_zone_update_error_shown'):
            print(f"[ERROR] 更新protectlid区域可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            env._protectlid_zone_update_error_shown = True


# ==============================================================================
# Task2专用：检测物体是否在两个判定区的交集中
# ==============================================================================
def object_in_task2_intersection(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    protectlid_cfg: SceneEntityCfg = SceneEntityCfg("protectlid"),
    # 矩形区域参数（绝对坐标）
    rect_position: tuple[float, float, float] = (0.689, -0.262, 0.113),
    rect_size: tuple[float, float, float] = (0.144, 0.144, 0.050),
    # protectlid下方区域尺寸
    protectlid_zone_size: tuple[float, float, float] = (0.18, 0.24, 0.040),
    # 物体位置偏移补偿值
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    # 是否输出调试信息
    verbose: bool = False,
) -> torch.Tensor:
    """检测物体判定点是否在两个判定区的交集中。
    
    两个判定区：
    1. 目标矩形区域（固定位置）
    2. protectlid下方区域（跟随protectlid移动）
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        protectlid_cfg: protectlid配置
        rect_position: 矩形区域中心位置 (x, y, z)
        rect_size: 矩形区域尺寸 (x, y, z)
        protectlid_zone_size: protectlid下方区域尺寸 (x, y, z)
        object_offset: 物体位置偏移补偿值 (x, y, z)
        verbose: 是否输出调试信息（仅在状态变化时输出）
    
    Returns:
        布尔张量，表示每个环境的判定点是否在交集中
    """
    object_entity: RigidObject = env.scene[object_cfg.name]
    protectlid: RigidObject = env.scene[protectlid_cfg.name]
    
    # 优先从配置中读取object_offset，如果没有则使用传入的参数
    actual_object_offset = object_offset
    try:
        if (hasattr(env, 'cfg') and 
            hasattr(env.cfg, 'terminations') and 
            env.cfg.terminations.success is not None and
            hasattr(env.cfg.terminations.success, 'params')):
            object_offsets = env.cfg.terminations.success.params.get("object_offsets", {})
            if object_cfg.name in object_offsets:
                actual_object_offset = object_offsets[object_cfg.name]
    except:
        pass  # 如果无法从配置读取，使用传入的参数
    
    # 获取物体位置和旋转，应用偏移补偿
    object_pos = object_entity.data.root_pos_w.clone()
    object_quat = object_entity.data.root_quat_w.clone()
    
    # 将偏移值从局部坐标系转换到世界坐标系（考虑物体旋转）
    if actual_object_offset != (0.0, 0.0, 0.0):
        offset_local = torch.tensor(actual_object_offset, device=env.device, dtype=object_pos.dtype)
        offset_world = quat_apply(object_quat, offset_local.unsqueeze(0).repeat(env.num_envs, 1))
        detection_point_pos = object_pos + offset_world
    else:
        detection_point_pos = object_pos
    
    # 计算矩形区域的边界
    rect_x, rect_y, rect_z = rect_position
    rect_size_x, rect_size_y, rect_size_z = rect_size
    rect_x_min = rect_x - rect_size_x / 2.0
    rect_x_max = rect_x + rect_size_x / 2.0
    rect_y_min = rect_y - rect_size_y / 2.0
    rect_y_max = rect_y + rect_size_y / 2.0
    rect_z_min = rect_z - rect_size_z / 2.0
    rect_z_max = rect_z + rect_size_z / 2.0
    
    # 检查判定点是否在矩形区域内
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
    
    # 获取protectlid位置
    protectlid_pos = protectlid.data.root_pos_w  # (num_envs, 3)
    
    # 计算protectlid下方区域的边界
    protectlid_zone_size_x, protectlid_zone_size_y, protectlid_zone_size_z = protectlid_zone_size
    protectlid_zone_x_min = protectlid_pos[:, 0] - protectlid_zone_size_x / 2.0
    protectlid_zone_x_max = protectlid_pos[:, 0] + protectlid_zone_size_x / 2.0
    protectlid_zone_y_min = protectlid_pos[:, 1] - protectlid_zone_size_y / 2.0
    protectlid_zone_y_max = protectlid_pos[:, 1] + protectlid_zone_size_y / 2.0
    # 蓝色判定区向上调整0.05米
    protectlid_zone_z_min = protectlid_pos[:, 2] - protectlid_zone_size_z + 0.05
    protectlid_zone_z_max = protectlid_pos[:, 2] + 0.05
    
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
    in_intersection = torch.logical_and(in_rect, in_protectlid_zone)
    
    # 更新protectlid区域可视化位置
    enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
    if enable_vis and hasattr(env, '_protectlid_zone_prim_paths'):
        try:
            update_protectlid_zone_visualization(env, protectlid_cfg, protectlid_zone_size)
        except Exception as e:
            if not hasattr(env, '_protectlid_zone_update_error_shown'):
                print(f"[WARNING] 更新protectlid区域可视化时出错: {e}")
    
    # 输出调试信息（仅在状态变化时输出）
    if verbose:
        if not hasattr(env, '_task2_intersection_state'):
            env._task2_intersection_state = {}
        
        state_key = f"{object_cfg.name}_intersection"
        if state_key not in env._task2_intersection_state:
            env._task2_intersection_state[state_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
        prev_state = env._task2_intersection_state[state_key]
        # 检测状态变化：从False变为True
        state_changed = torch.logical_and(in_intersection, ~prev_state)
        
        for env_id in range(env.num_envs):
            if state_changed[env_id]:
                dp_pos = detection_point_pos[env_id].cpu().numpy()
                pl_pos = protectlid_pos[env_id].cpu().numpy()
                print(f"[Env {env_id}] ✓ 物体 {object_cfg.name} 的判定点已进入两个判定区的交集")
                print(f"          判定点位置: ({dp_pos[0]:.3f}, {dp_pos[1]:.3f}, {dp_pos[2]:.3f})")
                print(f"          protectlid位置: ({pl_pos[0]:.3f}, {pl_pos[1]:.3f}, {pl_pos[2]:.3f})")
        
        # 更新状态
        env._task2_intersection_state[state_key] = in_intersection.clone()
    
    return in_intersection


# ==============================================================================
# 可视化判定区域
# ==============================================================================
def create_detection_zone_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    container_cfg: Optional[SceneEntityCfg] = None,
    container_name: str = "zone",
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    size: tuple[float, float, float] = (0.2, 0.2, 0.2),
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    z_range: Optional[tuple[float, float]] = None,
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),  # 默认红色
    line_width: float = 2.0,
) -> None:
    """在Isaac Sim中创建判定区域的可视化线框盒子。
    
    可以基于容器（container_cfg）的相对位置，或直接使用绝对位置（position）。
    
    Args:
        env: 环境实例
        container_cfg: 容器配置（可选，如果提供则基于容器位置）
        container_name: 容器名称（用于命名可视化对象）
        position: 绝对位置 (x, y, z)（当container_cfg为None时使用）
        size: 盒子尺寸 (x, y, z)
        x_range: X轴范围（当container_cfg提供时使用）
        y_range: Y轴范围（当container_cfg提供时使用）
        z_range: Z轴范围（当container_cfg提供时使用）
        color: RGB颜色 (0-1范围)
        line_width: 线宽（像素）
    """
    if not USD_AVAILABLE:
        print("[WARNING] USD API不可用，无法创建可视化")
        return
    
    try:
        stage = env.sim.stage
        
        # 确定盒子尺寸和中心
        if container_cfg is not None:
            # 基于容器的相对位置
            container: RigidObject = env.scene[container_cfg.name]
            
            # 获取容器的prim路径
            try:
                container_prim_path = container.root_physx_view.prim_paths[0]
            except:
                container_prim_path = None
                for env_id in range(min(1, env.num_envs)):
                    test_path = f"/World/envs/env_{env_id}/Scene/{container_cfg.name}"
                    if stage.GetPrimAtPath(test_path).IsValid():
                        container_prim_path = test_path
                        break
                
                if container_prim_path is None:
                    print(f"[ERROR] 无法找到容器 {container_cfg.name} 的prim路径")
                    return
            
            # 从容器路径提取环境命名空间
            path_parts = container_prim_path.split("/")
            if len(path_parts) >= 4:
                env_base_path = "/".join(path_parts[:4])
            else:
                env_base_path = "/World/envs/env_0"
            
            # 使用范围计算尺寸和中心
            if x_range is not None and y_range is not None and z_range is not None:
                x_min, x_max = x_range
                y_min, y_max = y_range
                z_min, z_max = z_range
                
                box_size_x = x_max - x_min
                box_size_y = y_max - y_min
                box_size_z = z_max - z_min
                
                box_center_x = (x_min + x_max) / 2.0
                box_center_y = (y_min + y_max) / 2.0
                box_center_z = (z_min + z_max) / 2.0
            else:
                box_size_x, box_size_y, box_size_z = size
                box_center_x = box_center_y = box_center_z = 0.0
            
            # 为每个环境创建可视化
            for env_id in range(env.num_envs):
                # 获取容器位置
                container_pos = container.data.root_pos_w[env_id].cpu().numpy()
                
                # 计算可视化盒子的世界位置（容器位置 + 相对中心）
                # 确保所有值都是Python float类型（对应C++ double）
                vis_pos = (
                    float(container_pos[0] + box_center_x),
                    float(container_pos[1] + box_center_y),
                    float(container_pos[2] + box_center_z),
                )
                
                # 创建可视化prim路径
                env_path = env_base_path.replace("env_0", f"env_{env_id}")
                prim_path = f"{env_path}/Scene/DetectionZone_{container_name}"
                
                # 如果已存在，先删除
                existing_prim = stage.GetPrimAtPath(prim_path)
                if existing_prim.IsValid():
                    stage.RemovePrim(prim_path)
                
                # 创建Xform作为父节点
                xform = UsdGeom.Xform.Define(stage, prim_path)
                xform_ops = UsdGeom.Xformable(xform)
                xform_ops.ClearXformOpOrder()
                translate_op = xform_ops.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                translate_op.Set(Gf.Vec3d(*vis_pos))
                
                # 创建Cube几何体
                cube_path = f"{prim_path}/Cube"
                cube = UsdGeom.Cube.Define(stage, cube_path)
                
                # 设置尺寸
                cube.GetSizeAttr().Set(1.0)
                
                # 设置缩放以匹配实际尺寸
                cube_prim = cube.GetPrim()
                xformable = UsdGeom.Xformable(cube_prim)
                xformable.ClearXformOpOrder()
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
                scale_op.Set(Gf.Vec3f(box_size_x, box_size_y, box_size_z))
                
                # 设置颜色
                color_attr = cube.CreateDisplayColorAttr()
                color_attr.Set([Gf.Vec3f(*color)])
                
                # 创建半透明材质
                if UsdShade is not None:
                    try:
                        material_path = f"{prim_path}/Material"
                        material = UsdShade.Material.Define(stage, material_path)
                        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                        shader.CreateIdAttr("UsdPreviewSurface")
                        shader.CreateInput("diffuseColor", UsdShade.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                        shader.CreateInput("roughness", UsdShade.Sdf.ValueTypeNames.Float).Set(0.1)
                        shader.CreateInput("metallic", UsdShade.Sdf.ValueTypeNames.Float).Set(0.0)
                        shader.CreateInput("opacity", UsdShade.Sdf.ValueTypeNames.Float).Set(0.3)
                        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                        UsdShade.MaterialBindingAPI(cube_prim).Bind(material)
                    except Exception as mat_e:
                        print(f"[WARNING] 设置材质时出错（可忽略）: {mat_e}")
                
                print(f"[INFO] 创建可视化Cube: {cube_path}")
                print(f"      - 位置: ({vis_pos[0]:.3f}, {vis_pos[1]:.3f}, {vis_pos[2]:.3f})")
                print(f"      - 尺寸: ({box_size_x:.3f}, {box_size_y:.3f}, {box_size_z:.3f})")
                print(f"      - 颜色: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
            
            print(f"[INFO] 已为容器 {container_name} 创建判定区域可视化（{env.num_envs}个环境）")
        else:
            # 使用绝对位置
            box_size_x, box_size_y, box_size_z = size
            # 确保所有值都是Python float类型（对应C++ double）
            vis_pos = (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
            
            # 为每个环境创建可视化
            for env_id in range(env.num_envs):
                env_path = f"/World/envs/env_{env_id}"
                prim_path = f"{env_path}/Scene/DetectionZone_{container_name}"
                
                # 如果已存在，先删除
                existing_prim = stage.GetPrimAtPath(prim_path)
                if existing_prim.IsValid():
                    stage.RemovePrim(prim_path)
                
                # 创建Xform作为父节点
                xform = UsdGeom.Xform.Define(stage, prim_path)
                xform_ops = UsdGeom.Xformable(xform)
                xform_ops.ClearXformOpOrder()
                translate_op = xform_ops.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                translate_op.Set(Gf.Vec3d(*vis_pos))
                
                # 创建Cube几何体
                cube_path = f"{prim_path}/Cube"
                cube = UsdGeom.Cube.Define(stage, cube_path)
                
                # 设置尺寸
                cube.GetSizeAttr().Set(1.0)
                
                # 设置缩放以匹配实际尺寸
                cube_prim = cube.GetPrim()
                xformable = UsdGeom.Xformable(cube_prim)
                xformable.ClearXformOpOrder()
                scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
                scale_op.Set(Gf.Vec3f(box_size_x, box_size_y, box_size_z))
                
                # 设置颜色
                color_attr = cube.CreateDisplayColorAttr()
                color_attr.Set([Gf.Vec3f(*color)])
                
                # 创建半透明材质
                if UsdShade is not None:
                    try:
                        material_path = f"{prim_path}/Material"
                        material = UsdShade.Material.Define(stage, material_path)
                        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                        shader.CreateIdAttr("UsdPreviewSurface")
                        shader.CreateInput("diffuseColor", UsdShade.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                        shader.CreateInput("roughness", UsdShade.Sdf.ValueTypeNames.Float).Set(0.1)
                        shader.CreateInput("metallic", UsdShade.Sdf.ValueTypeNames.Float).Set(0.0)
                        shader.CreateInput("opacity", UsdShade.Sdf.ValueTypeNames.Float).Set(0.3)
                        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                        UsdShade.MaterialBindingAPI(cube_prim).Bind(material)
                    except Exception as mat_e:
                        print(f"[WARNING] 设置材质时出错（可忽略）: {mat_e}")
                
                print(f"[INFO] 创建可视化Cube: {cube_path}")
                print(f"      - 位置: ({vis_pos[0]:.3f}, {vis_pos[1]:.3f}, {vis_pos[2]:.3f})")
                print(f"      - 尺寸: ({box_size_x:.3f}, {box_size_y:.3f}, {box_size_z:.3f})")
                print(f"      - 颜色: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
            
            print(f"[INFO] 已创建判定区域可视化 {container_name}（{env.num_envs}个环境）")
        
    except Exception as e:
        print(f"[ERROR] 创建可视化时出错: {e}")
        import traceback
        traceback.print_exc()


def create_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_cfg: SceneEntityCfg,
    object_name: str,
    object_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: tuple[float, float, float] = (1.0, 1.0, 0.0),  # 默认黄色
    radius: float = 0.015,  # 1.5cm半径
) -> None:
    """在Isaac Sim中创建物体判定点的可视化球体。
    
    显示物体应用偏移补偿后的实际判定位置。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        object_name: 物体名称（用于命名可视化对象）
        object_offset: 物体位置偏移补偿值 (x, y, z)
        color: RGB颜色 (0-1范围)
        radius: 球体半径（米）
    """
    if not USD_AVAILABLE:
        print("[WARNING] USD API不可用，无法创建可视化")
        return
    
    try:
        stage = env.sim.stage
        object_entity: RigidObject = env.scene[object_cfg.name]
        
        # 获取物体的prim路径
        try:
            object_prim_path = object_entity.root_physx_view.prim_paths[0]
        except:
            object_prim_path = None
            for env_id in range(min(1, env.num_envs)):
                test_path = f"/World/envs/env_{env_id}/Scene/{object_cfg.name}"
                if stage.GetPrimAtPath(test_path).IsValid():
                    object_prim_path = test_path
                    break
            
            if object_prim_path is None:
                print(f"[ERROR] 无法找到物体 {object_cfg.name} 的prim路径")
                return
        
        # 从物体路径提取环境命名空间
        path_parts = object_prim_path.split("/")
        if len(path_parts) >= 4:
            env_base_path = "/".join(path_parts[:4])
        else:
            env_base_path = "/World/envs/env_0"
        
        # 为每个环境创建可视化
        for env_id in range(env.num_envs):
            # 获取物体位置和旋转
            object_pos = object_entity.data.root_pos_w[env_id].cpu()
            object_quat = object_entity.data.root_quat_w[env_id].cpu()
            
            # 将偏移值从局部坐标系转换到世界坐标系（考虑物体旋转）
            offset_local = torch.tensor(object_offset, device=object_pos.device, dtype=object_pos.dtype)
            offset_world = quat_apply(object_quat, offset_local)
            
            # 计算判定点的世界位置（物体位置 + 旋转后的偏移补偿）
            detection_point_pos = (
                float(object_pos[0] + offset_world[0]),
                float(object_pos[1] + offset_world[1]),
                float(object_pos[2] + offset_world[2]),
            )
            
            # 创建可视化prim路径
            env_path = env_base_path.replace("env_0", f"env_{env_id}")
            prim_path = f"{env_path}/Scene/DetectionPoint_{object_name}"
            
            # 如果已存在，先删除
            existing_prim = stage.GetPrimAtPath(prim_path)
            if existing_prim.IsValid():
                stage.RemovePrim(prim_path)
            
            # 创建Xform作为父节点
            xform = UsdGeom.Xform.Define(stage, prim_path)
            xform_ops = UsdGeom.Xformable(xform)
            xform_ops.ClearXformOpOrder()
            translate_op = xform_ops.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            translate_op.Set(Gf.Vec3d(*detection_point_pos))
            
            # 创建Sphere几何体
            sphere_path = f"{prim_path}/Sphere"
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            
            # 设置半径
            sphere.GetRadiusAttr().Set(radius)
            
            # 设置颜色
            color_attr = sphere.CreateDisplayColorAttr()
            color_attr.Set([Gf.Vec3f(*color)])
            
            # 创建不透明材质以便清晰观察
            if UsdShade is not None:
                try:
                    material_path = f"{prim_path}/Material"
                    material = UsdShade.Material.Define(stage, material_path)
                    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", UsdShade.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                    shader.CreateInput("roughness", UsdShade.Sdf.ValueTypeNames.Float).Set(0.2)
                    shader.CreateInput("metallic", UsdShade.Sdf.ValueTypeNames.Float).Set(0.0)
                    shader.CreateInput("opacity", UsdShade.Sdf.ValueTypeNames.Float).Set(1.0)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    
                    sphere_prim = sphere.GetPrim()
                    UsdShade.MaterialBindingAPI(sphere_prim).Bind(material)
                except Exception as mat_e:
                    print(f"[WARNING] 设置材质时出错（可忽略）: {mat_e}")
            
            print(f"[INFO] 创建判定点可视化Sphere: {sphere_path}")
            print(f"      - 位置: ({detection_point_pos[0]:.3f}, {detection_point_pos[1]:.3f}, {detection_point_pos[2]:.3f})")
            print(f"      - 半径: {radius:.3f}m")
            print(f"      - 颜色: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
            print(f"      - 偏移: ({object_offset[0]:.5f}, {object_offset[1]:.5f}, {object_offset[2]:.5f})")
        
        print(f"[INFO] 已为物体 {object_name} 创建判定点可视化（{env.num_envs}个环境）")
        
        # 存储prim路径和偏移值，以便后续更新
        if not hasattr(env, '_detection_point_visualizations'):
            env._detection_point_visualizations = {}
        
        env._detection_point_visualizations[object_name] = {
            'prim_paths': [f"{env_base_path.replace('env_0', f'env_{i}')}/Scene/DetectionPoint_{object_name}" 
                          for i in range(env.num_envs)],
            'object_offset': object_offset,
            'object_cfg': object_cfg,
        }
        
    except Exception as e:
        print(f"[ERROR] 创建判定点可视化时出错: {e}")
        import traceback
        traceback.print_exc()


def update_object_detection_point_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    object_name: str,
) -> None:
    """更新物体判定点可视化球体的位置，使其跟随物体移动。
    
    Args:
        env: 环境实例
        object_name: 物体名称
    """
    if not USD_AVAILABLE:
        return
    
    if not hasattr(env, '_detection_point_visualizations'):
        return
    
    if object_name not in env._detection_point_visualizations:
        return
    
    try:
        stage = env.sim.stage
        vis_info = env._detection_point_visualizations[object_name]
        object_entity: RigidObject = env.scene[vis_info['object_cfg'].name]
        
        # 优先从配置中获取最新的偏移值，如果没有则使用存储的值
        object_offset = vis_info['object_offset']
        try:
            if (hasattr(env, 'cfg') and 
                hasattr(env.cfg, 'terminations') and 
                env.cfg.terminations.success is not None and
                hasattr(env.cfg.terminations.success, 'params')):
                object_offsets = env.cfg.terminations.success.params.get("object_offsets", {})
                if object_name in object_offsets:
                    new_offset = object_offsets[object_name]
                    if new_offset != object_offset:
                        if not hasattr(env, '_offset_change_shown'):
                            env._offset_change_shown = {}
                        if object_name not in env._offset_change_shown:
                            print(f"[DEBUG] 检测到偏移值变化 ({object_name}): {object_offset} -> {new_offset}")
                            env._offset_change_shown[object_name] = True
                        object_offset = new_offset
                        vis_info['object_offset'] = object_offset
        except:
            pass
        
        # 更新每个环境的球体位置
        for env_id, prim_path in enumerate(vis_info['prim_paths']):
            # 获取物体当前位置和旋转
            object_pos = object_entity.data.root_pos_w[env_id].cpu()
            object_quat = object_entity.data.root_quat_w[env_id].cpu()
            
            # 将偏移值从局部坐标系转换到世界坐标系（考虑物体旋转）
            offset_local = torch.tensor(object_offset, device=object_pos.device, dtype=object_pos.dtype)
            offset_world = quat_apply(object_quat, offset_local)
            
            # 计算判定点的世界位置（物体位置 + 旋转后的偏移补偿）
            detection_point_pos = (
                float(object_pos[0] + offset_world[0]),
                float(object_pos[1] + offset_world[1]),
                float(object_pos[2] + offset_world[2]),
            )
            
            # 获取prim并更新位置
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                if not hasattr(env, '_prim_missing_warned'):
                    env._prim_missing_warned = set()
                if prim_path not in env._prim_missing_warned:
                    print(f"[WARNING] 判定点prim不存在: {prim_path}")
                    env._prim_missing_warned.add(prim_path)
                continue
                
            try:
                xformable = UsdGeom.Xformable(prim)
                xform_ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if translate_op is None:
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                
                translate_op.Set(Gf.Vec3d(*detection_point_pos))
                
            except Exception as update_e:
                if not hasattr(env, '_update_vis_error_shown'):
                    print(f"[ERROR] 更新prim位置时出错 ({prim_path}): {update_e}")
                    env._update_vis_error_shown = True
        
    except Exception as e:
        if not hasattr(env, '_update_vis_error_shown'):
            print(f"[ERROR] 更新判定点可视化时出错: {e}")
            env._update_vis_error_shown = True