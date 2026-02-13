import torch
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
# 通用放置检测 (检查物体是否在容器上方指定区域)
# ==============================================================================
def object_in_container(
        env: ManagerBasedRLEnv | DirectRLEnv,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        container_cfg: SceneEntityCfg = SceneEntityCfg("container"),
        # 容器的有效范围 (Box check)
        x_range: tuple[float, float] = (-0.20, 0.20), 
        y_range: tuple[float, float] = (-0.02, 0.02),
        z_range: tuple[float, float] = (0.10, 0.15),
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
# 可视化判定区域
# ==============================================================================
def create_detection_zone_visualization(
    env: ManagerBasedRLEnv | DirectRLEnv,
    container_cfg: SceneEntityCfg,
    container_name: str,
    x_range: tuple[float, float] = (-0.20, 0.20),
    y_range: tuple[float, float] = (-0.02, 0.02),
    z_range: tuple[float, float] = (0.10, 0.15),
    color: tuple[float, float, float] = (1.0, 0.0, 0.0),  # 默认红色
    line_width: float = 2.0,
) -> None:
    """在Isaac Sim中创建判定区域的可视化线框盒子。
    
    使用Cube几何体并设置为wireframe模式来显示判定区域。
    
    Args:
        env: 环境实例
        container_cfg: 容器配置
        container_name: 容器名称（用于命名可视化对象）
        x_range: X轴范围
        y_range: Y轴范围
        z_range: Z轴范围
        color: RGB颜色 (0-1范围)
        line_width: 线宽（像素）
    """
    if not USD_AVAILABLE:
        print("[WARNING] USD API不可用，无法创建可视化")
        return
    
    try:
        stage = env.sim.stage
        container: RigidObject = env.scene[container_cfg.name]
        
        # 获取容器的prim路径
        try:
            container_prim_path = container.root_physx_view.prim_paths[0]
        except:
            # 如果无法获取，尝试从场景中查找
            container_prim_path = None
            for env_id in range(min(1, env.num_envs)):  # 只检查第一个环境
                test_path = f"/World/envs/env_{env_id}/Scene/{container_cfg.name}"
                if stage.GetPrimAtPath(test_path).IsValid():
                    container_prim_path = test_path
                    break
            
            if container_prim_path is None:
                print(f"[ERROR] 无法找到容器 {container_cfg.name} 的prim路径")
                return
        
        # 从容器路径提取环境命名空间
        # 例如: /World/envs/env_0/Scene/rack -> /World/envs/env_0
        path_parts = container_prim_path.split("/")
        if len(path_parts) >= 4:
            env_base_path = "/".join(path_parts[:4])  # /World/envs/env_0
        else:
            env_base_path = "/World/envs/env_0"
        
        # 计算盒子尺寸和中心
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        
        box_size_x = x_max - x_min
        box_size_y = y_max - y_min
        box_size_z = z_max - z_min
        
        box_center_x = (x_min + x_max) / 2.0
        box_center_y = (y_min + y_max) / 2.0
        box_center_z = (z_min + z_max) / 2.0
        
        # 为每个环境创建可视化
        for env_id in range(env.num_envs):
            # 获取容器位置
            container_pos = container.data.root_pos_w[env_id].cpu().numpy()
            
            # 计算可视化盒子的世界位置（容器位置 + 相对中心）
            vis_pos = (
                container_pos[0] + box_center_x,
                container_pos[1] + box_center_y,
                container_pos[2] + box_center_z,
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
            
            # 设置尺寸（使用实际的盒子尺寸）
            cube.GetSizeAttr().Set(1.0)  # 先设置为1，然后通过scale来调整
            
            # 设置缩放以匹配实际尺寸
            cube_prim = cube.GetPrim()
            xformable = UsdGeom.Xformable(cube_prim)
            xformable.ClearXformOpOrder()
            scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
            scale_op.Set(Gf.Vec3f(box_size_x, box_size_y, box_size_z))
            
            # 设置颜色（使用displayColor）
            color_attr = cube.CreateDisplayColorAttr()
            color_attr.Set([Gf.Vec3f(*color)])
            
            # 创建半透明材质以便更好地观察
            if UsdShade is not None:
                try:
                    material_path = f"{prim_path}/Material"
                    material = UsdShade.Material.Define(stage, material_path)
                    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", UsdShade.Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                    shader.CreateInput("roughness", UsdShade.Sdf.ValueTypeNames.Float).Set(0.1)
                    shader.CreateInput("metallic", UsdShade.Sdf.ValueTypeNames.Float).Set(0.0)
                    # 设置透明度（0.3 = 70%透明）
                    shader.CreateInput("opacity", UsdShade.Sdf.ValueTypeNames.Float).Set(0.3)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    
                    # 绑定材质到cube
                    UsdShade.MaterialBindingAPI(cube_prim).Bind(material)
                except Exception as mat_e:
                    print(f"[WARNING] 设置材质时出错（可忽略）: {mat_e}")
            
            # 打印调试信息
            print(f"[INFO] 创建可视化Cube: {cube_path}")
            print(f"      - 位置: ({vis_pos[0]:.3f}, {vis_pos[1]:.3f}, {vis_pos[2]:.3f})")
            print(f"      - 尺寸: ({box_size_x:.3f}, {box_size_y:.3f}, {box_size_z:.3f})")
            print(f"      - 颜色: RGB({color[0]:.1f}, {color[1]:.1f}, {color[2]:.1f})")
        
        print(f"[INFO] 已为容器 {container_name} 创建判定区域可视化（{env.num_envs}个环境）")
        
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
            # 如果无法获取，尝试从场景中查找
            object_prim_path = None
            for env_id in range(min(1, env.num_envs)):  # 只检查第一个环境
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
            env_base_path = "/".join(path_parts[:4])  # /World/envs/env_0
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
                    # 不透明
                    shader.CreateInput("opacity", UsdShade.Sdf.ValueTypeNames.Float).Set(1.0)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    
                    # 绑定材质到sphere
                    sphere_prim = sphere.GetPrim()
                    UsdShade.MaterialBindingAPI(sphere_prim).Bind(material)
                except Exception as mat_e:
                    print(f"[WARNING] 设置材质时出错（可忽略）: {mat_e}")
            
            # 打印调试信息
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
        object_offset = vis_info['object_offset']  # 默认使用存储的值
        try:
            if (hasattr(env, 'cfg') and 
                hasattr(env.cfg, 'terminations') and 
                env.cfg.terminations.success is not None and
                hasattr(env.cfg.terminations.success, 'params')):
                object_offsets = env.cfg.terminations.success.params.get("object_offsets", {})
                if object_name in object_offsets:
                    # 使用配置中的最新偏移值
                    new_offset = object_offsets[object_name]
                    # 检查偏移值是否改变
                    if new_offset != object_offset:
                        if not hasattr(env, '_offset_change_shown'):
                            env._offset_change_shown = {}
                        if object_name not in env._offset_change_shown:
                            print(f"[DEBUG] 检测到偏移值变化 ({object_name}): {object_offset} -> {new_offset}")
                            env._offset_change_shown[object_name] = True
                        object_offset = new_offset
                        # 同时更新存储的值
                        vis_info['object_offset'] = object_offset
        except:
            # 如果无法从配置读取，使用存储的值
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
                # 如果prim不存在，输出警告（但不删除，因为可能只是还没创建）
                if not hasattr(env, '_prim_missing_warned'):
                    env._prim_missing_warned = set()
                if prim_path not in env._prim_missing_warned:
                    print(f"[WARNING] 判定点prim不存在: {prim_path}")
                    print(f"         物体位置: ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})")
                    print(f"         偏移值: {object_offset}")
                    print(f"         计算位置: ({detection_point_pos[0]:.3f}, {detection_point_pos[1]:.3f}, {detection_point_pos[2]:.3f})")
                    env._prim_missing_warned.add(prim_path)
                continue
                
            try:
                xformable = UsdGeom.Xformable(prim)
                # 获取现有的translate操作
                xform_ops = xformable.GetOrderedXformOps()
                translate_op = None
                for op in xform_ops:
                    if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                        translate_op = op
                        break
                
                if translate_op is None:
                    # 如果没有translate操作，创建一个
                    xformable.ClearXformOpOrder()
                    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
                
                # 更新位置（直接设置，不使用EditContext）
                translate_op.Set(Gf.Vec3d(*detection_point_pos))
                
                # 添加调试输出（仅第一次）
                if not hasattr(env, '_update_debug_shown'):
                    env._update_debug_shown = set()
                if f"{object_name}_{env_id}" not in env._update_debug_shown:
                    print(f"[DEBUG] 更新判定点位置 ({object_name}, Env {env_id}):")
                    print(f"         物体位置: ({object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f})")
                    print(f"         偏移值: {object_offset}")
                    print(f"         判定点位置: ({detection_point_pos[0]:.3f}, {detection_point_pos[1]:.3f}, {detection_point_pos[2]:.3f})")
                    print(f"         Prim路径: {prim_path}")
                    env._update_debug_shown.add(f"{object_name}_{env_id}")
                
            except Exception as update_e:
                # 输出更新错误，但不影响主流程
                if not hasattr(env, '_update_vis_error_shown'):
                    print(f"[ERROR] 更新prim位置时出错 ({prim_path}): {update_e}")
                    import traceback
                    traceback.print_exc()
                    env._update_vis_error_shown = True
        
    except Exception as e:
        # 输出错误以便调试
        if not hasattr(env, '_update_vis_error_shown'):
            print(f"[ERROR] 更新判定点可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            env._update_vis_error_shown = True