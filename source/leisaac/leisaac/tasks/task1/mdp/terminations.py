from __future__ import annotations

import torch
from typing import List, Dict, Optional

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv, DirectRLEnv

from leisaac.utils.robot_utils import is_xtrainer_at_rest_pose

# 导入object_in_container函数和可视化函数
from .observations import object_in_container, create_detection_zone_visualization


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


def task1_success(
    env: ManagerBasedRLEnv | DirectRLEnv,
    good_objects_cfg: List[SceneEntityCfg],
    bad_objects_cfg: List[SceneEntityCfg],
    klt_good_cfg: SceneEntityCfg = SceneEntityCfg("KLT_good"),
    klt_bad_cfg: SceneEntityCfg = SceneEntityCfg("KLT_bad"),
    x_range: tuple[float, float] = (-0.12, 0.12),
    y_range: tuple[float, float] = (-0.155, 0.155),
    z_range: tuple[float, float] = (-0.10, 0.10),
    object_offsets: Optional[Dict[str, tuple[float, float, float]]] = None,
    verbose: bool = True,
    visualize: bool = True,
) -> torch.Tensor:
    """判定task1任务是否成功完成。
    
    成功条件：
    1. 所有带good标签的物体（Nova_J1_good_split, Nova_J4_good_split, Nova_J5_good_split）都在KLT_good中
    2. 所有带bad标签的物体（Nova_J1_bigbad_split, Nova_J4_bigbad_split, Nova_J5_bigbad_split）都在KLT_bad中
    
    Args:
        env: 环境实例
        good_objects_cfg: good物体配置列表
        bad_objects_cfg: bad物体配置列表
        klt_good_cfg: KLT_good容器配置
        klt_bad_cfg: KLT_bad容器配置
        x_range: X轴判定范围
        y_range: Y轴判定范围
        z_range: Z轴判定范围
        object_offsets: 物体位置偏移补偿值字典，格式为 {"物体名称": (x, y, z)}
        verbose: 是否输出调试信息
        visualize: 是否创建可视化（仅在第一次调用时创建）
    
    Returns:
        布尔张量，表示每个环境是否完成任务
    """
    # 添加调试输出，确认函数被调用
    if not hasattr(env, '_task1_success_called'):
        print("[DEBUG] ========== task1_success函数被调用 ==========")
        env._task1_success_called = True
    
    if object_offsets is None:
        object_offsets = {}
    
    # 创建可视化（仅在第一次调用时，且配置中启用了可视化）
    enable_vis = getattr(env.cfg, 'enable_visualization', True) if hasattr(env, 'cfg') else True
    if visualize and enable_vis and not hasattr(env, '_task1_visualization_created'):
        print("[DEBUG] ========== 开始创建Task1判定区域可视化 ==========")
        try:
            # 为KLT_good创建绿色可视化
            print(f"[DEBUG] 创建KLT_good可视化...")
            create_detection_zone_visualization(
                env=env,
                container_cfg=klt_good_cfg,
                container_name="KLT_good",
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                color=(0.0, 1.0, 0.0),  # 绿色
                line_width=3.0,
            )
            # 为KLT_bad创建红色可视化
            print(f"[DEBUG] 创建KLT_bad可视化...")
            create_detection_zone_visualization(
                env=env,
                container_cfg=klt_bad_cfg,
                container_name="KLT_bad",
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                color=(1.0, 0.0, 0.0),  # 红色
                line_width=3.0,
            )
            env._task1_visualization_created = True
            print("[INFO] ✓ Task1判定区域可视化已创建（绿色=KLT_good, 红色=KLT_bad）")
            print("[INFO] 如果看不到可视化，请在Isaac Sim中检查场景层级，路径为: /World/envs/env_X/Scene/DetectionZone_*")
        except Exception as e:
            print(f"[ERROR] 创建可视化时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 初始化所有环境都成功
    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    # 保存最近一次判定快照，避免env重置后评分为0
    if not hasattr(env, "_task1_last_in_container_state"):
        env._task1_last_in_container_state = {}
    
    # 检查所有good物体是否在KLT_good中
    for obj_cfg in good_objects_cfg:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        in_good_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_good_cfg,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=verbose,
        )
        
        done = torch.logical_and(done, in_good_klt)
        env._task1_last_in_container_state[f"{obj_name}_KLT_good"] = in_good_klt.clone()
    
    # 检查所有bad物体是否在KLT_bad中
    for obj_cfg in bad_objects_cfg:
        obj_name = obj_cfg.name
        offset = object_offsets.get(obj_name, (0.0, 0.0, 0.0))
        
        in_bad_klt = object_in_container(
            env=env,
            object_cfg=obj_cfg,
            container_cfg=klt_bad_cfg,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            object_offset=offset,
            verbose=verbose,
        )
        
        done = torch.logical_and(done, in_bad_klt)
        env._task1_last_in_container_state[f"{obj_name}_KLT_bad"] = in_bad_klt.clone()
    
    # 输出任务完成信息
    if verbose:
        for env_id in range(env.num_envs):
            if done[env_id]:
                print(f"[Env {env_id}] ✓ Task1 任务成功完成！所有物体已正确分类放置。")
    
    return done