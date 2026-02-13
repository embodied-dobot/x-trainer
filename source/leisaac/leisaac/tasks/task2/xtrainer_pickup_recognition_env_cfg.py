import torch
from pathlib import Path

# Isaac Lab 核心
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
import isaaclab.sim as sim_utils

# 项目工具
from leisaac.utils.constant import ASSETS_ROOT
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, domain_randomization

# 引用当前目录下的 mdp
from . import mdp
from .mdp.observations import create_detection_zone_visualization
# 引用模板
from ..template import XTrainerArmTaskSceneCfg, XTrainerArmTaskEnvCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg

# ------------------------------------------------------------------------------
# 1. 路径指向你的训练场景
# ------------------------------------------------------------------------------
# 确保这个路径下真的有 training_env.usd 文件
TRAINING_ENV_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "task2" / "training_env.usd")

TABLE_WITH_GOODS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAINING_ENV_USD_PATH,
    )
)

# ------------------------------------------------------------------------------
# 2. 场景配置
# ------------------------------------------------------------------------------
@configclass
class PickupRecognitionSceneCfg(XTrainerArmTaskSceneCfg):
    """场景配置"""
    # 加载 USD
    scene: AssetBaseCfg = TABLE_WITH_GOODS_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    # 定义双臂末端 (保持不变)
    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6", 
                name="left_flange"
            ),
        ],
    )
    
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6", 
                name="right_grasp_center",
            ),
        ],
    )

# ------------------------------------------------------------------------------
# 3. 观测配置
# ------------------------------------------------------------------------------
@configclass
class ObservationsCfg(XTrainerArmObservationsCfg):
    """观测配置"""
    
    @configclass
    class SubtaskCfg(ObsGroup):
        """定义观测逻辑（用于更新可视化）"""
        
        # 为每个物体创建观测项，用于更新判定点可视化
        # 这些观测项每个步骤都会被调用，从而更新可视化球体位置
        update_emergency_stop_button = ObsTerm(
            func=mdp.update_task2_detection_point,
            params={
                "object_cfg": SceneEntityCfg("Emergency_Stop_Button"),
                "object_offset": (0.0, 0.0, 0.0),
                "verbose": False,
            }
        )
        
        update_nut = ObsTerm(
            func=mdp.update_task2_detection_point,
            params={
                "object_cfg": SceneEntityCfg("factory_nut_loose"),
                "object_offset": (0.0, 0.0, 0.0),
                "verbose": False,
            }
        )
        
        update_chip = ObsTerm(
            func=mdp.update_task2_detection_point,
            params={
                "object_cfg": SceneEntityCfg("chip"),
                "object_offset": (0.0, 0.0, 0.0),
                "verbose": False,
            }
        )
        
        # 检测物体是否在两个判定区的交集中，并输出信息
        # 注意：object_offset会在函数内部从配置中读取，这里只是占位
        intersection_emergency_stop = ObsTerm(
            func=mdp.object_in_task2_intersection,
            params={
                "object_cfg": SceneEntityCfg("Emergency_Stop_Button"),
                "protectlid_cfg": SceneEntityCfg("protectlid"),
                "rect_position": (0.689, -0.262, 0.113),
                "rect_size": (0.144, 0.144, 0.050),
                "protectlid_zone_size": (0.18, 0.24, 0.040),  # 蓝色判定区大小
                "object_offset": (0.0, 0.0, 0.0),  # 会在函数内部从配置读取
                "verbose": True,
            }
        )
        
        intersection_nut = ObsTerm(
            func=mdp.object_in_task2_intersection,
            params={
                "object_cfg": SceneEntityCfg("factory_nut_loose"),
                "protectlid_cfg": SceneEntityCfg("protectlid"),
                "rect_position": (0.689, -0.262, 0.113),
                "rect_size": (0.144, 0.144, 0.050),
                "protectlid_zone_size": (0.18, 0.24, 0.040),  # 蓝色判定区大小
                "object_offset": (0.0, 0.0, 0.0),  # 会在函数内部从配置读取
                "verbose": True,
            }
        )
        
        intersection_chip = ObsTerm(
            func=mdp.object_in_task2_intersection,
            params={
                "object_cfg": SceneEntityCfg("chip"),
                "protectlid_cfg": SceneEntityCfg("protectlid"),
                "rect_position": (0.689, -0.262, 0.113),
                "rect_size": (0.144, 0.144, 0.050),
                "protectlid_zone_size": (0.18, 0.24, 0.040),  # 蓝色判定区大小
                "object_offset": (0.0, 0.0, 0.0),  # 会在函数内部从配置读取
                "verbose": True,
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    
    subtask_terms: SubtaskCfg = SubtaskCfg()

# ------------------------------------------------------------------------------
# 4. 终止条件 (判定成功)
# ------------------------------------------------------------------------------
@configclass
class TerminationsCfg(XTrainerArmTerminationsCfg):
    """终止条件配置"""
    
    # Task2成功判定：Emergency_Stop_Button、nut、chip的判定点都被放置在指定矩形区域
    # 且被protectlid下方的区域盖住
    success = DoneTerm(func=mdp.task2_success, params={
        "objects_cfg": [
            SceneEntityCfg("Emergency_Stop_Button"),
            SceneEntityCfg("factory_nut_loose"),  # nut
            SceneEntityCfg("chip"),
        ],
        "protectlid_cfg": SceneEntityCfg("protectlid"),
        # 矩形区域参数（位置: (0.689, -0.262, 0.113)，尺寸: (0.144, 0.144, 0.050)）
        "rect_position": (0.689, -0.262, 0.113),
        "rect_size": (0.144, 0.144, 0.050),
        # 物体位置偏移补偿值（坐标系偏差默认设置为0）
        "object_offsets": {
            "Emergency_Stop_Button": (0,0.05166,0.31674),
            "factory_nut_loose": (0.0, 0.0, 0.02),
            "chip": (0.0, 0.0, 0.0),
        },
        "verbose": True,  # 输出调试信息
        "visualize": True,  # 创建判定区域可视化（实际是否创建由enable_visualization控制）
    })

# ------------------------------------------------------------------------------
# 5. 总环境入口
# ------------------------------------------------------------------------------
@configclass
class Task2EnvCfg(XTrainerArmTaskEnvCfg):
    """总环境配置"""

    scene: PickupRecognitionSceneCfg = PickupRecognitionSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # 判定可视化开关：True=显示判定区域和判定点，False=不显示（默认关闭，可通过命令行参数 --enable_visualization 开启）
    enable_visualization: bool = False
    
    def __post_init__(self) -> None:
        super().__post_init__()
        
        # 1. 机器人初始位置
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # 1. 修改 Time Steps Per Second (物理频率)
        # 目标: 180 Hz -> dt = 1.0 / 180.0
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True
        # 2. 动态加载组件
        # parse_usd_and_create_subassets 会找到 USD 里外层的 Nova (因为你加了刚体，它被识别为 RigidBody)
        # 并以 'Nova' 这个名字注册进 env.scene
        # parse_usd_and_create_subassets(
        #     TRAINING_ENV_USD_PATH, 
        #     self, 
        #     specific_name_list=['Nova', 'KLT_good', 'KLT_bad']
        # )

        nova_parts = [
            "dustpan", "Emergency_Stop_Button", "protectlid", "chip", "factory_nut_loose"
        ]
        trash_parts = [
            "Emergency_Stop_Button", "chip", "factory_nut_loose"
        ]
        
        parse_usd_and_create_subassets(
            TRAINING_ENV_USD_PATH, 
            self, 
            specific_name_list=nova_parts
        )
        
        # 3. 随机化
        random_opts = []
        
        # 使用网格随机化部件
        from isaaclab.managers import EventTermCfg
        
        random_opts.append(EventTermCfg(
            func=reset_tube_grid,
            mode="reset",
            params={
                "asset_names": trash_parts,
                "x_range": (-0.3, 0.1),
                "y_range": (-0.1, 0.15),
                "grid_shape": (2, 3) # 2行3列，共6个格子
            }
        ))
        
        # 4. 添加可视化初始化事件（在环境reset后创建可视化）
        # 必须在domain_randomization调用之前添加！
        # 从TerminationsCfg中获取偏移值
        terminations_cfg = self.terminations
        object_offsets_dict = {}
        if (hasattr(terminations_cfg, 'success') and 
            hasattr(terminations_cfg.success, 'params') and
            'object_offsets' in terminations_cfg.success.params):
            object_offsets_dict = terminations_cfg.success.params['object_offsets']
        
        def init_visualization(env, env_ids):
            """在环境reset后创建可视化"""
            # 检查可视化开关
            enable_vis = getattr(env.cfg, 'enable_visualization', True)
            if not enable_vis:
                return  # 如果关闭可视化，直接返回
            
            print(f"[DEBUG] ========== init_visualization函数被调用！env_ids={env_ids} ==========")
            if not hasattr(env, '_task2_visualization_created'):
                print("[DEBUG] ========== 在reset事件中创建Task2判定区域可视化 ==========")
                try:
                    # 获取矩形区域参数
                    rect_position = (0.689, -0.262, 0.113)
                    rect_size = (0.144, 0.144, 0.050)
                    
                    # 创建矩形区域可视化（绿色）
                    print("[DEBUG] 开始创建目标矩形区域可视化...")
                    create_detection_zone_visualization(
                        env=env,
                        container_cfg=None,  # 使用绝对位置
                        container_name="target_rect",
                        position=rect_position,
                        size=rect_size,
                        color=(0.0, 1.0, 0.0),  # 绿色
                        line_width=3.0,
                    )
                    
                    # 获取protectlid位置并创建protectlid下方区域可视化（蓝色）
                    protectlid = env.scene["protectlid"]
                    protectlid_pos = protectlid.data.root_pos_w[0].cpu().numpy()  # 使用第一个环境的位置作为参考
                    protectlid_zone_size = (0.18, 0.24, 0.040)  # 区域大小
                    protectlid_zone_pos = (
                        protectlid_pos[0],
                        protectlid_pos[1],
                        # protectlid_pos[2] + 0.5,
                        protectlid_pos[2] - protectlid_zone_size[2] / 2.0 + 0.05,  # protectlid下方
                    )
                    print("[DEBUG] 开始创建protectlid下方区域可视化...")
                    create_detection_zone_visualization(
                        env=env,
                        container_cfg=None,
                        container_name="protectlid_zone",
                        position=protectlid_zone_pos,
                        size=protectlid_zone_size,
                        color=(0.0, 0.0, 1.0),  # 蓝色
                        line_width=3.0,
                    )
                    
                    # 为所有物体创建判定点可视化
                    from .mdp.observations import create_object_detection_point_visualization
                    
                    objects = [
                        "Emergency_Stop_Button",
                        "factory_nut_loose",
                        "chip",
                    ]
                    for obj_name in objects:
                        print(f"[DEBUG] 开始创建{obj_name}判定点可视化...")
                        obj_offset = object_offsets_dict.get(obj_name, (0.0, 0.0, 0.0))
                        print(f"[DEBUG] {obj_name} 使用偏移值: {obj_offset}")
                        create_object_detection_point_visualization(
                            env=env,
                            object_cfg=SceneEntityCfg(obj_name),
                            object_name=obj_name,
                            object_offset=obj_offset,
                            color=(1.0, 1.0, 0.0),  # 黄色
                            radius=0.015,  # 1.5cm半径
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
                    
                    env._task2_visualization_created = True
                    print("[INFO] ✓ Task2判定区域可视化已创建")
                    print("      - 绿色框=目标矩形区域（固定位置）")
                    print("      - 蓝色框=protectlid下方区域（跟随protectlid移动）")
                    print("      - 黄色球体=物体判定点")
                except Exception as e:
                    print(f"[ERROR] 创建可视化时出错: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("[DEBUG] 可视化已创建，跳过...")
        
        random_opts.append(EventTermCfg(
            func=init_visualization,
            mode="reset",
            params={}
        ))
        
        print("[DEBUG] ========== 已添加可视化初始化事件到random_opts（共{}个事件）==========".format(len(random_opts)))
        print("[DEBUG] 注意：判定点可视化会在task2_success检查时自动更新")

        domain_randomization(self, random_options=random_opts)
# ------------------------------------------------------------------------------
def reset_tube_grid(
    env, 
    env_ids: torch.Tensor, 
    asset_names: list[str], 
    x_range: tuple[float, float], 
    y_range: tuple[float, float], 
    grid_shape: tuple[int, int] = (2, 3)
):
    """
    将多个 Nova 部件随机分配到网格中，防止重叠。
    
    Args:
        env: 环境对象
        env_ids: 需要重置的环境 ID
        asset_names: Nova 部件的名称列表
        x_range: X 轴范围 (min, max)
        y_range: Y 轴范围 (min, max)
        grid_shape: 网格形状 (rows, cols)，默认 (2, 3) 共 6 个格子
    """
    num_envs = env.num_envs
    # 如果 env_ids 为 None，则重置所有环境 (虽然通常 MDP 会传入具体的 env_ids)
    if env_ids is None:
        env_ids = torch.arange(num_envs, device=env.device)
    
    num_reset = len(env_ids)
    rows, cols = grid_shape
    num_cells = rows * cols
    
    if len(asset_names) > num_cells:
        raise ValueError(f"Asset count ({len(asset_names)}) exceeds grid cells ({num_cells})!")

    # 1. 计算网格中心点
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # 每个格子的尺寸
    cell_width = (x_max - x_min) / cols
    cell_height = (y_max - y_min) / rows
    
    # 生成所有格子的中心点坐标 (num_cells, 2)
    # x 对应 cols, y 对应 rows
    # 注意：这里假设 x 是横向，y 是纵向，或者反之，根据 range 决定
    # 通常 Isaac Sim 中 Z 是向上，XY 是平面
    
    # 生成网格索引
    # x_indices: [0, 1, 2, 0, 1, 2]
    # y_indices: [0, 0, 0, 1, 1, 1]
    x_indices = torch.arange(cols, device=env.device).repeat(rows)
    y_indices = torch.arange(rows, device=env.device).repeat_interleave(cols)
    
    # 计算中心点
    # x_center = x_min + (x_idx + 0.5) * width
    grid_centers_x = x_min + (x_indices + 0.5) * cell_width
    grid_centers_y = y_min + (y_indices + 0.5) * cell_height
    
    # stack 得到 (num_cells, 2) -> (x, y)
    grid_centers = torch.stack([grid_centers_x, grid_centers_y], dim=-1)
    
    # 2. 为每个环境分配格子
    # 我们需要为每个环境的每个部件分配一个不重复的格子索引
    # 策略：对每个环境，生成 0~num_cells-1 的随机排列，取前 len(asset_names) 个
    
    # rand_perm: (num_reset, num_cells)
    # argsort 得到随机排列的索引
    rand_noise = torch.rand((num_reset, num_cells), device=env.device)
    cell_indices = torch.argsort(rand_noise, dim=-1)[:, :len(asset_names)] # (num_reset, num_assets)
    
    # 3. 设置每个部件的位置
    for i, asset_name in enumerate(asset_names):
        asset = env.scene[asset_name]
        
        # 获取当前部件在所有重置环境中的目标格子索引
        # target_cell_indices: (num_reset,)
        target_cell_indices = cell_indices[:, i]
        
        # 获取格子中心: (num_reset, 2)
        centers = grid_centers[target_cell_indices]
        
        # 在格子内添加随机偏移 (比如 +/- 25% 的格子大小，留出 50% 间隙)
        # rand shape: (num_reset, 2) -> range [-1, 1]
        offset_noise = 2 * torch.rand((num_reset, 2), device=env.device) - 1
        offset_x = offset_noise[:, 0] * (cell_width * 0.15)
        offset_y = offset_noise[:, 1] * (cell_height * 0.15)
        
        # 最终 XY 偏移量 (相对于 default_root_state)
        final_offset_x = centers[:, 0] + offset_x
        final_offset_y = centers[:, 1] + offset_y
        
        # 获取当前 root state
        # root_state: (num_envs, 13) -> [pos(3), rot(4), lin_vel(3), ang_vel(3)]
        # 注意：我们需要只更新 env_ids 对应的行
        root_state = asset.data.default_root_state[env_ids].clone()
        
        # 更新位置 (x, y, z)
        # 叠加到默认位置上 (因为 x_range/y_range 是相对于默认位置的偏移范围)
        root_state[:, 0] += final_offset_x
        root_state[:, 1] += final_offset_y
        # root_state[:, 2] 保持默认
        
        # 重置速度为 0
        root_state[:, 7:] = 0.0
        
        # 应用状态
        asset.write_root_state_to_sim(root_state, env_ids=env_ids)
