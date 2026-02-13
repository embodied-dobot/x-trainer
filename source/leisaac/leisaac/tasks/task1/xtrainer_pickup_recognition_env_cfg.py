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
TRAINING_ENV_USD_PATH = str(Path(ASSETS_ROOT) / "scenes" / "task1" / "training_env.usd")

TABLE_WITH_GOODS_CFG = AssetBaseCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRAINING_ENV_USD_PATH,
    )
)

# ------------------------------------------------------------------------------
# 3. 自定义随机化函数
# ------------------------------------------------------------------------------
def reset_nova_grid(
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
        offset_x = offset_noise[:, 0] * (cell_width * 0.25)
        offset_y = offset_noise[:, 1] * (cell_height * 0.25)
        
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

# # ------------------------------------------------------------------------------
# # 3. 观测配置
# # ------------------------------------------------------------------------------
@configclass
class ObservationsCfg(XTrainerArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """定义观测逻辑"""
        
        # [修复] 使用正则匹配所有 "good_split" 的物体
        pick_nova = ObsTerm(
            func=mdp.object_grasped, 
            params={
                # 这里的正则会匹配: Nova_J1_good_split, Nova_J4_good_split 等
                "object_cfg": SceneEntityCfg("Nova_J1_good_split"), 
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"), 
            }
        )
        
        # Good物体放入KLT_good的判定
        put_in_good_klt_j1 = ObsTerm(
            func=mdp.object_in_container, 
            params={
                "object_cfg": SceneEntityCfg("Nova_J1_good_split"), 
                "container_cfg": SceneEntityCfg("KLT_good"), 
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (0.12309, 0.17763, -0.13233),
                "verbose": True,
            }
        )
        
        put_in_good_klt_j4 = ObsTerm(
            func=mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Nova_J4_good_split"),
                "container_cfg": SceneEntityCfg("KLT_good"),
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (-0.19401, -0.41001, 0.23779),
                "verbose": True,
            }
        )
        
        put_in_good_klt_j5 = ObsTerm(
            func=mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Nova_J5_good_split"),
                "container_cfg": SceneEntityCfg("KLT_good"),
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (-0.11345, -0.00266, -0.29996),
                "verbose": True,
            }
        )
        
        # Bad物体放入KLT_bad的判定
        put_in_bad_klt_j1 = ObsTerm(
            func=mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Nova_J1_bigbad_split"),
                "container_cfg": SceneEntityCfg("KLT_bad"),
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (-0.05194, 0.5316, -0.01087),
                "verbose": True,
            }
        )
        
        put_in_bad_klt_j4 = ObsTerm(
            func=mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Nova_J4_bigbad_split"),
                "container_cfg": SceneEntityCfg("KLT_bad"),
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (0.1216, 0.10313, -0.04597),
                "verbose": True,
            }
        )
        
        put_in_bad_klt_j5 = ObsTerm(
            func=mdp.object_in_container,
            params={
                "object_cfg": SceneEntityCfg("Nova_J5_bigbad_split"),
                "container_cfg": SceneEntityCfg("KLT_bad"),
                "x_range": (-0.12, 0.12),
                "y_range": (-0.155, 0.155),
                "z_range": (-0.10, 0.10),
                "object_offset": (0.35327, 1.2108, -0.06794),
                "verbose": True,
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()

# # ------------------------------------------------------------------------------
# # 4. 终止条件 (判定成功)
# # ------------------------------------------------------------------------------
@configclass
class TerminationsCfg(XTrainerArmTerminationsCfg):

    # Task1成功判定：所有good物体在KLT_good中，所有bad物体在KLT_bad中
    success = DoneTerm(func=mdp.task1_success, params={
        "good_objects_cfg": [
            SceneEntityCfg("Nova_J1_good_split"),
            SceneEntityCfg("Nova_J4_good_split"),
            SceneEntityCfg("Nova_J5_good_split"),
        ],
        "bad_objects_cfg": [
            SceneEntityCfg("Nova_J1_bigbad_split"),
            SceneEntityCfg("Nova_J4_bigbad_split"),
            SceneEntityCfg("Nova_J5_bigbad_split"),
        ],
        "klt_good_cfg": SceneEntityCfg("KLT_good"),
        "klt_bad_cfg": SceneEntityCfg("KLT_bad"),
        "x_range": (-0.12, 0.12),
        "y_range": (-0.155, 0.155),
        "z_range": (-0.10, 0.10),
        # 物体位置偏移补偿值（用于补偿物体坐标原点不是真实中心的情况）
        "object_offsets": {
            #"Nova_J1_good_split": (0,0,0),
            "Nova_J1_good_split": (0.12309, 0.17763, -0.13233),
            # 其他物体的偏移值可以在这里添加
            "Nova_J4_good_split": (-0.19401, -0.41001, 0.23779),
            "Nova_J5_good_split": (-0.11345, -0.00266, -0.29996),
            "Nova_J1_bigbad_split": (-0.05194, 0.5316, -0.01087),
            "Nova_J4_bigbad_split": (0.1216, 0.10313, -0.04597),
            "Nova_J5_bigbad_split": (0.35327, 1.2108, -0.06794),
        },
        "verbose": True,  # 输出调试信息
        "visualize": True,  # 创建判定区域可视化（实际是否创建由enable_visualization控制）
    })
# ------------------------------------------------------------------------------
# 5. 总环境入口
# ------------------------------------------------------------------------------
@configclass
class PickupRecognitionEnvCfg(XTrainerArmTaskEnvCfg):
    """总环境配置"""

    scene: PickupRecognitionSceneCfg = PickupRecognitionSceneCfg(env_spacing=8.0)
    observations: ObservationsCfg = ObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    
    # 判定可视化开关：True=显示判定区域和判定点，False=不显示（默认关闭，可通过命令行参数 --enable_visualization 开启）
    enable_visualization: bool = False

    # observations: ObservationsCfg = None
    # terminations: TerminationsCfg = None
    
    def __post_init__(self) -> None:
        super().__post_init__()
        
        # 1. 机器人初始位置
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        # 2. 动态加载组件
        # parse_usd_and_create_subassets 会找到 USD 里外层的 Nova (因为你加了刚体，它被识别为 RigidBody)
        # 并以 'Nova' 这个名字注册进 env.scene
        # parse_usd_and_create_subassets(
        #     TRAINING_ENV_USD_PATH, 
        #     self, 
        #     specific_name_list=['Nova', 'KLT_good', 'KLT_bad']
        # )

        # 1. 修改 Time Steps Per Second (物理频率)
        # 目标: 180 Hz -> dt = 1.0 / 180.0
        self.sim.dt = 1.0 / 120.0
        self.sim.physx.enable_ccd = True

        nova_parts = [
            "Nova_J1_good_split", "Nova_J4_good_split", "Nova_J5_good_split",
            "Nova_J1_bigbad_split", "Nova_J4_bigbad_split", "Nova_J5_bigbad_split"
        ]
        
        parse_usd_and_create_subassets(
            TRAINING_ENV_USD_PATH, 
            self, 
            specific_name_list=nova_parts + ['KLT_good', 'KLT_bad']
        )
        
        # 3. 随机化
        random_opts = []
        
        # 使用网格随机化 Nova 部件
        from isaaclab.managers import EventTermCfg
        from isaaclab.managers import SceneEntityCfg
        
        random_opts.append(EventTermCfg(
            func=reset_nova_grid,
            mode="reset",
            params={
                "asset_names": nova_parts,
                "x_range": (-0.40, 0.40),
                "y_range": (0.0, 0.35),
                "grid_shape": (2, 3) # 2行3列，共6个格子
            }
        ))
            
        # # 添加 KLT 的随机化
        # random_opts.append(randomize_object_uniform("KLT_good", pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}))
        # random_opts.append(randomize_object_uniform("KLT_bad", pose_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (0.0, 0.0)}))

        # 4. 添加可视化初始化事件（在环境reset后创建可视化）
        # 必须在domain_randomization调用之前添加！
        # 从TerminationsCfg中获取偏移值（在函数外部，避免访问env.cfg）
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
            if not hasattr(env, '_task1_visualization_created'):
                print("[DEBUG] ========== 在reset事件中创建Task1判定区域可视化 ==========")
                try:
                    # 为KLT_good创建绿色可视化
                    print("[DEBUG] 开始创建KLT_good可视化...")
                    create_detection_zone_visualization(
                        env=env,
                        container_cfg=SceneEntityCfg("KLT_good"),
                        container_name="KLT_good",
                        x_range=(-0.12, 0.12),
                        y_range=(-0.155, 0.155),
                        z_range=(-0.10, 0.10),
                        color=(0.0, 1.0, 0.0),  # 绿色
                        line_width=3.0,
                    )
                    # 为KLT_bad创建红色可视化
                    print("[DEBUG] 开始创建KLT_bad可视化...")
                    create_detection_zone_visualization(
                        env=env,
                        container_cfg=SceneEntityCfg("KLT_bad"),
                        container_name="KLT_bad",
                        x_range=(-0.12, 0.12),
                        y_range=(-0.155, 0.155),
                        z_range=(-0.10, 0.10),
                        color=(1.0, 0.0, 0.0),  # 红色
                        line_width=3.0,
                    )
                    # 为所有物体创建判定点可视化
                    from .mdp.observations import create_object_detection_point_visualization
                    
                    # Good物体的判定点可视化（黄色球体）
                    good_objects = [
                        "Nova_J1_good_split",
                        "Nova_J4_good_split",
                        "Nova_J5_good_split",
                    ]
                    for obj_name in good_objects:
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
                    
                    # Bad物体的判定点可视化（橙色球体）
                    bad_objects = [
                        "Nova_J1_bigbad_split",
                        "Nova_J4_bigbad_split",
                        "Nova_J5_bigbad_split",
                    ]
                    for obj_name in bad_objects:
                        print(f"[DEBUG] 开始创建{obj_name}判定点可视化...")
                        obj_offset = object_offsets_dict.get(obj_name, (0.0, 0.0, 0.0))
                        print(f"[DEBUG] {obj_name} 使用偏移值: {obj_offset}")
                        create_object_detection_point_visualization(
                            env=env,
                            object_cfg=SceneEntityCfg(obj_name),
                            object_name=obj_name,
                            object_offset=obj_offset,
                            color=(1.0, 0.5, 0.0),  # 橙色
                            radius=0.015,  # 1.5cm半径
                        )
                    
                    env._task1_visualization_created = True
                    print("[INFO] ✓ Task1判定区域可视化已创建")
                    print("      - 绿色框=KLT_good, 红色框=KLT_bad")
                    print("      - 黄色球体=Good物体判定点, 橙色球体=Bad物体判定点")
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
        print("[DEBUG] 注意：判定点可视化会在object_in_container检查时自动更新")

        domain_randomization(self, random_options=random_opts)