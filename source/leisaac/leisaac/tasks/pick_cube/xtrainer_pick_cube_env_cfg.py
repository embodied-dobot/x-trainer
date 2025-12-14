import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg

from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization

from . import mdp
from ..template import XTrainerArmTaskSceneCfg, XTrainerArmTaskEnvCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg

@configclass
class PickCubeSceneCfg(XTrainerArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6", 
                name="left_flange"
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J1_6", 
                name="left_grasp_center",
                # pos=(0.0, 0.0, -0.16),
            ),
            
        ],
    )
    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6", 
                name="right_flange"
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/x_trainer_asm_0226_SLDASM/J2_6", 
                name="right_grasp_center",
                # pos=(0.0, 0.0, -0.16),
            ),
        ],
    )

@configclass
class ObservationsCfg(XTrainerArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        
        pick_cube = ObsTerm(
            func=mdp.object_grasped, 
            params={
                "object_cfg": SceneEntityCfg("cube"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"), 
            }
        )
        
        put_cube_to_plate = ObsTerm(
            func=mdp.put_cube_to_plate, 
            params={
                "object_cfg": SceneEntityCfg("cube"), 
                "plate_cfg": SceneEntityCfg("Plate"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
            }
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(XTrainerArmTerminationsCfg):

    success = DoneTerm(func=mdp.task_done, params={
        "objects_cfg": [SceneEntityCfg("cube")],
        "plate_cfg": SceneEntityCfg("Plate")
    })


@configclass
class PickCubeEnvCfg(XTrainerArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickCubeSceneCfg = PickCubeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()
    
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self, 
                                       specific_name_list=['cube', 'Plate'])

        self.scene.Plate.init_state.pos = (0.7, 0.0, 0.1) 
        self.scene.cube.init_state.pos = (0.5, 0.0, 0.1)

        domain_randomization(self, random_options=[
            randomize_object_uniform("cube", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            # randomize_camera_uniform("top", pose_range={
            #     "x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025),
            #     "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
            #     "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
            #     "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180)}, convention="ros"),
        ])
