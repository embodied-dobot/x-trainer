import torch

from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg

from leisaac.assets.scenes.kitchen import KITCHEN_WITH_ORANGE_CFG, KITCHEN_WITH_ORANGE_USD_PATH
from leisaac.utils.general_assets import parse_usd_and_create_subassets
from leisaac.utils.domain_randomization import randomize_object_uniform, randomize_camera_uniform, domain_randomization

from . import mdp
from ..template import XTrainerArmTaskSceneCfg, XTrainerArmTaskEnvCfg, XTrainerArmTerminationsCfg, XTrainerArmObservationsCfg

@configclass
class PickOrangeSceneCfg(XTrainerArmTaskSceneCfg):
    """Scene configuration for the pick orange task."""

    scene: AssetBaseCfg = KITCHEN_WITH_ORANGE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")


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
            ),
        ],
    )

@configclass
class ObservationsCfg(XTrainerArmObservationsCfg):

    @configclass
    class SubtaskCfg(ObsGroup):
        """Observations for subtask group."""
        
        pick_orange001 = ObsTerm(
            func=mdp.orange_grasped, 
            params={
                "object_cfg": SceneEntityCfg("Orange001"),
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"), 
                "grasp_threshold": 0.04
            }
        )
        put_orange001_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate, 
            params={
                "object_cfg": SceneEntityCfg("Orange001"), 
                "plate_cfg": SceneEntityCfg("Plate"),
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "grasp_threshold": 0.04
            }
        )

        pick_orange002 = ObsTerm(
            func=mdp.orange_grasped, 
            params={
                "object_cfg": SceneEntityCfg("Orange002"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "grasp_threshold": 0.04
            }
        )
        put_orange002_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate, 
            params={
                "object_cfg": SceneEntityCfg("Orange002"), 
                "plate_cfg": SceneEntityCfg("Plate"),
                "ee_frame_cfg": SceneEntityCfg("right_ee_frame"),
                "grasp_threshold": 0.04
            }
        )

        pick_orange003 = ObsTerm(
            func=mdp.orange_grasped, 
            params={
                "object_cfg": SceneEntityCfg("Orange003"),
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"), 
                "grasp_threshold": 0.04
            }
        )
        put_orange003_to_plate = ObsTerm(
            func=mdp.put_orange_to_plate, 
            params={
                "object_cfg": SceneEntityCfg("Orange003"), 
                "plate_cfg": SceneEntityCfg("Plate"),
                "ee_frame_cfg": SceneEntityCfg("left_ee_frame"),
                "grasp_threshold": 0.04
            }
        )
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(XTrainerArmTerminationsCfg):

    success = DoneTerm(func=mdp.task_done, params={
        "oranges_cfg": [SceneEntityCfg("Orange001"), SceneEntityCfg("Orange002"), SceneEntityCfg("Orange003")],
        "plate_cfg": SceneEntityCfg("Plate")
    })


@configclass
class PickOrangeEnvCfg(XTrainerArmTaskEnvCfg):
    """Configuration for the pick orange environment."""

    scene: PickOrangeSceneCfg = PickOrangeSceneCfg(env_spacing=8.0)

    observations: ObservationsCfg = ObservationsCfg()

    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.robot.init_state.pos = (2.0, -1.0, 0.1)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)

        parse_usd_and_create_subassets(KITCHEN_WITH_ORANGE_USD_PATH, self, specific_name_list=['Orange001', 'Orange002', 'Orange003', 'Plate'])

        self.scene.Plate.init_state.pos = (2.7, -1.0, 0.1) 
        self.scene.Orange001.init_state.pos = (2.3, -1.1, 0.1) 
        self.scene.Orange002.init_state.pos = (2.5, -1.0, 0.1) 
        self.scene.Orange003.init_state.pos = (2.45, -0.9, 0.1) 

        domain_randomization(self, random_options=[
            # randomize_object_uniform("Orange001", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            # randomize_object_uniform("Orange002", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            # randomize_object_uniform("Orange003", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            randomize_object_uniform("Plate", pose_range={"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)}),
            randomize_camera_uniform("top", pose_range={
                "x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.025, 0.025),
                "roll": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                "pitch": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180),
                "yaw": (-2.5 * torch.pi / 180, 2.5 * torch.pi / 180)}, convention="ros"),
        ])
