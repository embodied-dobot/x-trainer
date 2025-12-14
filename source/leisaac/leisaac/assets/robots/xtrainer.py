from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from leisaac.utils.constant import ASSETS_ROOT


"""Configuration for the Dobot X-Trainer Robot (4-Arm Configuration)."""
XTRAINER_FOLLOWER_ASSET_PATH = Path(ASSETS_ROOT) / "robots" / "x_trainer.usd"

XTRAINER_FOLLOWER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(XTRAINER_FOLLOWER_ASSET_PATH),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # position in the world
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            # --- Arm 1 (Left) ---
            "J1_1": 0.0, "J1_2": 0.0, "J1_3": 0.0,
            "J1_4": 0.0, "J1_5": 0.0, "J1_6": 0.0,
            "J1_7": 0.0, "J1_8": 0.0,

            # --- Arm 2 (Right) ---
            "J2_1": 0.0, "J2_2": 0.0, "J2_3": 0.0,
            "J2_4": 0.0, "J2_5": 0.0, "J2_6": 0.0,
            "J2_7": 0.0, "J2_8": 0.0,

            # # --- Arm 3 ---
            # "J3_1": 0.0, "J3_2": 0.0, "J3_3": 0.0,
            # "J3_4": 0.0, "J3_5": 0.0, "J3_6": 0.0,

            # # --- Arm 4 ---
            # "J4_1": 0.0, "J4_2": 0.0, "J4_3": 0.0,
            # "J4_4": 0.0, "J4_5": 0.0, "J4_6": 0.0,
        }
    ),
    
    actuators={
        "xtrainer_shoulder": ImplicitActuatorCfg(
            joint_names_expr=[
                "J1_1", "J1_2", "J1_3",
                "J2_1", "J2_2", "J2_3",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=10.0,
            stiffness=800.0,
            damping=80.0,
        ),
        "xtrainer_forearm": ImplicitActuatorCfg(
            joint_names_expr=[
                "J1_4", "J1_5", "J1_6",
                "J2_4", "J2_5", "J2_6",
            ],
            effort_limit_sim=200.0,
            velocity_limit_sim=10.0,
            stiffness=200.0,
            damping=20.0,
        ),
        "xtrainer_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "J1_7", "J1_8",
                "J2_7", "J2_8",
            ],
            effort_limit_sim=50.0,
            velocity_limit_sim=1.0,
            stiffness=400.0,  # To prevent the gripper from becoming unstable due to collisions with the robotic arm itself
            damping=40.0,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)

# joint limit written in USD (degree)
XTRAINER_FOLLOWER_USD_JOINT_LIMITS = {
    "J1_1": (-3.14159, 3.14159), "J1_2": (-3.14159, 3.14159), "J1_3": (-3.14159, 3.14159),
    "J1_4": (-3.14159, 3.14159), "J1_5": (-3.14159, 3.14159), "J1_6": (-3.14159, 3.14159),
    
    "J2_1": (-3.14159, 3.14159), "J2_2": (-3.14159, 3.14159), "J2_3": (-3.14159, 3.14159),
    "J2_4": (-3.14159, 3.14159), "J2_5": (-3.14159, 3.14159), "J2_6": (-3.14159, 3.14159),
    
    "J1_7": (-0.04, 0.0),  # 0.04 close 0 open
    "J1_8": (0.0, 0.04),
    "J2_7": (-0.04, 0.0),
    "J2_8": (0.0, 0.04),
}

# motor limit written in real device (normalized to related range)
XTRAINER_FOLLOWER_MOTOR_LIMITS = {
    "J1_1": (-3.14159, 3.14159), "J1_2": (-3.14159, 3.14159), "J1_3": (-3.14159, 3.14159),
    "J1_4": (-3.14159, 3.14159), "J1_5": (-3.14159, 3.14159), "J1_6": (-3.14159, 3.14159),
    
    "J2_1": (-3.14159, 3.14159), "J2_2": (-3.14159, 3.14159), "J2_3": (-3.14159, 3.14159),
    "J2_4": (-3.14159, 3.14159), "J2_5": (-3.14159, 3.14159), "J2_6": (-3.14159, 3.14159),

    # actual situation in the leader arm: 0 close，1 open
    # use 1.0-raw_data in xtrainer_leader.py to maintain consistency in meaning during normalization
    "J1_7": (-1.0, 0.0), "J1_8": (0.0, 1.0),
    "J2_7": (-1.0, 0.0), "J2_8": (0.0, 1.0),
}

XTRAINER_FOLLOWER_REST_POSE_RANGE = {
    "J1_1": (-5.0, 5.0),
    "J1_2": (-5.0, 5.0),
    "J1_3": (-5.0, 5.0),
    "J1_4": (-5.0, 5.0),
    "J1_5": (-5.0, 5.0),
    "J1_6": (-5.0, 5.0),
    "J2_1": (-5.0, 5.0),
    "J2_2": (-5.0, 5.0),
    "J2_3": (-5.0, 5.0),
    "J2_4": (-5.0, 5.0),
    "J2_5": (-5.0, 5.0),
    "J2_6": (-5.0, 5.0),
    "J1_7": (0.0, 0.0),
    "J1_8": (0.0, 0.0),
    "J2_7": (0.0, 0.0),
    "J2_8": (0.0, 0.0),
}

XTRAINER_FOLLOWER_REST_POSE_RANGE = {
    "J1_1": (-30.0, 30.0),
    "J1_2": (-30.0, 30.0),
    "J1_3": (-30.0, 30.0),
    "J1_4": (-30.0, 30.0),
    "J1_5": (-30.0, 30.0),
    "J1_6": (-30.0, 30.0),
    "J2_1": (-30.0, 30.0),
    "J2_2": (-30.0, 30.0),
    "J2_3": (-30.0, 30.0),
    "J2_4": (-30.0, 30.0),
    "J2_5": (-30.0, 30.0),
    "J2_6": (-30.0, 30.0),
    "J1_7": (-30.0, 30.0),
    "J1_8": (-30.0, 30.0),
    "J2_7": (-30.0, 30.0),
    "J2_8": (-30.0, 30.0),
}
