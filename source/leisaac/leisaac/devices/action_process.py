import torch
from typing import Any

import isaaclab.envs.mdp as mdp

from leisaac.assets.robots.lerobot import SO101_FOLLOWER_USD_JOINT_LIMLITS
from leisaac.assets.robots.xtrainer import XTRAINER_FOLLOWER_USD_JOINT_LIMITS

def init_action_cfg(action_cfg, device):
    if device in ['so101leader']:
        action_cfg.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['keyboard']:
        action_cfg.arm_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    elif device in ['bi-so101leader']:
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="left_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="right_arm",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['xtrainerleader']:
        # Left Arm + Gripper
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J1_1", "J1_2", "J1_3", "J1_4", "J1_5", "J1_6"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J1_7", "J1_8"],
            scale=1.0,
        )
        # Right Arm + Gripper
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J2_1", "J2_2", "J2_3", "J2_4", "J2_5", "J2_6"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J2_7", "J2_8"],
            scale=1.0,
        )
    elif device in ['xtrainer_vr']:
        # Left Arm + Gripper
        # IK, input: [x, y, z, qw, qx, qy, qz]
        action_cfg.left_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["J1_1", "J1_2", "J1_3", "J1_4", "J1_5", "J1_6"],
            body_name="J1_6", 
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose", 
                ik_method="dls", 
                use_relative_mode=False
            ),
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J1_7", "J1_8"],
            scale=1.0,
        )
        # Right Arm + Gripper
        action_cfg.right_arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["J2_1", "J2_2", "J2_3", "J2_4", "J2_5", "J2_6"],
            body_name="J2_6",
            controller=mdp.DifferentialIKControllerCfg(
                command_type="pose", 
                ik_method="dls", 
                use_relative_mode=False
            ),
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J2_7", "J2_8"],
            scale=1.0,
        )
    elif device in ['bi_keyboard']:
        # Left Arm + Gripper
        action_cfg.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J1_1", "J1_2", "J1_3", "J1_4", "J1_5", "J1_6"],
            scale=1.0,
        )
        action_cfg.left_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J1_7", "J1_8"],
            scale=0.7,
        )
        # Right Arm + Gripper
        action_cfg.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J2_1", "J2_2", "J2_3", "J2_4", "J2_5", "J2_6"],
            scale=1.0,
        )
        action_cfg.right_gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["J2_7", "J2_8"],
            scale=0.7,
        )
    elif device in ['mimic_so101leader']:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=1.0,
        )
    elif device in ['mimic_keyboard']:
        action_cfg.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            body_name="gripper",
            controller=mdp.DifferentialIKControllerCfg(command_type="pose", ik_method="dls", use_relative_mode=False),
        )
        action_cfg.gripper_action = mdp.RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            scale=0.7,
        )
    else:
        action_cfg.arm_action = None
        action_cfg.gripper_action = None
    return action_cfg


joint_names_to_motor_ids = {
    "shoulder_pan": 0,
    "shoulder_lift": 1,
    "elbow_flex": 2,
    "wrist_flex": 3,
    "wrist_roll": 4,
    "gripper": 5,
}

xtrainer_joint_names_to_motor_ids = {
    "J1_1": 0, "J1_2": 1, "J1_3": 2, "J1_4": 3, "J1_5": 4, "J1_6": 5,
    "J1_7": 6, "J1_8": 7,
    
    "J2_1": 8, "J2_2": 9, "J2_3": 10, "J2_4": 11, "J2_5": 12, "J2_6": 13,
    "J2_7": 14, "J2_8": 15,
}


def convert_action_from_so101_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
    joint_limits = SO101_FOLLOWER_USD_JOINT_LIMLITS
    for joint_name, motor_id in joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        processed_degree = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
            * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
        processed_radius = processed_degree / 180.0 * torch.pi  # convert degree to radius
        processed_action[:, motor_id] = processed_radius
    return processed_action

def convert_action_from_xtrainer_leader(joint_state: dict[str, float], motor_limits: dict[str, tuple[float, float]], teleop_device) -> torch.Tensor:
    """
    Real (6+1 DoF) -> Sim (6+2 DoF)
    """
    # Tensor: [LeftArm(6), LeftGrip(2), RightArm(6), RightGrip(2)] -> Total 16
    processed_action = torch.zeros(teleop_device.env.num_envs, 16, device=teleop_device.env.device)
    joint_limits = XTRAINER_FOLLOWER_USD_JOINT_LIMITS
    for joint_name, motor_id in xtrainer_joint_names_to_motor_ids.items():
        motor_limit_range = motor_limits[joint_name]
        joint_limit_range = joint_limits[joint_name]
        processed_radius = (joint_state[joint_name] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
            * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]

        processed_action[:, motor_id] = processed_radius
    
    return processed_action

def convert_action_from_xtrainer_vr(joint_state: dict[str, float], teleop_device) -> torch.Tensor:
    processed_action = torch.zeros(teleop_device.env.num_envs, 18, device=teleop_device.env.device)

    joint_limits = XTRAINER_FOLLOWER_USD_JOINT_LIMITS
    if joint_state.get('left') is not None:
        vr_data = torch.tensor(joint_state['left'], device=teleop_device.env.device)
        processed_action[:, 0:7] = vr_data[0:7]
        
        # gripper: (0.0 = Open, 1.0 = Closed)-->(0.0 = Open, 0.04 = Closed)
        processed_action[:, 8] = vr_data[7] * (joint_limits["J1_8"][1] - joint_limits["J1_8"][0]) + joint_limits["J1_8"][0]
        processed_action[:, 7] = -processed_action[:, 8]

    if joint_state.get('right') is not None:
        vr_data = torch.tensor(joint_state['right'], device=teleop_device.env.device)
        processed_action[:, 9:16] = vr_data[0:7]
        
        # gripper: (0.0 = Open, 1.0 = Closed)-->(0.0 = Open, 0.04 = Closed)
        processed_action[:, 17] = vr_data[7] * (joint_limits["J2_8"][1] - joint_limits["J2_8"][0]) + joint_limits["J2_8"][0]
        processed_action[:, 16] = -processed_action[:, 17]
        
    return processed_action

def preprocess_device_action(action: dict[str, Any], teleop_device) -> torch.Tensor:
    if action.get('so101_leader') is not None:
        processed_action = convert_action_from_so101_leader(action['joint_state'], action['motor_limits'], teleop_device)
    elif action.get('keyboard') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 6, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('bi_so101_leader') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 12, device=teleop_device.env.device)
        processed_action[:, :6] = convert_action_from_so101_leader(action['joint_state']['left_arm'], action['motor_limits']['left_arm'], teleop_device)
        processed_action[:, 6:] = convert_action_from_so101_leader(action['joint_state']['right_arm'], action['motor_limits']['right_arm'], teleop_device)
    elif action.get('xtrainer_leader') is not None:
        processed_action = convert_action_from_xtrainer_leader(
                                                              action['joint_state'], 
                                                              action['motor_limits'], 
                                                              teleop_device
                                                            )
        # print(processed_action)
    elif action.get('bi_keyboard') is not None:
        processed_action = torch.zeros(teleop_device.env.num_envs, 16, device=teleop_device.env.device)
        processed_action[:, :] = action['joint_state']
    elif action.get('xtrainer_vr') is not None:
        processed_action = convert_action_from_xtrainer_vr(
                                                           action['joint_state'], 
                                                           teleop_device
                                                         )
    else:
        raise NotImplementedError("Only teleoperation with so101_leader, bi_so101_leader, keyboard, xtrainer_leader, bi_keyboard, xtrainer_vr is supported for now.")
    return processed_action