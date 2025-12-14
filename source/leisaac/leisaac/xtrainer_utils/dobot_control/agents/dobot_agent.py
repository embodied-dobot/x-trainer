import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .agent import Agent
from ..robots.dynamixel import DynamixelRobot
import time
import configparser
import os


@dataclass
class DobotRobotConfig:
    joint_ids: Sequence[int]
    append_id: int
    baud_rate: int
    port: str
    joint_offsets: Sequence[float]
    joint_signs: Sequence[int]
    gripper_config: Tuple[int, int, int]
    start_joints: Sequence[float]
    using_sensor: int

    def __post_init__(self):
        assert len(self.joint_ids) == len(self.joint_offsets)
        assert len(self.joint_ids) == len(self.joint_signs)

    def make_robot(self, start_joints: Optional[np.ndarray] = None) -> DynamixelRobot:
        return DynamixelRobot(
            joint_ids=self.joint_ids,
            append_id=self.append_id,
            joint_offsets=list(self.joint_offsets),
            real=True,
            joint_signs=list(self.joint_signs),
            port=self.port,
            gripper_config=self.gripper_config,
            start_joints=start_joints,
            baudrate=self.baud_rate,
            using_sensor= True if self.using_sensor==1 else False
        )

class DobotAgent(Agent):
    def __init__(
        self,
        which_hand: str,
        dobot_config: Optional[DobotRobotConfig] = None,
        start_joints: Optional[np.ndarray] = None,
    ):
        self.which_hand = which_hand
        self.torque_enable = True
        assert dobot_config
        self._robot = dobot_config.make_robot(start_joints=start_joints)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        return self._robot.get_joint_state()

    def set_torque(self, _flag = False):
        self._robot.set_torque_mode(_flag)
        self.torque_enable = _flag

    def get_keys(self):
        return self._robot.get_key_status()


def main() -> None:
    pass


if __name__ == "__main__":
    ini_file_path = os.path.dirname(__file__).replace("dobot_control/agents", '')+"scripts/dobot_config/dobot_settings.ini"
    ini_file = configparser.ConfigParser()
    ini_file.read(ini_file_path)