from abc import abstractmethod
from typing import Dict, Protocol

import numpy as np
import time

class Robot(Protocol):
    """Robot protocol.

    A protocol for a robot that can be controlled.
    """

    @abstractmethod
    def num_dofs(self) -> int:
        """Get the number of joints of the robot.

        Returns:
            int: The number of joints of the robot.
        """
        raise NotImplementedError

    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        raise NotImplementedError

    @abstractmethod
    def command_joint_state(self, joint_state: np.ndarray, flag_in: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            flag_in:
            joint_state (np.ndarray): The state to command the leader robot to.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        raise NotImplementedError

    def set_do_status(self, which_do):
        """Get the current observations of the robot.

        Args:
            which_do:
        """
        raise NotImplementedError

    def get_DI_state(self, i):
        raise NotImplementedError

    @abstractmethod
    def get_XYZrxryrz_state(self) -> np.ndarray:
        """Get the current X Y Z rx ry rz state of the robot.
        Returns:
            T: The current X Y Z rx ry rz state of the leader robot.
        """
        raise NotImplementedError


class PrintRobot(Robot):
    """A robot that prints the commanded joint state."""

    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        # self._joint_state = np.zeros(num_dofs)
        self._joint_state = np.deg2rad(
            [90, 0, 90, 0, -90, 0, 90]
        )  # DOBOT

        self._dont_print = dont_print

    def num_dofs(self) -> int:
        return self._num_dofs

    def get_joint_state(self) -> np.ndarray:
        return self._joint_state

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        self._joint_state = joint_state
        if not self._dont_print:
            print(self._joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }


class BimanualRobot(Robot):
    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r

        self.frequency_ = 1/0.015
        # Set delta time to be used by receiveCallback
        self.delta_time_ = 1 / self.frequency_
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"

    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()

    def get_joint_state(self) -> np.ndarray:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )

    def command_joint_state(self, joint_state: np.ndarray, flag_in) -> None:
        t_start = time.time()
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        # print("t_start:",t_start)
        if flag_in[0]:
            self._robot_l.command_joint_state(joint_state[: self._robot_l.num_dofs()])
        # t_start1 = time.time()
        if flag_in[1]:
            self._robot_r.command_joint_state(joint_state[self._robot_l.num_dofs() :])
        # t_start2 = time.time()
        return 1

    def get_observations(self) -> Dict[str, np.ndarray]:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()

        return return_obs

    def set_do_status(self, which_do):
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        self._robot_l.set_do_status(which_do)

    def get_XYZrxryrz_state(self) -> np.ndarray:
        assert not self._robot_l.robot_is_err, "left robot error!"
        assert not self._robot_r.robot_is_err, "right robot error!"
        return np.concatenate(
            (self._robot_l.get_XYZrxryrz_state(), self._robot_r.get_XYZrxryrz_state())
        )

def main():
    pass


if __name__ == "__main__":
    main()
