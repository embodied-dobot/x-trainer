import time
from threading import Event, Lock, Thread
from typing import Protocol, Sequence
from leisaac.xtrainer_utils.utils.function_util import mismatch_data_write, wait_period, log_write, mk_dir, scan_port
import numpy as np
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_read import GroupSyncRead
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.group_sync_write import GroupSyncWrite
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.packet_handler import PacketHandler
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.port_handler import PortHandler
from leisaac.xtrainer_utils.third_party.DynamixelSDK.python.src.dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

# Constants
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
ADDR_PRESENT_POSITION = 132
ADDR_PRESENT_POSITION = 140
LEN_PRESENT_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


class DynamixelDriverProtocol(Protocol):
    def set_joints(self, joint_angles: Sequence[float]):
        """Set the joint angles for the Dynamixel servos.

        Args:
            joint_angles (Sequence[float]): A list of joint angles.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_joints(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""

    def get_keys(self) -> np.ndarray:
        """Close the driver."""


class FakeDynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int]):
        self._ids = ids
        self._joint_angles = np.zeros(len(ids), dtype=int)
        self._torque_enabled = False

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")
        self._joint_angles = np.array(joint_angles)

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        self._torque_enabled = enable

    def get_joints(self) -> np.ndarray:
        return self._joint_angles.copy()

    def close(self):
        pass

    def get_keys(self) -> np.ndarray:
        pass


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(
        self, ids: Sequence[int], append_id: int, port: str = "/dev/ttyUSB0", baudrate: int = 1000000, using_sensor = False
    ):

        """Initialize the DynamixelDriver class.

        Args:
            ids (Sequence[int]): A list of IDs for the Dynamixel servos.
            port (str): The USB port to connect to the arm.
            baudrate (int): The baudrate for communication.
        """
        self._ids = ids
        self._append_id = append_id
        self.using_sensor = using_sensor
        self._joint_angles = None
        self._joint_keys = np.zeros(3)  # add 2 keys
        self._lock = Lock()

        # Initialize the port handler, packet handler, and group sync read/write
        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler,
            self._packetHandler,
            ADDR_PRESENT_POSITION,
            LEN_PRESENT_POSITION,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler,
            self._packetHandler,
            ADDR_GOAL_POSITION,
            LEN_GOAL_POSITION,
        )

        # Open the port and set the baudrate
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")

        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        # Add parameters for each Dynamixel servo to the group sync read
        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(
                    f"Failed to add parameter for Dynamixel with ID {dxl_id}"
                )

        # Disable torque for each Dynamixel servo
        self._torque_enabled = True
        assert not self.set_torque_mode(self._torque_enabled), "fail set torque"
        self.set_pid_P()

        self._stop_thread = Event()
        self._start_reading_thread()

    def _internal_getKey(self):
        txBuf = [0xAA, 0x55, 0xAA]
        self._portHandler.writePort(txBuf)
        self._portHandler.setPacketTimeout(2)
        rxpacket = []
        if self.using_sensor:
            waitLen = 4
        else:
            waitLen = 2
        while True:
            rxpacket.extend(self._portHandler.readPort(2))
            rx_length = len(rxpacket)
            if rx_length >= waitLen:
                break
            if self._portHandler.isPacketTimeout():
                break
        if self.using_sensor:
            flag_in = ((rxpacket[0] + rxpacket[1] + rxpacket[2] + rxpacket[3]) == 255*2 )
        else:
            flag_in = len(rxpacket) == waitLen and rxpacket[0] | rxpacket[1] == 0xff and rxpacket[0] & rxpacket[1] == 0x00
        if flag_in:
            if self.using_sensor:
                return [(rxpacket[0] >> 4) & 0b1111, rxpacket[0] & 0b00001111, rxpacket[2]]
            else:
                return [(rxpacket[0] >> 4) & 0b1111, rxpacket[0] & 0b00001111, 0]

        return []

    def set_joints(self, joint_angles: Sequence[float]):
        if len(joint_angles) != len(self._ids):
            raise ValueError(
                "The length of joint_angles must match the number of servos"
            )
        if not self._torque_enabled:
            raise RuntimeError("Torque must be enabled to set joint angles")

        for dxl_id, angle in zip(self._ids, joint_angles):
            # Convert the angle to the appropriate value for the servo
            position_value = int(angle * 2048 / np.pi)

            # Allocate goal position value into byte array
            param_goal_position = [
                DXL_LOBYTE(DXL_LOWORD(position_value)),
                DXL_HIBYTE(DXL_LOWORD(position_value)),
                DXL_LOBYTE(DXL_HIWORD(position_value)),
                DXL_HIBYTE(DXL_HIWORD(position_value)),
            ]

            # Add goal position value to the Syncwrite parameter storage
            dxl_addparam_result = self._groupSyncWrite.addParam(
                dxl_id, param_goal_position
            )
            if not dxl_addparam_result:
                raise RuntimeError(
                    f"Failed to set joint angle for Dynamixel with ID {dxl_id}"
                )

        # Syncwrite goal position
        dxl_comm_result = self._groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Failed to syncwrite goal position")

        # Clear syncwrite parameter storage
        self._groupSyncWrite.clearParam()

    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def set_torque_mode(self, enable: bool):
        torque_value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        id_all = tuple(self._ids) + (self._append_id,)
        with self._lock:
            for dxl_id in id_all:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, torque_value
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    # print(dxl_comm_result)
                    # print(dxl_error)
                    raise RuntimeError(
                        f"Failed to set torque mode for Dynamixel with ID {dxl_id}"
                    )

        self._torque_enabled = enable

    def _start_reading_thread(self):
        self._reading_thread = Thread(target=self._read_joint_angles)
        self._reading_thread.daemon = True
        self._reading_thread.start()

    def _read_joint_angles(self):
        # Continuously read joint angles and update the joint_angles array
        while not self._stop_thread.is_set():
            time.sleep(0.001)
            with self._lock:
                _joint_angles = np.zeros(len(self._ids), dtype=int)
                # print("aaaa: ", self._ids)
                dxl_comm_result = self._groupSyncRead.txRxPacket()
                if dxl_comm_result != COMM_SUCCESS:
                    print(f"warning, comm failed: {dxl_comm_result}")
                    continue
                for i, dxl_id in enumerate(self._ids):
                    if self._groupSyncRead.isAvailable(
                        dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                    ):
                        angle = self._groupSyncRead.getData(
                            dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION
                        )
                        angle = np.int32(np.uint32(angle))
                        _joint_angles[i] = angle
                    else:
                        raise RuntimeError(
                            f"Failed to get joint angles for Dynamixel with ID {dxl_id}"
                        )
                self._joint_angles = _joint_angles
                keys = self._internal_getKey()
                if keys is not None:
                    self._joint_keys = keys
            # self._groupSyncRead.clearParam() # TODO what does this do? should i add it

    def get_joints(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_angles is None:
            time.sleep(0.1)
        with self._lock:
            _j = self._joint_angles.copy()
        # print(_j, self._ids)
        return _j / 2048.0 * np.pi

    def close(self):
        self._stop_thread.set()
        self._reading_thread.join()
        self._portHandler.closePort()

    def get_keys(self) -> np.ndarray:
        # Return a copy of the joint_angles array to avoid race conditions
        while self._joint_keys is None:
            time.sleep(0.01)
        with self._lock:
            key = self._joint_keys.copy()
        # log_write(__file__, "ButtonA: ["+str(key)+"] unlock")
        return key

    # set p
    def set_joint_arg_2Byte(self, _id, addr, val):
        with self._lock:
            dxl_comm_result, dxl_error = self._packetHandler.write2ByteTxRx(
                self._portHandler, _id, addr, val
            )
            if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                # print(dxl_comm_result)
                # print(dxl_error)
                raise RuntimeError(
                    f"Failed to set joint args ,ID {_id}"
                )

    def set_pid_P(self):
        id_all = list(tuple(self._ids)[:6] + (self._append_id,))
        id_all.sort()
        para = [1000, 2000, 2000, 600, 3000, 3000, 3000]
        # para = [460, 460, 460, 400, 400, 400, 400]
        for i in range(len(id_all)):
            self.set_joint_arg_2Byte(id_all[i], 84, para[i])


def main():
    # Set the port, baudrate, and servo IDs
    scan_port()
    ids_left = (1, 2, 4, 5, 6, 7, 8)
    # ids_right = (11, 12, 14, 15, 16, 17, 18)
    # Create a DynamixelDriver instance
    driver1 = DynamixelDriver(ids_left, 3, "/dev/ttyUSB2")
    # driver2 = DynamixelDriver(ids_right, 13, "/dev/ttyUSB1")
    driver1.set_torque_mode(True)
    # driver2.set_torque_mode(False)
    # while True:
    joint_angles1 = driver1.get_joints()
    print(joint_angles1)
        # joint_angles2 = driver2.get_joints()
        # print(joint_angles1[-1], joint_angles2[-1])
        # print(driver1.get_keys())


if __name__ == "__main__":
    main()
