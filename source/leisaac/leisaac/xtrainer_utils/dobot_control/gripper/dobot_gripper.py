# gripper for hapatics2 
import threading
import time
import struct
from enum import Enum
from typing import Tuple
import os, sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import serial.tools.list_ports as serial_stl
from leisaac.xtrainer_utils.utils.function_util import save_videos, mk_dir, free_limit_and_set_one

BASE_DIR += "/../"
# 将根目录添加到path中
import time

sys.path.append(BASE_DIR)

from leisaac.xtrainer_utils.third_party.feetech.scservo_sdk import *


def deal_hex_send_data(send_data):
    r_send_data = struct.pack("%dB" % (len(send_data)), *send_data)
    return r_send_data


class DobotGripper:
    """Communicates with the gripper ,using usb <==> 485 module connect to gripper """

    def __init__(self, port: str, id_name: int, servo_pos: list):
        """Constructor."""
        self.socket = None
        self.port = port
        self._portHandler = PortHandler(port)
        self._packHandler = protocol_packet_handler(portHandler=self._portHandler,
                                                    protocol_end=0)  # SCServo bit end(STS/SMS=0, SCS=1)
        # self._packHandler.write2ByteTxRx()
        self.servo = sms_sts(self._portHandler)
        self.command_lock = threading.Lock()
        self._min_position = 0  # 对外的最小位置
        self._max_position = 255  # 对外的最大位置
        self._min_speed = 0  # 对外的最小速度
        self._max_speed = 100  # 对外的最大速度
        self._min_force = 0  # 对外的最小力   暂时未能实现，舵机不支持
        self._max_force = 100  # 对外的最大力   暂时未能实现，舵机不支持
        self._min_servo_pos = servo_pos[0]  # 内部的 舵机最小位置 ，和实际的硬件有关系！
        self._max_servo_pos = servo_pos[1]  # 内部的 舵机最大位置 ，和实际的硬件有关系！
        self._min_servo_speed = 0  # 舵机的最小速度 和舵机型号有关系
        self._max_servo_speed = 4096  # 舵机的最大速度 和舵机型号有关系
        self._min_servo_force = 0  # 舵机的最小力 和舵机型号有关系 舵机不支持
        self._max_servo_force = 100  # 舵机的最大力 和舵机型号有关系 舵机不支持
        self._servo_id = id_name  # 舵机的ID ,和 舵机的配置有关系  SMS 舵机默认是1
        self._portHandler.setBaudRate(1000000)  # SMS 舵机的默认波特率是1000000

        self.ping()
        self.set_torque_limit(300)
        self.set_latency_timer()

    def set_latency_timer(self):
        _port_name = self.port.split("/")[-1]
        free_limit_and_set_one(f"/sys/bus/usb-serial/devices/{_port_name}/latency_timer")

    def ping(self) -> None:
        """Connects to a gripper at the given address.       
        :param port: usb com port  eg: /dev/ttyUSB0
        :param timeout: Timeout for blocking
        """
        model_number, result, error = self.servo.ping(self._servo_id)
        # print("ping servo ", self._servo_id, "result in ", model_number, result, error)
        assert error == 0, "ping failed..."
        return result

    def set_torque_limit(self, limit: int):
        ft_comm_result, ft_error = self._packHandler.write2ByteTxRx(
            self._servo_id, 48, limit
        )
        if ft_comm_result != COMM_SUCCESS or ft_error != 0:
            # print(ft_comm_result)
            # print(ft_error)
            raise RuntimeError(
                f"Failed to set torque mode for Feetech with ID {self._servo_id}"
            )

    def disconnect(self) -> None:
        """Closes the connection with the gripper."""
        print("will release servo and port ")
        self._portHandler.closePort()
        self._portHandler = None
        self._packHandler = None
        self.servo = None
        # self.socket.close()    

    def get_min_position(self) -> int:
        """Returns the minimum position the gripper can reach (open position)."""
        return self._min_position

    def get_max_position(self) -> int:
        """Returns the maximum position the gripper can reach (closed position)."""
        return self._max_position

    def get_open_position(self) -> int:
        """Returns what is considered the open position for gripper (minimum position value)."""
        return self.get_min_position()

    def get_closed_position(self) -> int:
        """Returns what is considered the closed position for gripper (maximum position value)."""
        return self.get_max_position()

    def is_open(self):
        """Returns whether the current position is considered as being fully open."""
        return self.get_current_position() <= self.get_open_position()

    def is_closed(self):
        """Returns whether the current position is considered as being fully closed."""
        return self.get_current_position() >= self.get_closed_position()

    # dobot
    def get_current_position(self) -> int:
        """Returns the current position as returned by the physical hardware."""
        position, result, error = self.servo.ReadPos(self._servo_id)
        if result != 0:
            raise ValueError("read pos failed !!! ", error)
        localPos = int((position - self._min_servo_pos) / (self._max_servo_pos - self._min_servo_pos) * (
                self._max_position - self._min_position))
        # print("read servo pos = ",position,"ret = ",localPos)
        return localPos

    # dobot
    def move(self, position: int, speed: int, force: int) -> Tuple[bool, int]:
        """  这里应该有 转换公式,将位置 力  速度等信息转换未舵机的数据！ """
        if position > self._max_position or position < self._min_position:
            raise ValueError("position over range [{}, {}]".format(self._min_position, self._max_position))
        if speed < self._min_speed or speed > self._max_speed:
            raise ValueError("Speed out of range [{}, {}]".format(self._min_speed, self._max_speed))

        localPos = int((position / (self._max_position - self._min_position)) * (
                self._max_servo_pos - self._min_servo_pos) + self._min_servo_pos)
        localSpd = int((speed / (self._max_speed - self._min_speed)) * (
                self._max_servo_speed - self._min_servo_speed) + self._min_servo_speed)
        # print("write pos ", localPos)
        # print("write spd ", localSpd)

        return self.servo.WritePosEx(self._servo_id, localPos, localSpd, 0)


def main():
    print(2048 + 2048 - 98)
    gripper = DobotGripper(port="/dev/ttyUSB0", servo_pos=[2048, 3998], id_name=22)
    idx = 10
    for i in range(3):
        tic = time.time()
        idx += 10 * i
        gripper.move(idx, 100, 1)
        time.sleep(1)
        print(idx, gripper.get_current_position())
        toc = time.time()
        print(toc - tic)


if __name__ == "__main__":
    main()
