import pickle
import threading
from typing import Any, Dict

import numpy as np
import zmq
import time
from .robot import Robot
from leisaac.xtrainer_utils.utils.function_util import log_write

DEFAULT_ROBOT_PORT = 6000

import threading


class ZMQServerRobot:
    def __init__(
        self,
        robot: Robot,
        port: int = DEFAULT_ROBOT_PORT,
        host: str = "127.0.0.1",
    ):
        self.port = port
        self._robot = robot
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        addr = f"tcp://{host}:{port}"
        debug_message = f"Robot Sever Binding to {addr}, Robot: {robot}"
        print(debug_message)
        self._timout_message = f"Timeout in Robot Server, Robot: {robot}"
        self._socket.bind(addr)
        self._stop_event = threading.Event()

        self.ON = 1
        self.OFF = 0
        self.Enable = self.OFF
        self.Follow = self.OFF
        self.Record = self.OFF

    def serve(self) -> None:
        """Serve the leader robot state over ZMQ."""
        self._socket.setsockopt(zmq.RCVTIMEO, 1)  # Set timeout to 1000 ms
        print("*"*100)
        while not self._stop_event.is_set():
            try:
                # Wait for next request from client
                message = self._socket.recv()
                request = pickle.loads(message)

                # Call the appropriate method based on the request
                method = request.get("method")
                args = request.get("args", {})
                result: Any
                if method == "num_dofs":
                    result = self._robot.num_dofs()
                elif method == "get_joint_state":
                    # log_write(str(self.port) + ": get_joint_state start")
                    result = self._robot.get_joint_state()
                    # log_write(str(self.port) + ": get_joint_state end")
                elif method == "command_joint_state":
                    tic = time.time()
                    toc = time.time()
                    # log_write(str(self.port) + ": command_joint_state start")
                    result = self._robot.command_joint_state(**args)
                    # log_write(str(self.port) + ": command_joint_state end")
                elif method == "get_observations":
                    result = self._robot.get_observations()
                elif method == "set_do_status":
                    result = self._robot.set_do_status(**args)
                elif method == "get_XYZrxryrz_state":
                    result = self._robot.get_XYZrxryrz_state()
                else:
                    result = {"error": "Invalid method"}
                    print(result)
                    raise NotImplementedError(
                        f"Invalid method: {method}, {args, result}"
                    )

                self._socket.send(pickle.dumps(result))
            except zmq.Again:
                print(self._timout_message)
                # Timeout occurred, check if the stop event is set

    def stop(self) -> None:
        """Signal the server to stop serving."""
        self._stop_event.set()


class ZMQClientRobot(Robot):
    """A class representing a ZMQ client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        request = {"method": "num_dofs"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def get_joint_state(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        request = {"method": "get_joint_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def command_joint_state(self, joint_state: np.ndarray, flag_in) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (T): The state to command the leader robot to.
            flag_in
        """
        request = {
            "method": "command_joint_state",
            "args": {"joint_state": joint_state,
                     "flag_in": flag_in},
        }
        send_message = pickle.dumps(request)
        start = time.time()
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        end = time.time()
        t = end-start
        return result

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        request = {"method": "get_observations"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result

    def set_do_status(self, which_do: np.ndarray):
        request = {
            "method": "set_do_status",
            "args": {"which_do": which_do},
        }
        send_message = pickle.dumps(request)
        start = time.time()
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        end = time.time()
        return result

    def close(self):
        self._socket.close()

    def get_XYZrxryrz_state(self):
        request = {"method": "get_XYZrxryrz_state"}
        send_message = pickle.dumps(request)
        self._socket.send(send_message)
        result = pickle.loads(self._socket.recv())
        return result
