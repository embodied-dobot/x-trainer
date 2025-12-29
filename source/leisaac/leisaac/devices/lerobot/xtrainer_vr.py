from collections.abc import Callable

from ..device_base import Device

from leisaac.assets.robots.xtrainer import XTRAINER_FOLLOWER_MOTOR_LIMITS

import time
import numpy as np
import threading


import os
import asyncio
import http.server
import ssl
import socket
from scipy.spatial.transform import Rotation as R

from ...xtrainer_utils.XLeVR.xlevr.config import XLeVRConfig
from ...xtrainer_utils.XLeVR.xlevr.inputs.vr_ws_server import VRWebSocketServer
from ...xtrainer_utils.XLeVR.xlevr.inputs.base import ControlGoal, ControlMode
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
XLEVR_PATH = os.path.join(base_dir, "xtrainer_utils", "XLeVR")

class XTrainerVR(Device):
    """A XTrainer VR device for joint control.
    """

    def __init__(self, env, left_disabled: bool = False):
        super().__init__(env)
        self.left_disabled = left_disabled
        self._motor_limits = XTRAINER_FOLLOWER_MOTOR_LIMITS

        self.vr_origin = {
            "left": {"pos": None, "rot": None}, 
            "right": {"pos": None, "rot": None}
            }
        self.calibration_triggered = {"left": False, "right": False}

        raw_home_data = {
            "left":  np.array([0.345, -0.1175, 0.4157, 180.0, 0.0, 90.0]),
            "right": np.array([0.715, -0.1175, 0.4157, -180.0, 0.0, -90.0])
        }
        self.ROBOT_HOME = {}
        for hand, data in raw_home_data.items():
            pos = data[:3]
            euler = data[3:]
            r = R.from_euler('XYZ', euler, degrees=True)
            scipy_quat = r.as_quat()  # [x, y, z, w]
            self.ROBOT_HOME[hand] = {
                "pos": pos,
                "rot_matrix": r
            }

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        self._display_controls()

        # {arm_name: [x,y,z, qw,qx,qy,qz, gripper]}
        self.latest_state = {
            "left": None,
            "right": None
        }
        self._last_buttons = {
            "x": False, "y": False,  # Left Controller
            "a": False, "b": False   # Right Controller
        }

        print("=" * 60)
        print("VR Control Information Monitor")
        print("=" * 60)

        self.config = XLeVRConfig()
        self.config.enable_vr = True
        self.config.enable_https = True
        
        cert_path = os.path.join(XLEVR_PATH, 'cert.pem')
        key_path = os.path.join(XLEVR_PATH, 'key.pem')
        self.config.certfile = cert_path
        self.config.keyfile = key_path

        self.command_queue = asyncio.Queue()

        self.vr_server = VRWebSocketServer(
            command_queue=self.command_queue,
            config=self.config,
            print_only=False
        )

        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_vr_services, daemon=True)
        self.thread.start()

        time.sleep(1)
        self._display_connection_info()
        
    def __str__(self) -> str:
        """Returns: A string containing the information of xtrainer vr."""
        msg = "XTrainer-VR device for control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove XTrainer-VR to control XTrainer-Follower\n"
        return msg

    def _display_controls(self):
        print("\n🎮 VR Controller Mapping:")
        print("   [Right  B] -> Start Control")
        print("   [Left  X] -> Reset (Fail)")
        print("   [Left Y] -> Reset (Success/Next)")
        print("")

    def _run_vr_services(self):
        """Background thread: Runs both the HTTPS server and the WebSocket server simultaneously."""
        asyncio.set_event_loop(self.loop)
        
        try:
            handler = SimpleFileHandler
            handler.web_root = os.path.join(XLEVR_PATH, "web-ui") 
            
            httpd = http.server.HTTPServer((self.config.host_ip, self.config.https_port), handler)
            
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(self.config.certfile, self.config.keyfile)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            
            http_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            http_thread.start()
            print(f"🌐 HTTPS Server running on port {self.config.https_port}")
        except Exception as e:
            print(f"❌ HTTPS Server start failed: {e}")

        try:
            self.loop.run_until_complete(self.vr_server.start())
            print(f"✅ VR WebSocket Server running on port {self.config.websocket_port}")
            self.loop.run_forever()
        except Exception as e:
            print(f"❌ VR Loop failed: {e}")

    def get_device_state(self):
        while not self.command_queue.empty():
            try:
                goal = self.command_queue.get_nowait()
                if goal.arm != 'headset':  # skip headset
                    # button
                    if goal.metadata and "buttons" in goal.metadata:
                        hand_info = goal.metadata.get("hand", goal.arm)
                        self._check_buttons(goal.metadata["buttons"], hand_info)

                    # pose
                    if self._started and goal.target_position is not None: # after pressing B
                        pose_data = self._convert_goal_to_pose(goal)
                        self.latest_state[goal.arm] = pose_data
            except asyncio.QueueEmpty:
                break
        
        joint_state = {}

        if not self.left_disabled and self.latest_state["left"] is not None:
            joint_state["left"] = self.latest_state["left"]
        
        if self.latest_state["right"] is not None:
            joint_state["right"] = self.latest_state["right"]

        # print(joint_state)
        return joint_state
    
    def _check_buttons(self, buttons_dict, hand):
        is_right = 'right' in hand.lower()
        is_left = 'left' in hand.lower()

        for key, is_pressed in buttons_dict.items():
            k = str(key).lower()
            
            if is_right:
                if k == 'a': self._process_single_button('right_a', is_pressed)
                elif k == 'b': self._process_single_button('right_b', is_pressed)
            elif is_left:
                if k == 'a': self._process_single_button('left_x', is_pressed)
                elif k == 'b': self._process_single_button('left_y', is_pressed)
    
    def _process_single_button(self, unique_key, is_pressed):
        was_pressed = self._last_buttons.get(unique_key, False)
         
        if is_pressed and not was_pressed:
            parts = unique_key.split('_')
            self._handle_button_press(parts[1], parts[0])
            
        self._last_buttons[unique_key] = is_pressed

    def _handle_button_press(self, btn_name, hand_key):
        if btn_name == "b":
            print("🟢 [VR] Button B Pressed -> START Control")
            self._started = True
            self._reset_state = False
            self.calibration_triggered["left"] = True
            self.calibration_triggered["right"] = True
            
        elif btn_name == "x":
            print("🔴 [VR] Button X Pressed -> RESET (Fail)")
            self._started = False
            self._reset_state = True
            self.reset()
            if "R" in self._additional_callbacks:
                self._additional_callbacks["R"]()
                
        elif btn_name == "y":
            print("🔵 [VR] Button Y Pressed -> RESET (Success)")
            self._started = False
            self._reset_state = True
            self.reset()
            if "N" in self._additional_callbacks:
                self._additional_callbacks["N"]()

    def _convert_goal_to_pose(self, goal):
        """[x, y, z, qw, qx, qy, qz, gripper]"""
        # Position
        raw_pos = goal.target_position
        current_absolute_pos = np.array([  # rotate -90 degrees around the x-axis (Align the wrist coordinate system with Jx_6)
                                        raw_pos[0],  # Sim X = VR X
                                        -raw_pos[2], # Sim Y = VR -Z
                                        raw_pos[1]   # Sim Z = VR Y
                                       ])
        # Rotation
        raw_quat = np.array([0.0, 0.0, 0.0, 1.0])
        if goal.metadata and 'quaternion' in goal.metadata:
            q = goal.metadata['quaternion']
            raw_quat = np.array([
                q.get('x', 0.0), 
                q.get('y', 0.0), 
                q.get('z', 0.0), 
                q.get('w', 1.0) 
            ])

        # r_debug = R.from_quat(raw_quat)
        # euler_debug = r_debug.as_euler('xyz', degrees=True)
        # print(f"🔍 [{goal.arm}] Raw Euler (XYZ): {euler_debug}")

        # B: sim, A: VR.
        # A is obtained by rotating B around the x-axis by 90 degrees.
        r_BA = R.from_euler('x', 90, degrees=True)  
        r_A = R.from_quat(raw_quat)

        # Using the startup pose as the initial pose.
        hand_name = goal.arm
        if self.calibration_triggered.get(hand_name, False):
            self.vr_origin[hand_name]["pos"] = current_absolute_pos
            self.vr_origin[hand_name]["rot"] = r_A
            self.calibration_triggered[hand_name] = False
            print(f"📍 [{hand_name}] VR Calibrated!")

        # calculate pose
        home_pos = self.ROBOT_HOME[hand_name]["pos"]
        home_rot = self.ROBOT_HOME[hand_name]["rot_matrix"]

        if self.vr_origin[hand_name].get("pos") is not None and self.vr_origin[hand_name].get("rot") is not None:
            pos = home_pos + (current_absolute_pos - self.vr_origin[hand_name]["pos"])
            # print(hand_name, ": ", current_absolute_pos - self.vr_origin[hand_name]["pos"])

            # global rotation increment in VR world frame
            r_diff_raw = r_A * self.vr_origin[hand_name]["rot"].inv()
            # map the rotation increment to the robot coordinate system
            r_diff_sim = r_BA * r_diff_raw * r_BA.inv()
            # apply the initial rotation of the end effector
            r_final = r_diff_sim * home_rot

        else:
            pos = home_pos
            r_final = home_rot

        final_quat_scipy = r_final.as_quat() # [x, y, z, w]
        quat = np.array([final_quat_scipy[3], final_quat_scipy[0], final_quat_scipy[1], final_quat_scipy[2]])

        # Gripper (0.0 = Open, 1.0 = Closed)
        trigger = 0.0
        if goal.metadata and 'trigger' in goal.metadata:
            trigger = goal.metadata['trigger']
        
        return np.concatenate([pos, quat, [trigger]])

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self._started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self._started
        ac_dict['xtrainer_vr'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = self._motor_limits
        return ac_dict

    def reset(self):
        self.vr_origin = {
                "left": {"pos": None, "rot": None}, 
                "right": {"pos": None, "rot": None}
            }
        self.calibration_triggered = {"left": False, "right": False}

    def _display_connection_info(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except:
            ip = "localhost"
            
        print("\n" + "="*50)
        print(f"🎧 VR Ready! Connect to:")
        print(f"👉 https://{ip}:{self.config.https_port}")
        print("="*50 + "\n")

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

class SimpleFileHandler(http.server.SimpleHTTPRequestHandler):
    web_root = ""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=self.web_root, **kwargs)

    def log_message(self, format, *args):
        pass