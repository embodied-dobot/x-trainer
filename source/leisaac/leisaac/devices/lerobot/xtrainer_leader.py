from collections.abc import Callable

from ..device_base import Device

from leisaac.assets.robots.xtrainer import XTRAINER_FOLLOWER_MOTOR_LIMITS

import carb
import omni

import time
import numpy as np
import threading
from leisaac.xtrainer_utils.dobot_control.agents.agent import BimanualAgent
from leisaac.xtrainer_utils.dobot_control.agents.dobot_agent import DobotAgent
import datetime

from leisaac.xtrainer_utils.utils.manipulate_utils import load_ini_data_hands


class XTrainerLeader(Device):
    """A XTrainer Leader device for joint control.
    """

    def __init__(self, env, left_disabled: bool = False):
        super().__init__(env)
        # Thread button: [lock or nor, servo or not, record or not]
        # 0: lock, 1: unlock
        # 0: stop servo, 1: servo
        # 0: stop recording, 1: recording
        self.what_to_do = np.array(([0, 0, 0], [0, 0, 0]))
        self.dt_time = np.array([20240507161455])
        self.using_sensor_protection = False
        self.is_falling = np.array([0])
        self.left_disabled = left_disabled

        print(f"Initializing X-Trainer Leader...")
        _, hands_dict = load_ini_data_hands()
        left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
        right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
        self.agent = BimanualAgent(left_agent, right_agent)

        self._motor_limits = XTRAINER_FOLLOWER_MOTOR_LIMITS

        self.last_status = np.array(([0, 0, 0], [0, 0, 0]))  # init lock
        thread_button = threading.Thread(target=self.button_monitor_realtime)
        thread_button.start()
        print("button thread init success...")

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

        # some flags and callbacks
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}

        self._display_controls()

        # self.agent.set_torque(2, True)  # lock left and right hands

        self._last_smoothed_data = None
        self._filter_alpha = 1.0 # filter coefficients(0.0 ~ 1.0), the smaller the value, the smoother the performance, but the higher the latency.

    def __del__(self):
        """Release the keyboard interface."""
        self.stop_keyboard_listener()

    def stop_keyboard_listener(self):
        if hasattr(self, '_input') and hasattr(self, '_keyboard') and hasattr(self, '_keyboard_sub'):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of xtrainer leader."""
        msg = "XTrainer-Leader device for joint control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove XTrainer-Leader to control XTrainer-Follower\n"
        return msg

    def _display_controls(self):
        """
        Method to pretty print controls.
        """

        def print_command(char, info):
            char += " " * (30 - len(char))
            print("{}\t{}".format(char, info))

        print("")
        print_command("b", "start control")
        print_command("r", "reset simulation and set task success to False")
        print_command("n", "reset simulation and set task success to True")
        print_command("move leader", "control follower in the simulation")
        print_command("Control+C", "quit")
        print("")

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events using carb."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "B":
                self._started = True
                self._reset_state = False
            elif event.input.name == "R":
                self._started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif event.input.name == "N":
                self._started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
        return True

    def get_device_state(self):
        assert not self.is_falling, "sensor detection!"

        raw_data = self.agent.act({})
        dev_what_to_do = self.what_to_do.copy()-self.last_status
        self.last_status = self.what_to_do.copy()
        for i in range(2):  # decide whether to lock arms according to button A
            if dev_what_to_do[i, 0] != 0:
                self.agent.set_torque(i, not self.what_to_do[i, 0])

        # print(np.array2string(raw_data, precision=2, suppress_small=True, floatmode='fixed'))

        if len(raw_data) != 14:
            print(f"[Warning] XTrainer data length mismatch! Expected 14, got {len(raw_data)}")
            return {}

        # -------------------------------------------------------
        # EMA filtering
        # -------------------------------------------------------
        current_data = np.array(raw_data, dtype=np.float32)
        
        if self._last_smoothed_data is None:
            self._last_smoothed_data = current_data
        else:
            # EMA: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
            self._last_smoothed_data = (self._filter_alpha * current_data) + \
                                     ((1.0 - self._filter_alpha) * self._last_smoothed_data)
        data_smoothed = self._last_smoothed_data
        # -------------------------------------------------------

        joint_state = {}
        
        if not self.left_disabled:
            joint_state["J1_1"] = data_smoothed[0]
            joint_state["J1_2"] = data_smoothed[1]
            joint_state["J1_3"] = data_smoothed[2]
            joint_state["J1_4"] = data_smoothed[3]
            joint_state["J1_5"] = data_smoothed[4]
            joint_state["J1_6"] = data_smoothed[5]
            joint_state["J1_7"] = -(1.0-data_smoothed[6])
            joint_state["J1_8"] = 1.0-data_smoothed[6]
        else:
            joint_state["J1_1"] = 0.0
            joint_state["J1_2"] = 0.0
            joint_state["J1_3"] = 0.0
            joint_state["J1_4"] = 0.0
            joint_state["J1_5"] = 0.0
            joint_state["J1_6"] = 0.0
            joint_state["J1_7"] = 0.0
            joint_state["J1_8"] = 0.0

        joint_state["J2_1"] = data_smoothed[7]
        joint_state["J2_2"] = data_smoothed[8]
        joint_state["J2_3"] = data_smoothed[9]
        joint_state["J2_4"] = data_smoothed[10]
        joint_state["J2_5"] = data_smoothed[11]
        joint_state["J2_6"] = data_smoothed[12]
        joint_state["J2_7"] = -(1.0-data_smoothed[13])
        joint_state["J2_8"] = 1.0-data_smoothed[13]

        return joint_state

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
        ac_dict['xtrainer_leader'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        ac_dict['motor_limits'] = self._motor_limits
        return ac_dict

    def reset(self):
        pass

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def button_monitor_realtime(self):
        # servo
        last_keys_status = np.array(([0, 0, 0], [0, 0, 0]))
        start_press_status = np.array(([0, 0], [0, 0]))  # start press
        keys_press_count = np.array(([0, 0, 0], [0, 0, 0]))

        while not self.is_falling[0]:
            now_keys = self.agent.get_keys()
            dev_keys = now_keys - last_keys_status
            # button a
            for i in range(2):
                if dev_keys[i, 0] == -1:  # button a: start
                    tic = time.time()
                    start_press_status[i, 0] = 1
                if dev_keys[i, 0] == 1 and start_press_status[i, 0]:  # button a: end
                    start_press_status[i, 0] = 0
                    toc = time.time()
                    if toc-tic < 0.5:
                        keys_press_count[i, 0] += 1
                        # print(i, keys_press_count[i, 0], "short press", toc-tic)
                        if keys_press_count[i, 0] % 2 == 1:
                            self.what_to_do[i, 0] = 1
                            # log_write(__file__, "ButtonA: ["+str(i)+"] unlock")
                            print("ButtonA: [" + str(i) + "] unlock", self.what_to_do)
                        else:
                            self.what_to_do[i, 0] = 0
                            # log_write(__file__, "ButtonA: [" + str(i) + "] lock")
                            print("ButtonA: [" + str(i) + "] lock", self.what_to_do)

                    elif toc-tic > 1:
                        keys_press_count[i, 1] += 1
                        # print(i, keys_press_count[i, 1], "long press", toc-tic)
                        if keys_press_count[i, 1] % 2 == 1:
                            self.what_to_do[i, 1] = 1
                            # log_write(__file__, "ButtonA: [" + str(i) + "] servo")
                            print("ButtonA: [" + str(i) + "] servo")
                        else:
                            self.what_to_do[i, 1] = 0
                            # log_write(__file__, "ButtonA: [" + str(i) + "] stop servo")
                            print("ButtonA: [" + str(i) + "] stop servo")

            # button B
            # more than one start servo
            for i in range(2):
                if dev_keys[i, 1] == -1:  # B button pressed
                    start_press_status[i, 1] = 1
                if dev_keys[i, 1] == 1:
                    start_press_status[i, 1] = 0
                    if keys_press_count[0, 2] % 2 == 1:
                        if keys_press_count[0, 1] % 2 == 1 or keys_press_count[1, 1] % 2 == 1:
                            self.what_to_do[0, 2] = 1
                            # log_write(__file__, "ButtonB: [" + str(i) + "] recording")
                            # new recording
                            now_time = datetime.datetime.now()
                            self.dt_time[0] = int(now_time.strftime("%Y%m%d%H%M%S"))
                            keys_press_count[0, 2] += 1
                    else:
                        self.what_to_do[0, 2] = 0
                        keys_press_count[0, 2] += 1
                        # log_write(__file__, "ButtonB: [" + str(i) + "] stop recording")

            # status
            if self.using_sensor_protection:
                for i in range(2):
                    if now_keys[i, 2] and self.what_to_do[i, 0]:  # button a: 1 unlock
                        self.agent.set_torque(2, True)
                        self.is_falling[0] = 1

            last_keys_status = now_keys