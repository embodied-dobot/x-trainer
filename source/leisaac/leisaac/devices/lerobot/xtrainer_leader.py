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
    """A XTrainer Leader device for joint control."""

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

        print("Initializing X-Trainer Leader...")
        _, hands_dict = load_ini_data_hands()
        left_agent = DobotAgent(which_hand="LEFT", dobot_config=hands_dict["HAND_LEFT"])
        right_agent = DobotAgent(which_hand="RIGHT", dobot_config=hands_dict["HAND_RIGHT"])
        self.agent = BimanualAgent(left_agent, right_agent)

        self._motor_limits = XTRAINER_FOLLOWER_MOTOR_LIMITS

        self.last_status = np.array(([0, 0, 0], [0, 0, 0]))
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_hz = 120.0
        self._stale_timeout_s = 0.2
        self._last_stale_warning_time = 0.0

        self._latest_raw_data = np.zeros(14, dtype=np.float32)
        self._latest_joint_state = self._build_joint_state(self._latest_raw_data)
        self._latest_keys = np.zeros((2, 3), dtype=np.int32)
        self._latest_timestamp = 0.0
        self._last_sampled_joint_state = self._latest_joint_state.copy()

        self._button_last_keys = np.zeros((2, 3), dtype=np.int32)
        self._button_press_started = np.zeros((2, 2), dtype=np.float64)
        self._button_press_active = np.zeros((2, 2), dtype=bool)
        self._keys_press_count = np.zeros((2, 3), dtype=np.int32)

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

        self._last_smoothed_data = None
        self._filter_alpha = 1.0 # filter coefficients(0.0 ~ 1.0), the smaller the value, the smoother the performance, but the higher the latency.

        self._sampler_thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._sampler_thread.start()
        print("xtrainer sampler thread init success...")

    def __del__(self):
        """Release background resources."""
        self.stop()
        self.stop_keyboard_listener()

    def stop(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if hasattr(self, "_sampler_thread") and self._sampler_thread.is_alive():
            self._sampler_thread.join(timeout=1.0)

    def stop_keyboard_listener(self):
        if hasattr(self, "_input") and hasattr(self, "_keyboard") and hasattr(self, "_keyboard_sub"):
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __str__(self) -> str:
        msg = "XTrainer-Leader device for joint control.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tMove XTrainer-Leader to control XTrainer-Follower\n"
        return msg

    def _display_controls(self):
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

    def _sampling_loop(self):
        sample_period = 1.0 / self._sample_hz
        while not self._stop_event.is_set() and not self.is_falling[0]:
            tic = time.time()
            raw_data = self.agent.act({})
            keys = self.agent.get_keys()

            if raw_data is not None and len(raw_data) == 14:
                current_data = np.array(raw_data, dtype=np.float32)
                if self._last_smoothed_data is None:
                    self._last_smoothed_data = current_data
                else:
                    # EMA: y_t = alpha * x_t + (1 - alpha) * y_{t-1}
                    self._last_smoothed_data = (
                        (self._filter_alpha * current_data)
                        + ((1.0 - self._filter_alpha) * self._last_smoothed_data)
                    )

                joint_state = self._build_joint_state(self._last_smoothed_data)
                with self._state_lock:
                    self._latest_raw_data = current_data.copy()
                    self._latest_joint_state = joint_state
                    self._latest_timestamp = tic
                    self._last_sampled_joint_state = joint_state.copy()

            if keys is not None and len(keys):
                key_array = np.array(keys, dtype=np.int32)
                with self._state_lock:
                    self._latest_keys = key_array.copy()
                self._update_button_state(key_array, tic)

            remaining = sample_period - (time.time() - tic)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def _build_joint_state(self, data_smoothed):
        joint_state = {}

        if not self.left_disabled:
            joint_state["J1_1"] = data_smoothed[0]
            joint_state["J1_2"] = data_smoothed[1]
            joint_state["J1_3"] = data_smoothed[2]
            joint_state["J1_4"] = data_smoothed[3]
            joint_state["J1_5"] = data_smoothed[4]
            joint_state["J1_6"] = data_smoothed[5]
            joint_state["J1_7"] = -(1.0 - data_smoothed[6])
            joint_state["J1_8"] = 1.0 - data_smoothed[6]
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
        joint_state["J2_7"] = -(1.0 - data_smoothed[13])
        joint_state["J2_8"] = 1.0 - data_smoothed[13]
        return joint_state

    def _update_button_state(self, now_keys, now_time):
        dev_keys = now_keys - self._button_last_keys

        for i in range(2):
            if dev_keys[i, 0] == -1:
                self._button_press_started[i, 0] = now_time
                self._button_press_active[i, 0] = True
            if dev_keys[i, 0] == 1 and self._button_press_active[i, 0]:
                self._button_press_active[i, 0] = False
                press_duration = now_time - self._button_press_started[i, 0]
                if press_duration < 0.5:
                    self._keys_press_count[i, 0] += 1
                    if self._keys_press_count[i, 0] % 2 == 1:
                        self.what_to_do[i, 0] = 1
                        print("ButtonA: [" + str(i) + "] unlock", self.what_to_do)
                    else:
                        self.what_to_do[i, 0] = 0
                        print("ButtonA: [" + str(i) + "] lock", self.what_to_do)
                    self.agent.set_torque(i, not self.what_to_do[i, 0])
                elif press_duration > 1.0:
                    self._keys_press_count[i, 1] += 1
                    if self._keys_press_count[i, 1] % 2 == 1:
                        self.what_to_do[i, 1] = 1
                        print("ButtonA: [" + str(i) + "] servo")
                    else:
                        self.what_to_do[i, 1] = 0
                        print("ButtonA: [" + str(i) + "] stop servo")

        for i in range(2):
            if dev_keys[i, 1] == -1:
                self._button_press_started[i, 1] = now_time
                self._button_press_active[i, 1] = True
            if dev_keys[i, 1] == 1 and self._button_press_active[i, 1]:
                self._button_press_active[i, 1] = False
                if self._keys_press_count[0, 2] % 2 == 1:
                    if self._keys_press_count[0, 1] % 2 == 1 or self._keys_press_count[1, 1] % 2 == 1:
                        self.what_to_do[0, 2] = 1
                        now_dt = datetime.datetime.now()
                        self.dt_time[0] = int(now_dt.strftime("%Y%m%d%H%M%S"))
                        self._keys_press_count[0, 2] += 1
                else:
                    self.what_to_do[0, 2] = 0
                    self._keys_press_count[0, 2] += 1

        if self.using_sensor_protection:
            for i in range(2):
                if now_keys[i, 2] and self.what_to_do[i, 0]:
                    self.agent.set_torque(2, True)
                    self.is_falling[0] = 1

        self.last_status = self.what_to_do.copy()
        self._button_last_keys = now_keys.copy()

    def get_device_state(self):
        assert not self.is_falling, "sensor detection!"

        with self._state_lock:
            joint_state = self._latest_joint_state.copy()
            timestamp = self._latest_timestamp

        sample_age = time.time() - timestamp if timestamp else float("inf")
        if sample_age > self._stale_timeout_s:
            now = time.time()
            if now - self._last_stale_warning_time > 1.0:
                print(f"[Warning] XTrainer leader sample is stale ({sample_age:.3f}s). Reusing last snapshot.")
                self._last_stale_warning_time = now

        return joint_state

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state["started"] = self._started
        if reset:
            self._reset_state = False
            return state
        if not self._started:
            return {"reset": reset, "started": self._started, "xtrainer_leader": True}

        state["joint_state"] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict["started"] = self._started
        ac_dict["xtrainer_leader"] = True
        if reset:
            return ac_dict
        ac_dict["joint_state"] = state["joint_state"]
        ac_dict["motor_limits"] = self._motor_limits
        return ac_dict

    def reset(self):
        with self._state_lock:
            self._last_smoothed_data = None
            self._latest_joint_state = self._last_sampled_joint_state.copy()
            self._latest_timestamp = time.time()
        self.last_status = self.what_to_do.copy()

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func
