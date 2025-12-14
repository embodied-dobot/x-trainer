import weakref
import numpy as np

from collections.abc import Callable

import carb
import omni

from ..device_base import Device


class BiKeyboard(Device):
    """A keyboard controller for sending SE(3) commands as delta poses for lerobot.
    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        J1_1                            Q                 SHIFT+Q
        J1_2                            W                 SHIFT+W
        J1_3                            E                 SHIFT+E
        J1_4                            A                 SHIFT+A
        J1_5                            S                 SHIFT+S
        J1_6                            D                 SHIFT+D
        J1_7+J1_8                       G                 SHIFT+G
        
        J2_1                            U                 SHIFT+U
        J2_2                            I                 SHIFT+I
        J2_3                            O                 SHIFT+O
        J2_4                            J                 SHIFT+J
        J2_5                            K                 SHIFT+K
        J2_6                            L                 SHIFT+L
        J2_7+J2_8                       H                 SHIFT+H
        ============================== ================= =================
    """

    def __init__(self, env, sensitivity: float = 0.05):
        super().__init__(env)
        """Initialize the keyboard layer.
        """
        # store inputs
        self.sensitivity = sensitivity

        # acquire omniverse interfaces
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )
        # bindings for keyboard to command
        self._create_key_bindings()

        # Command buffer: [LeftArm(6), LeftGrip(2), RightArm(6), RightGrip(2)] = 16 DoF
        self._delta_pos = np.zeros(16)

        # some flags and callbacks
        self.started = False
        self._reset_state = False
        self._additional_callbacks = {}
        
        # State tracking for SHIFT logic
        self._shift_pressed = False
        self._active_key_velocities = {}

    def __del__(self):
        """Release the keyboard interface."""
        self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
        self._keyboard_sub = None

    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = "BiKeyboard Controller for Dual-Arm X-Trainer (16 DoF).\n"
        msg += f"\tKeyboard name: {self._input.get_keyboard_name(self._keyboard)}\n"
        msg += "\tHint: Hold [SHIFT] key + Letter key to move in negative direction.\n"
        msg += "\t----------------------------------------------------------\n"
        msg += "\tJoint Index                  Left Arm Key     Right Arm Key\n"
        msg += "\t----------------------------------------------------------\n"
        msg += "\tJoint 1 (Shoulder Pan)       Q                U\n"
        msg += "\tJoint 2 (Shoulder Lift)      W                I\n"
        msg += "\tJoint 3 (Elbow Flex)         E                O\n"
        msg += "\tJoint 4 (Wrist Flex)         A                J\n"
        msg += "\tJoint 5 (Wrist Roll)         S                K\n"
        msg += "\tJoint 6 (Wrist Yaw)          D                L\n"
        msg += "\tGripper (Open/Close)         G                H\n"
        msg += "\t----------------------------------------------------------\n"
        msg += "\tStart Control:          B\n"
        msg += "\tTask Failed & Reset:    R\n"
        msg += "\tTask Success & Reset:   N\n"
        msg += "\tControl+C: quit"
        return msg

    def get_device_state(self):
        return self._delta_pos

    def input2action(self):
        state = {}
        reset = state["reset"] = self._reset_state
        state['started'] = self.started
        if reset:
            self._reset_state = False
            return state
        state['joint_state'] = self.get_device_state()

        ac_dict = {}
        ac_dict["reset"] = reset
        ac_dict['started'] = self.started
        ac_dict['bi_keyboard'] = True
        if reset:
            return ac_dict
        ac_dict['joint_state'] = state['joint_state']
        return ac_dict

    def reset(self):
        self._delta_pos = np.zeros(16)
        self._active_key_velocities.clear()
        self._shift_pressed = False

    def add_callback(self, key: str, func: Callable):
        self._additional_callbacks[key] = func

    def _on_keyboard_event(self, event, *args, **kwargs):
        # apply the command when pressed
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # tracking SHIFT state
            if event.input.name in ["LEFT_SHIFT", "RIGHT_SHIFT"]:
                self._shift_pressed = True
            
            # function keys (B, R, N)
            if event.input.name == "B":
                self.started = True
                self._reset_state = False
            elif event.input.name == "R":
                self.started = False
                self._reset_state = True
                if "R" in self._additional_callbacks:
                    self._additional_callbacks["R"]()
            elif event.input.name == "N":
                self.started = False
                self._reset_state = True
                if "N" in self._additional_callbacks:
                    self._additional_callbacks["N"]()
            
            # control keys (Q, W, E...)
            if event.input.name in self._target_indices_map:
                target_indices = self._target_indices_map[event.input.name]
                if event.input.name not in self._active_key_velocities:
                    direction = -1.0 if self._shift_pressed else 1.0
                    val = direction * self.sensitivity

                    if len(target_indices) == 2: # gripper
                        self._delta_pos[target_indices[0]] = -abs(val)
                        self._delta_pos[target_indices[1]] = abs(val)
                    else:
                        self._delta_pos[target_indices[0]] += val
                    
                    self._active_key_velocities[event.input.name] = val
                # print(self._active_key_velocities)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if event.input.name in ["LEFT_SHIFT", "RIGHT_SHIFT"]:
                self._shift_pressed = False
            
            if event.input.name in self._target_indices_map:
                target_indices = self._target_indices_map[event.input.name]
                if event.input.name in self._active_key_velocities:
                    # Remove the value when you release it
                    stored_val = self._active_key_velocities.pop(event.input.name)

                    if len(target_indices) == 2:
                        self._delta_pos[target_indices[0]] = abs(stored_val)
                        self._delta_pos[target_indices[1]] = -abs(stored_val)
                    else:
                        self._delta_pos[target_indices[0]] -= stored_val
                    
        return True

    def _create_key_bindings(self):
        """Creates default key binding."""
        self._target_indices_map = {
            # --- (Left Arm) ---
            "Q": [0], # J1_1
            "W": [1], # J1_2
            "E": [2], # J1_3
            "A": [3], # J1_4
            "S": [4], # J1_5
            "D": [5], # J1_6
            "G": [6, 7], # Gripper

            # --- (Right Arm) ---
            "U": [8],  # J2_1
            "I": [9],  # J2_2
            "O": [10], # J2_3
            "J": [11], # J2_4
            "K": [12], # J2_5
            "L": [13], # J2_6
            "H": [14, 15], # Gripper
        }