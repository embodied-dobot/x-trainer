import time
from typing import Any, Dict, Protocol

import numpy as np


class Agent(Protocol):
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns an action given an observation.

        Args:
            obs: observation from the environment.

        Returns:
            action: action to take on the environment.
        """
        raise NotImplementedError

    def set_torque(self, _flag=False):

        raise NotImplementedError

    def get_keys(self):

        raise NotImplementedError


class DummyAgent(Agent):
    def __init__(self, num_dofs: int):
        self.num_dofs = num_dofs

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        return np.zeros(self.num_dofs)


class BimanualAgent(Agent):
    def __init__(self, agent_left: Agent, agent_right: Agent):
        self.agent_left = agent_left
        self.agent_right = agent_right
        self._last_action = None
        self._last_keys = np.zeros((2, 3), dtype=np.int32)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        left_obs = {}
        right_obs = {}
        for key, val in obs.items():
            L = val.shape[0]
            half_dim = L // 2
            assert L == half_dim * 2, f"{key} must be even, something is wrong"
            left_obs[key] = val[:half_dim]
            right_obs[key] = val[half_dim:]
        left_action = self.agent_left.act(left_obs)
        right_action = self.agent_right.act(right_obs)

        if left_action is None or right_action is None:
            return self._last_action

        action = np.concatenate([left_action, right_action])
        self._last_action = action
        return action

    def set_torque(self, which_hand=2,  _flag=False):
        if which_hand == 0:
            self.agent_left.set_torque(_flag)
        elif which_hand == 1:
            self.agent_right.set_torque(_flag)
        else:
            self.agent_left.set_torque(_flag)
            self.agent_right.set_torque(_flag)

    def get_keys(self):
        left_keys = self.agent_left.get_keys()
        right_keys = self.agent_right.get_keys()

        if left_keys is None or right_keys is None:
            return self._last_keys.copy()

        if len(left_keys) == 0 or len(right_keys) == 0:
            return self._last_keys.copy()

        keys = np.array([left_keys, right_keys])
        self._last_keys = keys
        return keys.copy()