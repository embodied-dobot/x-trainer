# XLeVR VR Monitor

This project is a minimal, refactored version of the original [telegrip](https://github.com/DipFlip/telegrip) VR teleoperation system. It is designed for lightweight VR controller data monitoring and prototyping.

## Installation

Install dependencies (no need for pip install -e):

```bash
pip install -r requirements.txt
```

## Usage

### 1. Run vr_monitor.py directly

From the XLeVR directory:
```bash
python vr_monitor.py
```
- Open your VR headset browser and go to the HTTPS address shown in the terminal (e.g. `https://<your-ip>:8443`).
- Move your VR controllers; the terminal will print real-time control data.

### 2. Use vr_monitor.py from another folder

You can copy `vr_monitor.py` to any directory (e.g., your own project folder). Just:
- Make sure the `XLeVR` main directory path is correct (edit the `XLEVR_PATH` variable at the top of the script).
- Install dependencies as above.

### 3. Access VR data in your code

The `VRMonitor` class stores the latest control goals (controller/headset data) in `self.left_goal`, `self.right_goal`, and `self.headset_goal`.

Example:
```python
monitor = VRMonitor()
monitor.initialize()
# In your event loop or callback:
left_goal = monitor.get_left_goal_nowait()
right_goal = monitor.get_right_goal_nowait()
# left_goal/right_goal are ControlGoal objects with position, orientation, etc.
```

## Data Structure: ControlGoal

Each of `self.left_goal`, `self.right_goal`, and `self.headset_goal` is a `ControlGoal` object with the following main fields:

- `arm`: Which device this goal is for ("left", "right", or "headset").
- `mode`: Control mode (e.g., POSITION_CONTROL).
- `target_position`: 3D position as a numpy array (e.g., `[x, y, z]`).
- `wrist_roll_deg`: Wrist roll angle in degrees (float, may be None for headset).
- `wrist_flex_deg`: Wrist flex (pitch) angle in degrees (float, may be None for headset).
- `gripper_closed`: Boolean, whether the gripper is closed (for controllers).
- `metadata`: Dictionary with extra info (raw controller data, trigger state, thumbstick, etc).

Example:
```python
print(left_goal.arm)              # 'left'
print(left_goal.target_position)  # numpy array: [x, y, z]
print(left_goal.wrist_roll_deg)   # float (degrees)
print(left_goal.metadata)         # dict with extra controller info
```

---

For advanced usage or web UI, see code comments or the original telegrip documentation.
