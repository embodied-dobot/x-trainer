# X-Trainer-LeIsaac

[English](README.md) | [中文](README.zh.md)

![Isaac Sim 4.5](https://img.shields.io/badge/Isaac%20Sim-4.5-0a84ff?style=for-the-badge&logo=nvidia)
![Isaac Lab 0.47.1](https://img.shields.io/badge/Isaac%20Lab-0.47.1-34c759?style=for-the-badge&logo=nvidia)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-ff9500?style=for-the-badge&logo=python)

This project is based on the **Isaac Lab (LeIsaac)** framework. It implements a complete pipeline for **X-Trainer Dual-Arm Robot** simulation, keyboard teleoperation, and data collection using a real robot leader.

The system features a dual-arm coordination task (Lift Cube), multi-camera visual perception (RGB), and high-precision 30Hz data recording suitable for VLA (Vision-Language-Action) model training.

Data collected via this pipeline is fully compatible with the **LeRobot** framework for training. Post-training evaluation can be performed directly within this environment using asynchronous inference.

---

## Features

* **Dual-Arm Simulation Environment**: Full URDF import of the X-Trainer robot with tuned physical collision and dynamic parameters.
* **Multi-Modal Perception**: Integrated **3-view RGB cameras** (Left Wrist, Right Wrist, Top).
    * Resolution: **640x480** (4:3 aspect ratio).
    * FOV: **69°** (Aligned with Realsense D435i physics).
* **Dual-Arm Keyboard Control (`BiKeyboard`)**:
    * Independent control for 14 degrees of freedom (DoF) using relative positioning.
* **Real Robot Teleoperation (`XTrainerLeader`)**:
    * Reads joint angles (14-dims) from the real X-Trainer leader arms via USB serial.
    * Real-time mapping to the simulation environment (16-dims) for "Digital Twin" control.
* **High-Quality Data Collection**:
    * Strict **30Hz** frame synchronization using `Decimation=2` and `Step_Hz=30`.
    * HDF5 recording containing aligned images and joint states.
    * Data can be converted into the LeRobot format for seamless model training.
* **Model Visualization & Evaluation**:
    * Uses a server-client asynchronous inference approach to interact with the LeRobot project, enabling seamless visual evaluation.

---

## Installation

To begin, download the necessary [USD assets](https://huggingface.co/dstx123/xtrainer-leisaac/tree/main) from Hugging Face and place them into the designated assets folder. Furthermore, we provide 15 sample datasets in the file `lift_cube.hdf5`, which could be used to train the ACT model.

### **Install the environment using Anaconda**

Please ensure **Isaac Sim** and **Isaac Lab** are installed. We recommend referring to the [LeIsaac project documentation](https://lightwheelai.github.io/leisaac/docs/getting_started/installation) for the installation process and selecting the appropriate configuration. **Isaac Sim 4.5** has been verified to work correctly.

Install this project after installing Isaac Lab:
```bash
conda activate leisaac
pip install -e source/leisaac
```

### **Optional: Install the environment using Docker**

First, build the image using the Dockerfile:
```bash
git clone https://github.com/dstx123/xtrainer_leisaac.git
cd docker
docker build --network=host -t xtrainer-leisaac:v1 .
```

Once the build is complete, modify the code path in the `start_docker.sh` script to map your local code into the container. Example: `-v /home/xtrainer_leisaac:/workspace/xtrainer_leisaac:rw`.

Create the container using the script:
```bash
./create_docker.sh
```

After the container is created for the first time, you can enter the container using the following command:
```bash
./start_docker.sh
```

You can verify if IsaacLab is working correctly using the example script:
```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py
```

Install the project inside the container:
```bash
cd xtrainer_leisaac
pip install -e source/leisaac
```

### *Note On Real Robot Teleoperation*

*The [dobot_xtrainer](https://github.com/robotdexterity/dobot_xtrainer) project is integrated into this project and is located in the `source/leisaac/leisaac/xtrainer_utils` directory. Therefore, you can directly use the xtrainer leader arm in reality to control the follower arm in the simulation with this project, thereby achieving data collection.*

---

## Usage

### 1. Keyboard Teleoperation
Control the simulated robot using the keyboard.

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=bi_keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view
```

#### ⌨️ Key Mapping (`BiKeyboard`)

The keyboard layout is designed ergonomically: the **Left Hand** controls the Left Arm, and the **Right Hand** controls the Right Arm.

| Joint Name | Left Arm Key | Right Arm Key | Logic |
| :--- | :---: | :---: | :--- |
| **J1** | `Q` | `U` | Hold to move, Release to stop |
| **J2** | `W` | `I` | Hold to move, Release to stop |
| **J3** | `E` | `O` | Hold to move, Release to stop |
| **J4** | `A` | `J` | Hold to move, Release to stop |
| **J5** | `S` | `K` | Hold to move, Release to stop |
| **J6** | `D` | `L` | Hold to move, Release to stop |
| **Gripper** | **`G`** | **`H`** | **Hold to Close, Release to Open** |

* **Reverse Movement**: Hold `Shift` + Key.
* **System Controls**:
    * `B`: Start Control
    * `R`: Reset Environment (Fail)
    * `N`: Task Success & Reset

### 2. Real Robot Leader Teleoperation
Connect the real X-Trainer robot as the leader device to drive the simulation.

#### Step 1: Initialize configuration file

Run this script whenever the leader arm USB port assignment changes.

```bash
python scripts/find_port.py
```

#### Step 2: Zero Point Calibration

Execute this script after adjusting the leader arm to its default initial position (as shown in the figure below). This only needs to be performed once during the initial setup.

```bash
python scripts/get_offset.py
```

<img src="./assets/docs/initial_position.png" width="640" alt="Leader arm initial position" />

#### Step 3: Teleoperation via Leader Arm

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --teleop_device=xtrainerleader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view
```

### 3. Data Conversion

After data collection is complete, run the script to convert the data into the LeRobot format.

```bash
python scripts/convert/isaaclab2lerobot_xtrainer.py
```

It is recommended to create a separate `lerobot` environment in Conda for training.

### 4. Visual Evaluation

Once the model is trained, visual evaluation can be performed within this project.

**Start the server (in LeRobot environment):**
```bash
conda activate lerobot
cd ~/lerobot
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=5555 \
     --fps=30 
```

**Start the client (in LeIsaac environment):**
```bash
conda activate leisaac
cd ~/leisaac
python scripts/evaluation/policy_inference.py \
    --task=LeIsaac-XTrainer-PickCube-v0 \
    --eval_rounds=10 \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Grab cube and place into plate" \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="./checkpoints/last/pretrained_model"
```

**ACT Data Capture:**

<img src="./assets/docs/display1.gif" width="640" alt="ACT data capture" />

**ACT Model Demo:**

<img src="./assets/docs/display3.gif" width="640" alt="ACT Demo" />

**Multi-View Camera Outputs (Left Wrist / Right Wrist / Top):**

<img src="./assets/docs/display2.png" width="640" alt="Camera Views" />

### Upcoming Support

| Headset | Status |
| :-- | :-- |
| Quest 3 | 🔄 In progress |
| PICO 4 | 🔄 In progress |
| Vision Pro | 🔄 In progress |

---

## Contributing & Support

We welcome pull requests and issue reports. Please:

1. Fork the repository and create a feature branch.
2. Follow the existing coding style and add tests/demos when possible.
3. Submit a PR describing the motivation and testing status.

For bug reports or feature requests, open an issue on GitHub. You can also reach out via the issue tracker if you need help integrating your hardware or policies.

## Acknowledgements / Citation

This project builds upon:

* [Isaac Lab / LeIsaac](https://github.com/lightwheelai/leisaac)
* [LeRobot](https://github.com/huggingface/lerobot)
* [dobot_xtrainer](https://github.com/robotdexterity/dobot_xtrainer)

Please cite the respective projects if you use their components in academic work.

## License

This repository is distributed under the [BSD-3-Clause License](LICENSE).
