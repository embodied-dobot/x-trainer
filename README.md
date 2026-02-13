# X-Trainer-LeIsaac

[English](README.md) | [中文](README.zh.md)

![Isaac Sim 4.5](https://img.shields.io/badge/Isaac%20Sim-4.5-0a84ff?style=for-the-badge&logo=nvidia)
![Isaac Lab 0.47.1](https://img.shields.io/badge/Isaac%20Lab-0.47.1-34c759?style=for-the-badge&logo=nvidia)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-ff9500?style=for-the-badge&logo=python)

本项目基于 **Isaac Lab (LeIsaac)** 框架，提供了一套面向 **X-Trainer 双臂机器人** 的完整工作流：仿真、键盘遥操作、真实 Leader 设备驱动的数据采集、以及模型评估与评分系统。

系统包含三个比赛任务场景（task1/task2/task3）、三视角 RGB 视觉感知，并可在 30Hz 下高精度记录数据，适用于 VLA（Vision-Language-Action）模型训练。

采集的数据可直接在 **LeRobot** 框架中训练；训练后的模型同样能在本环境内进行异步推理评估和自动评分。

---

## 📋 目录

- [功能亮点](#功能亮点)
- [任务说明](#任务说明)
- [环境要求](#环境要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [使用方法](#使用方法)
  - [1. 键盘遥操作](#1-键盘遥操作)
  - [2. 真实 Leader 遥操作](#2-真实-leader-遥操作)
  - [3. 数据转换](#3-数据转换)
  - [4. 模型推理评估](#4-模型推理评估)
  - [5. 任务评分系统](#5-任务评分系统)
- [参数说明](#参数说明)
- [常见问题](#常见问题)
- [贡献与支持](#贡献与支持)

---

## 功能亮点

* **双臂仿真场景**：完整导入 X-Trainer 机器人 URDF，并调优碰撞/动力学参数。
* **多模态感知**：集成 **三视角 RGB 相机**（左腕、右腕、俯视），分辨率 640×480，FOV 69°，与 Realsense D435i 一致。
* **双臂键盘控制 (`BiKeyboard`)**：通过增量关节控制方式，独立控制 14 个自由度。
* **真实机器人遥操作 (`XTrainerLeader`)**：从真实 X-Trainer Leader 读取 14 维关节角，通过 USB 串口实时驱动仿真 16 维动作，实现数字孪生。
* **高质量数据采集**：使用 `Decimation=2`、`Step_Hz=30` 保证严格 30Hz 帧同步；以 HDF5 存储对齐的图像/关节数据，可直接转换为 LeRobot 数据。
* **模型可视化与评估**：通过服务端-客户端异步推理接口，与 LeRobot 项目无缝交互，方便可视化验证。
* **自动评分系统**：支持 task1/task2/task3 三个任务的自动评分，包含完成性、准确性、效率等多维度评分。

---

## 任务说明

本项目包含三个比赛任务场景，每个任务都有明确的成功判定标准和评分规则：

### Task1: 物体分类放置任务

**任务目标**：将桌面上的 6 个物体（3 个 good 物体 + 3 个 bad 物体）正确分类并放入对应的料箱。

**物体说明**：
- **Good 物体**（无污渍）：`Nova_J1_good_split`、`Nova_J4_good_split`、`Nova_J5_good_split` → 放入 **蓝色良品箱 (KLT_good)**
- **Bad 物体**（带黑色污渍）：`Nova_J1_bigbad_split`、`Nova_J4_bigbad_split`、`Nova_J5_bigbad_split` → 放入 **红色不良品箱 (KLT_bad)**

**成功条件**：所有 good 物体在 KLT_good 中，所有 bad 物体在 KLT_bad 中。

**评分标准**：
- **完成性得分（20分）**：放入任意料箱的物体数量 / 6 × 20
- **成功率得分（60分）**：正确分类放置的物体数量 / 6 × 60
- **工作效率得分（20分）**：仅在完成全部 6 个物体时计算，公式为 `max((t0 - t) / t0 * 20, 0)`，其中 t 为实际耗时，t0 为基准时间

### Task2: 收集与倾倒任务

**任务目标**：使用铲子收集三个物体（Emergency_Stop_Button、factory_nut_loose、chip）到绿色区域并正确使用盖子将他们完全覆盖。

**成功条件**：
1. 三个物体都置于**绿色判定区**（固定位置）
2. 三个物体同时置于**绿色判定区和盖子下方判定区的交集**（判定区跟随 protectlid 移动）

**评分标准**：
- **收集完整性得分（40分）**：收集到绿色判定区的物体数量 / 3 × 40
- **倾倒准确性得分（40分）**：置于两个判定区交集的物体数量 / 3 × 40
- **工作效率得分（20分）**：仅在全部 3 个物体都在交集内时计算，公式为 `max((t0 - t) / t0 * 20, 0)`

### Task3: 试管插入任务

**任务目标**：将 3 个试管（newTube_01、newTube_02、newTube_03）全部插入试管架（rack）的孔位。

**成功条件**：所有 tube 物体的判定点都在 rack 上方的绿色判定区域内。

**评分标准**：
- **试管插入得分（80分）**：成功插入的试管数量 / 3 × 80
- **工作效率得分（20分）**：仅在全部 3 个试管都插入时计算，公式为 `max((t0 - t) / t0 * 20, 0)`

---

## 环境要求

### 系统要求

- **操作系统**：Linux (Ubuntu 20.04/22.04 推荐)
- **GPU**：NVIDIA GPU，支持 CUDA（推荐 RTX 3060 或更高）
- **内存**：至少 16GB RAM
- **存储**：至少 50GB 可用空间

### 软件依赖

- **Isaac Sim 4.5**：NVIDIA 物理仿真平台
- **Isaac Lab 0.47.1**：基于 Isaac Sim 的强化学习框架
- **Python 3.10+**
- **CUDA 11.8+**（与 Isaac Sim 版本匹配）
- **LeRobot**：用于模型训练（可选，仅训练时需要）

---

## 安装指南

### 前置准备


1. **安装 Isaac Sim 和 Isaac Lab**：
   - 参考 [LeIsaac 官方文档](https://lightwheelai.github.io/leisaac/docs/getting_started/installation) 完成安装
   - 已验证 **Isaac Sim 4.5** 可正常工作
   - 确保 Isaac Lab 环境已正确配置

### 使用 Anaconda 安装（推荐）

1. **激活 LeIsaac 环境**：
```bash
conda activate leisaac
```

2. **安装本项目**：
```bash
cd /path/to/x-trainer-main
pip install -e source/leisaac
```

3. **验证安装**：
```bash
python -c "import leisaac; print('安装成功！')"
```

### （可选）使用 Docker 安装

1. **构建镜像**：
```bash
git clone 本项目的仓库地址
cd docker
docker build --network=host -t xtrainer-leisaac:v1 .
```

2. **修改路径映射**：编辑 `start_docker.sh`，设置正确的代码路径映射，例如：
```bash
-v /home/xtrainer_leisaac:/workspace/xtrainer_leisaac:rw
```

3. **创建并启动容器**：
```bash
./create_docker.sh
./start_docker.sh
```

4. **验证 Isaac Lab**：
```bash
cd /workspace/isaaclab
./isaaclab.sh -p scripts/tutorials/01_assets/run_rigid_object.py
```

5. **在容器内安装本项目**：
```bash
cd /workspace/xtrainer_leisaac
pip install -e source/leisaac
```

### 真实机器人遥操作注意事项

项目中已集成 [dobot_xtrainer](https://github.com/robotdexterity/dobot_xtrainer)（路径：`source/leisaac/leisaac/xtrainer_utils`），可直接使用真实 X-Trainer Leader 控制仿真 Follower，实现数据采集。

---

## 快速开始

### 1. 启动键盘遥操作（最快上手）

```bash
conda activate leisaac
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=bi_keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view
```

### 2. 查看任务可视化（可选）

添加 `--enable_visualization` 参数可查看判定区域和判定点：
- **绿色框**：良品箱/目标区域
- **红色框**：不良品箱
- **黄色/橙色球体**：物体判定点

> ⚠️ **注意**：可视化仅用于测试和调试，正式数据采集和评分时请关闭可视化。

---

## 使用方法

### 1. 键盘遥操作（不推荐）

#### 基本命令
以task1为例的键盘遥操作

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=bi_keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view
```

#### ⌨️ 键位说明（`BiKeyboard`）

布局按左右手分区：左手控制左臂，右手控制右臂；按住动作键移动，松开停止；按住 `Shift` + 键实现反向。

| 关节 | 左臂键位 | 右臂键位 | 说明 |
| :--- | :---: | :---: | :--- |
| **J1** | `Q` | `U` | 按住移动，松开停止 |
| **J2** | `W` | `I` | — |
| **J3** | `E` | `O` | — |
| **J4** | `A` | `J` | — |
| **J5** | `S` | `K` | — |
| **J6** | `D` | `L` | — |
| **夹爪** | **`G`** | **`H`** | **按住闭合，松开张开** |

**系统控制键**：
- `B`：开始控制
- `R`：失败并重置
- `N`：成功并重置

#### 可视化选项

- **默认**：关闭可视化（推荐用于数据采集）
- **启用可视化**：添加 `--enable_visualization` 参数（仅用于测试）

---

### 2. 真实 Leader 遥操作

#### 第一步：初始化串口配置

执行操作前请确保当前用户具有串口的操作权限。

```bash
python scripts/find_port.py
```

该脚本会扫描可用串口，找到 X-Trainer Leader 设备。

#### 第二步：零点标定

将 Leader 调整至初始姿态（见下图），运行：

```bash
python scripts/get_offset.py
```

<img src="./assets/docs/initial_position.png" width="640" alt="Leader 初始姿态" />

标定完成后会生成配置文件，用于后续遥操作。

#### 第三步：开始遥操作

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=xtrainerleader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view
```

> **注意**：判定可视化默认关闭，避免干扰操作。如需查看判定区域和判定点，可添加 `--enable_visualization` 参数。

---

### 3. 数据转换

采集完成后，HDF5 数据需要转换为 LeRobot 格式才能用于训练：

```bash
python scripts/convert/isaaclab2lerobot_xtrainer.py
```

**推荐**：单独创建 `lerobot` Conda 环境用于训练，避免依赖冲突。

---

### 4. 模型推理评估

训练好模型后，可在本项目中进行推理评估（不评分，仅可视化）。

#### 启动流程

**第一步：在 LeRobot 环境启动服务端**

```bash
conda activate lerobot
cd ~/lerobot
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=5555 \
     --fps=30
```

**第二步：在 LeIsaac 环境启动客户端**

```bash
conda activate leisaac
cd ~/x-trainer-main
python scripts/evaluation/policy_inference.py \
    --task=task1 \
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

> **注意**：推理评估脚本不会输出评分，仅用于可视化验证模型行为。如需评分，请使用评分脚本（见下一节）。

---

### 5. 任务评分系统

评分系统支持 task1/task2/task3 三个任务的自动评分，包含完成性、准确性、效率等多维度评分。

#### 启动流程

**第一步：在 LeRobot 环境启动服务端**

```bash
conda activate lerobot
cd ~/lerobot
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=5555 \
     --fps=30
```

**第二步：在 LeIsaac 环境启动评分客户端**

```bash
conda activate leisaac
cd ~/x-trainer-main
python scripts/evaluation/policy_scoring.py \
    --task=task1 \
    --eval_rounds=10 \
    --episode_length_s=60 \
    --score_t0_s=60 \
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

#### 评分参数说明

| 参数 | 说明 | 默认值 | 示例 |
| :--- | :--- | :---: | :--- |
| `--eval_rounds` | 评分回合数 | 0（无限） | `10` |
| `--episode_length_s` | 单回合最大时长（秒） | 60.0 | `60` |
| `--score_t0_s` | 效率分基准时间 t0（秒） | `episode_length_s` | `60` |

最终评分参数可能根据实际情况调整，以主办方说明为准。

#### 评分输出说明

**每回合输出示例**：
```
[Score][Task1] Episode 0 | 完成性 20.00 (6/6), 成功率 60.00 (6/6), 效率 18.33 (耗时 5.0s / t0 60.0s), 总分 98.33
```

**最终平均分输出**：
```
[Score][Task1] 平均分 | 完成性 19.50, 成功率 58.00, 效率 17.20, 总分 94.70
```

#### 评分机制说明

- **每回合重建 Client**：评分脚本会在每回合开始时重建 policy client，避免隐藏状态干扰，确保每回合从干净状态开始。
- **超时处理**：如果回合在 `episode_length_s` 内未完成，会自动超时并计分（效率分为 0）。
- **效率分计算**：仅在任务全部完成时计算效率分，公式为 `max((t0 - t) / t0 * 20, 0)`，其中 t 为实际耗时，t0 为基准时间。

---

## 参数说明

### 通用参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :---: |
| `--task` | 任务名称（task1/task2/task3） | 无 |
| `--device` | 计算设备（cuda/cpu） | `cuda` |
| `--num_envs` | 并行环境数量 | `1` |
| `--enable_cameras` | 启用相机 | `False` |
| `--multi_view` | 多视角显示 | `False` |
| `--enable_visualization` | 启用判定可视化 | `False` |

### 遥操作参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :---: |
| `--teleop_device` | 遥操作设备（bi_keyboard/xtrainerleader） | `bi_keyboard` |

### 推理/评分参数

| 参数 | 说明 | 默认值 |
| :--- | :--- | :---: |
| `--policy_type` | 策略类型（xtrainer_act/lerobot-xxx/openpi） | `gr00tn1.5` |
| `--policy_host` | 策略服务端地址 | `localhost` |
| `--policy_port` | 策略服务端端口 | `5555` |
| `--policy_timeout_ms` | 策略服务端超时（毫秒） | `15000` |
| `--policy_action_horizon` | 动作块大小 | `16` |
| `--policy_language_instruction` | 语言指令 | 无 |
| `--policy_checkpoint_path` | 模型检查点路径 | 无 |
| `--eval_rounds` | 评估回合数 | `0`（无限） |
| `--episode_length_s` | 单回合最大时长（秒） | `60.0` |
| `--score_t0_s` | 效率分基准时间（秒） | `episode_length_s` |

---

## 常见问题

### Q1: Isaac Sim 启动失败

**可能原因**：
- CUDA 版本不匹配
- 显卡驱动过旧
- 缺少必要的系统库

**解决方案**：
- 检查 CUDA 版本：`nvidia-smi`
- 更新显卡驱动
- 参考 [Isaac Sim 官方文档](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/requirements.html) 检查系统要求


### Q2: 真实 Leader 遥操作无法连接

**可能原因**：
- 串口未正确识别
- 零点标定未完成
- USB 权限问题

**解决方案**：
- 运行 `python scripts/find_port.py` 检查串口
- 完成零点标定：`python scripts/get_offset.py`
- 检查 USB 权限：`sudo chmod 666 /dev/ttyUSB0`（根据实际设备调整）

---

## 贡献与支持

欢迎提交 PR 与 Issue。建议流程：

1. Fork 仓库并创建特性分支
2. 遵循现有代码风格，必要时补充测试或 Demo
3. 在 PR 中说明动机与测试结果

如需反馈 bug、功能请求或寻求帮助，可直接在 GitHub Issue 中留言。

---

## 鸣谢 / 引用

本项目基于以下开源工作构建：

* [Isaac Lab / LeIsaac](https://github.com/lightwheelai/leisaac)
* [LeRobot](https://github.com/huggingface/lerobot)
* [dobot_xtrainer](https://github.com/robotdexterity/dobot_xtrainer)

如在学术工作中使用这些组件，请按照相应项目的引用要求引用。

---

## 许可证

本仓库遵循 [BSD-3-Clause License](LICENSE) 授权。

---

## 演示视频

**ACT 数据采集：**

<img src="./assets/docs/display1.gif" width="640" alt="ACT 模型演示" />

**ACT 模型演示：**

<img src="./assets/docs/display3.gif" width="640" alt="ACT 模型演示" />

**多视角相机画面（左腕 / 右腕 / 俯视）：**

<img src="./assets/docs/display2.png" width="640" alt="多视角相机画面" />

---

## 即将支持

| 头显 | 状态 |
| :-- | :-- |
| PICO 4 | 🔄 开发中 |
| Vision Pro | 🔄 开发中 |
