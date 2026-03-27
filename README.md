# 2026 Dobot 具身智能挑战赛指南

本文档于2026年3月21日更新，查看旧文档：[旧README](./README_OLD.md)

竞赛官网：https://challenge.dobot-robots.com/

钉钉群：群号170895003131，加群请备注：比赛

你可以自由发挥，实现更强大的强化学习模型。

测评端使用 lerobot 版本 0.4.2

## 比赛流程总览

```
1. 下载比赛标准环境（Isaac Sim + Isaac Lab + x-trainer 项目）
   ↓
2. 在环境中采集数据并训练你的模型
   ↓
3. 打包并提交 Docker 镜像（镜像内含推理服务端 + 模型权重）
   ↓
4. 主办方在测评机运行评分脚本，连接选手镜像的推理端口进行评分
   ↓
5. 公布成绩
```

注意：**提交物为 Docker 镜像**。镜像内需包含你的推理服务端（监听默认 5555 端口）以及模型与权重。测评时主办方在本地运行评分脚本 `policy_scoring.py`，通过端口连接到你镜像内的推理服务进行打分。请确保镜像拉取后启动即可暴露端口、且接口与 LeRobot 异步推理协议兼容（详见下文「提交与打包说明」）。

## 比赛任务简介

具体任务介绍和评分细则，请查看赛事官网中的“赛事任务”：[赛事官网](https://challenge.dobot-robots.com/)

<img src="./img/sorting2-fa2918a8.gif" style="zoom:50%;" />

Task 1: 料品分拣

- 任务：把 6 个物体分类，好的放蓝箱，坏的放红箱。
- 满分：100 分（完成性 20 分+ 成功率 60 分+ 效率 20 分）

<img src="./img/cleaning-bcbf048c.gif" style="zoom:50%;" />

Task 2: 工位清洁

- 任务：清扫 3 个物体到指定区域，并盖上盖子收集。
- 满分：100 分（收集完整性 40 分+ 覆盖准确性 40 分+ 效率 20 分）

<img src="img/tube_inserting-06de04e1.gif" alt="img/tube_inserting-06de04e1.gif" style="zoom:50%;" />

Task 3: 试管取放

- 任务：把 3 个试管插入试管架。
- 满分：100 分（插入成功 80 分+ 效率 20 分）

------

## 环境要求

### 系统要求

- 操作系统：Linux (推荐 Ubuntu 20.04/22.04)
- GPU：NVIDIA GPU，支持 CUDA（推荐 RTX 3060 或更高）
- 内存：至少 16GB RAM
- 存储：至少 50GB 可用空间

### 软件依赖

- Isaac Sim 4.5：NVIDIA 物理仿真平台
- Isaac Lab：基于 Isaac Sim 的强化学习框架（本指南以 2.1.0 为例，请以赛事或 [x-trainer dobot-challenge 分支](https://github.com/embodied-dobot/x-trainer/tree/dobot-challenge) 说明为准）
- Python 3.10
- CUDA 11.8+（与 Isaac Sim 版本匹配）
- LeRobot：用于模型训练与推理服务端（可选，仅训练/打包镜像时需要）

------

## 安装指南

### 1. 安装 Isaac Sim 和 Isaac Lab

#### 第一步：安装 Miniforge（你也可以选择 Anaconda ）

点此前往 [Miniforge GitHub 官方页面](https://github.com/conda-forge/miniforge) 下载适合你系统的版本。如Ubuntu22.04系统可选择x86_64版本下载。

#### 第二步：利用 conda 创建 Leisaac 环境

执行指令，创建 python 3.10版本的环境。

```bash
conda create -n leisaac python==3.10
```

创建完成后，可以进入创建好的Leisaac环境尝试是否正常

```bash
conda activate leisaac
```

#### 第三步：安装Isaac Sim

进入 Isaac Sim 官网网站：[点此进入](https://docs.robotsfan.com/isaacsim/4.5.0/installation/download.html#isaac-sim-latest-release)

![image](./img/lxY0Yx4eqR1ZYcV3BRMsC.png)

进入官网后，首先确认 “箭头1” 处为 4.5.0 版本，若不是，可以点击倒三角修改。

确认版本为 4.5.0 后，点击 “箭头2” 处，会开始下载 Isaac Sim 压缩包，会下载到系统默认下载路径。

压缩包下载完成后，在主目录下打开终端，运行指令创建dobot/isaacsim文件夹

```bash
mkdir ~/dobot
mkdir ~/dobot/isaacsim
```

运行指令，进入压缩包所在文件夹（在默认的下载路径）并把压缩包解压到 isaacsim 文件夹里

```bash
cd ~/下载路径
unzip "isaac-sim-standalone-4.5.0-linux-x86_64.zip" -d ~/dobot/isaacsim
```

解压完成后，进入 isaacsim 文件夹

```bash
cd ~/dobot/isaacsim
```

依次执行下面两条指令

```bash
./post_install.sh
./isaac-sim.selector.sh
```

在弹出窗口中，直接点击start，完成 Isaac Sim安装。

![image](./img/CnPVsgkEfXjg3T3zDGBPO.png)

Isaac Sim 首次打开时间较久，可能会出现“无响应”的提示，请耐心等待即可。

#### 第四步：安装Isaac lab 并配置项目

```bash
conda activate leisaac
pip install isaaclab==2.1.0 --extra-index-url  https://pypi.nvidia.com
```

- 注意：如果有报错安装失败，提示no module，可以使用下面这条指令尝试解决

```bash
pip install flatdict==4.0.1 --no-build-isolation
```

上一步安装完后，执行：

拉取项目文件

```bash
cd ~/dobot
git clone -b dobot-challenge https://github.com/embodied-dobot/x-trainer.git
```

随后

```bash
cd ~/dobot/x-trainer
pip install -e source/leisaac
```

至此，完成 Isaac Sim 和 Isaac Lab 的安装。

### 2. 下载资产并导入

请从赛事钉钉群下载资产并导入。

*注意：资产文件较大故没有放置在github中，若按上述操作从github中拉取，请从赛事官方的钉钉群中下载assets文件,解压并替换现有的assets文件夹。*

## 快速开始（验证项目是否安装完好）

#### 第一步：进入配置好的环境，并配置环境变量

依次执行下面每行指令（其中 `setup_conda_env.sh` 位于 Isaac Sim 安装目录）：

```bash
cd ~/dobot/x-trainer/
conda activate leisaac
source ~/dobot/isaacsim/setup_conda_env.sh
```

#### 第二步：打开Isaac Sim，启动键盘遥操作

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=bi_keyboard \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view  
```

可在指令中添加一行 `--enable_visualization` 参数查看评分判定区域和判定点：

- **绿色框**：良品箱
- **红色框**：不良品箱
- **黄色/橙色球体**：物体位置判定点

#### 键盘控制键位说明

布局按左右手分区：左手控制左臂，右手控制右臂；按住动作键移动，松开停止；按住 `Shift` + 键实现反向。

| 关节     | 左臂键位 | 右臂键位 | 说明                   |
| -------- | -------- | -------- | ---------------------- |
| **J1**   | `Q`      | `U`      | 按住移动，松开停止     |
| **J2**   | `W`      | `I`      | —                      |
| **J3**   | `E`      | `O`      | —                      |
| **J4**   | `A`      | `J`      | —                      |
| **J5**   | `S`      | `K`      | —                      |
| **J6**   | `D`      | `L`      | —                      |
| **夹爪** | **`G`**  | **`H`**  | **按住闭合，松开张开** |

**系统控制键**：

- `B`：开始控制
- `R`：重置

## 操作指南

### 1. 进行遥操作采集数据（可选）

如果你想自己采集数据：

#### 1.1 使用真实 X-Trainer 主臂（Leader）进行遥操作采集数据

#### 第一步：将 Leader 与电脑连接

连接机械臂的电源，并用数据线将 Leader 与电脑USB接口连接。

#### 第二步：打开读写权限

```bash
sudo chmod 777 /dev/tty
```

说明：输入该指令以开放终端设备权限，确保程序能正常访问硬件接口。

#### 第三步：进入配置好的环境，并配置环境变量

依次执行下面每行指令：

```bash
cd ~/dobot/x-trainer/
conda activate leisaac
source ~/dobot/isaacsim/setup_conda_env.sh
```

#### 第四步：初始化串口配置

```bash
python scripts/find_port.py
```

指令说明：自动识别串口连接的机械臂设备。

输出示例： ![image](./img/xHileZynSe6fpm5jBXfUE.png)

表明左右两个 Leader 串口均已检测到。

#### 第五步：零点标定

运行指令，启动遥操作功能，出现下图界面，同时 Leader 会通电启动。

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=xtrainerleader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras 
```

![image-20260131122829202_副本](https://cdn-uploads.huggingface.co/production/uploads/69610cb4cfde974962894d7f/SCiNClSjI4PK0Q7gdZaCW.png)

将 Leader 调整至与仿真环境中 Follower 初始姿态大致相同后，固定住机械臂，在终端中`Ctrl+C`关闭程序，接着输入标定指令：

```bash
python scripts/get_offset.py
```

说明：这次执行标定之后，若后续机械臂没有更换、关闭或插拔，则以后不再需要执行第五步。

#### 第六步：开始遥操作采集任务数据

输入指令：

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=xtrainerleader \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view \
    --record \
    --dataset_file=./datasets/test.hdf5
```

指令说明：要采集不同任务的数据只需要改变指令中 ` --task=task1 \` 中的1/2/3即可，task1 为料品分拣、task2为工位清洁，task3 为试管取放。

> 注意：各任务的评分判定可视化（绿色/红色判定区域框和黄色判定点球体）默认关闭，避免干扰操作。如需可视化的查看判定区域和判定点，可在指令中添加 `--enable_visualization` 参数。判定可视化仅供测试使用，请勿带可视化参数进行训练，最终评分需要关闭判定可视化进行。

#### 键位说明

系统控制键：

- `B`：开始控制
- `R`：记录任务失败并重置
- `N`：记录任务成功并重置

#### 采集流程

输入指令打开程序，鼠标先要在摄像头画面的任意位置点击一下，然后按`B`开始采集，完成一次采集之后要保留这一条数据就按`N`，不保留就按`R`，按完同时会重置任务，再按`B`开始下一条的采集。

采集完并关闭程序之后会在 `~/dobot/x-trainer/datasets` 文件夹中生成一个 `test.hdf5` 的数据文件，保存了刚才采集的所有数据，每条数据按demo_0，demo_1，...的顺序命名。

注意：下次再输入指令打开采集程序前需要将 `test.hdf5` 文件移动、删除或者改名，否则程序会为避免数据覆盖而报错打不开。

#### 1.2 使用 VR 设备进行遥操作采集数据

基于 `source/leisaac/leisaac/xtrainer_utils/XLeVR/` 中的 WebXR 服务，可直接使用 Quest 3 / PICO 4 等头显遥操作仿真双臂。

#### 第一步：安装依赖（仅首次使用时需要）

```bash
pip install -r source/leisaac/leisaac/xtrainer_utils/XLeVR/requirements.txt
```

#### 第二步：启动遥操作脚本

```bash
python scripts/environments/teleoperation/teleop_se3_agent.py \
    --task=task1 \
    --teleop_device=xtrainer_vr \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --multi_view \
    --record \
    --dataset_file=./datasets/test.hdf5
```

如只持有右手柄，可附加 `--left_disabled` 关闭左臂映射。

#### 第三步：在头显浏览器连接

```
* 头显与工作站保持同一局域网。
* 脚本启动后终端会打印访问地址，例如 `https://192.168.1.23:8443`。
* 首次访问需在 Quest Browser / PICO 浏览器中信任自签名证书，随后即可看到 `XLeVR` Web UI。
```

#### 键位说明

控制键：

- `右手 B`：开始控制并自动记录当前手柄姿态为零点。
- `左手 X`：任务失败并重置（触发 `R` 回调）。
- `左手 Y`：任务成功并重置（触发 `N` 回调）。
- 扳机（Trigger）对应夹爪闭合程度，松开即张开。

如需自定义端口或证书，可编辑同目录下的 `config.yaml` / `cert.pem` / `key.pem`。

### 2. 数据合并与转换

采集完成后，先要将 HDF5 文件转换为 LeRobot 训练所需的格式文件。

#### 第一步：准备 HDF5 文件（合并为可选）

多次采集会生成多个 HDF5 文件。可以任选其一进行转换，或将多个文件一起参与转换：

- **单文件**：在转换脚本中只填一个文件路径即可。
- **多文件**：打开 `dobot/x-trainer/scripts/convert/isaaclab2lerobot_xtrainer.py`，在 `convert_isaaclab_to_lerobot` 内将 `hdf5_files` 改为包含多个路径的列表，例如：
  `hdf5_files = [os.path.join(hdf5_root, "a.hdf5"), os.path.join(hdf5_root, "b.hdf5")]`
  脚本会按顺序读取这些文件中的 demo 并写入同一个 LeRobot 数据集。若你已有自己的合并脚本，也可先合并成一个 HDF5 再转换。

#### 第二步：修改转换参数

根据下列要求修改 `isaaclab2lerobot_xtrainer.py` 代码文件中的参数：

- 修改 `repo_id` 为转换后数据文件的输出路径以及文件名，根据你的习惯自定可用的路径。

- 修改 `hdf5_root` 为待转换的 HDF5 所在目录；修改 `file_name` 为单个文件名（当使用多文件时，见上一步的 `hdf5_files` 列表）。

- 修改 `task` 为使用该数据的任务的一句话描述（策略语言指令）：

  task1描述："Put good parts into the blue box and dirty ones into the red box."

  task2描述："Gather to the recycling area and cover with anti-static lids."

  task3描述："Insert the test tube on the desktop into the rack."

- 若只转换单个文件，修改 `file_name` 为待转换的 HDF5 文件名；若在 `hdf5_files` 中已列出多个文件，则 `file_name` 仅用于当前函数入参，可保留为其中一个文件名或与 `hdf5_files` 一致。

说明：默认的完整保存路径为 `~/.cache/huggingface/lerobot/sim_xtrainer/<repo_id 最后一段>`，例如 `repo_id='sim_xtrainer/xtrainer_task1'` 时对应目录为 `/home/dobot/.cache/huggingface/lerobot/sim_xtrainer/xtrainer_task1`。

#### 第三步：打开终端，进入 lerobot 环境：

```bash
cd ~/dobot/x-trainer
conda activate lerobot
```

#### 第四步：输入指令，开始转换

```bash
python scripts/convert/isaaclab2lerobot_xtrainer.py
```

### 3. 训练

我们在项目文件中提供了基于 LeRobot 框架的 ACT (Action Chunking with Transformers) 策略训练实现。

**若尚未安装 LeRobot 环境**：建议单独创建 conda 环境以避免与 LeIsaac 依赖冲突。可参考 [LeRobot 官方文档](https://github.com/huggingface/lerobot) 安装；本项目当前在 LeRobot v0.4.2 下验证过。安装后在该环境中执行下面的训练与推理服务端命令。

#### 第一步：打开一个终端，进入 lerobot 环境

```bash
cd ~/dobot/x-trainer/
conda activate lerobot
source ~/dobot/isaacsim/setup_conda_env.sh
```

#### 第二步：运行训练指令

```bash
  lerobot-train \
  --dataset.root=/home/dobot/.cache/huggingface/lerobot/sim_xtrainer/xtrainer_task1 \
  --dataset.repo_id=sim_xtrainer/xtrainer_task1 \
  --policy.type=act \
  --output_dir=/home/dobot/dobot/x-trainer/ckpts/act_task1 \
  --job_name=act_task1 \
  --policy.device=cuda \
  --batch_size=20 \
  --steps=200000 \
  --policy.push_to_hub=false 
```

指令说明：

- 修改 `--dataset.root` 为前面转换后数据文件的输出路径。
- 修改 `--dataset.repo_id` 为 `--dataset.root` 的后两项。
- 修改 `--output_dir` 为训练文件的输出路径和文件名。
- 修改 `--job_name` 为训练文件的文件名，即 `--output_dir` 的最后一项。
- `--batch_size` 为批处理大小，根据你的显存修改，如果爆显存了就调小。
- `--steps` 为总训练步数，步数越长，训练时间越长，根据自己的需要修改。

允许的训练操作：

- 使用任何训练框架（LeRobot、PyTorch、TensorFlow 等）
- 使用预训练模型（需注明）
- 数据增强
- 模型集成

### 4. 对推理进行任务评分

可在本项目中利用评分脚本进行评估：

#### 第一步：打开一个终端，在 LeRobot 环境启动服务端:

```bash
conda activate lerobot
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=5555 \
     --fps=30 
```

#### 第二步：再打开一个终端，进入 leisaac 环境

```bash
cd ~/dobot/x-trainer/
conda activate leisaac
source ~/dobot/isaacsim/setup_conda_env.sh
```

#### 第三步：在 leisaac 环境运行推理评估程序:

```bash
python scripts/evaluation/policy_scoring.py \
    --task=task3 \
    --eval_rounds=10 \
    --episode_length_s=60  \
    --score_t0_s=60  \
    --policy_type=xtrainer_act \
    --policy_host=localhost \
    --policy_port=5555 \
    --policy_timeout_ms=5000 \
    --policy_action_horizon=16 \
    --policy_language_instruction="Insert the test tube on the desktop into the rack." \
    --device=cuda \
    --enable_cameras \
    --policy_checkpoint_path="/home/dobot/dobot/x-trainer/ckpts/act_sort/checkpoints/100000/pretrained_model"
```

指令说明：

- 修改 `--task` 为对应的任务号。
- 修改 `--policy_language_instruction` 为任务的说明（策略语言指令）。
- 修改 `--policy_checkpoint_path` 为前面训练文件的输出路径。

> 注意：各任务的评分判定可视化（绿色/红色判定区域框和判定点球体）默认关闭，避免干扰操作。如需可视化的查看判定区域和判定点，可添加 `--enable_visualization` 参数。仅供测试使用，请勿带可视化参数进行推理验证，最终评分关闭判定可视化进行。

评分参数说明:

| 参数                 | 说明                    | 默认值 |
| -------------------- | ----------------------- | ------ |
| `--eval_rounds`      | 评分回合数              | 10     |
| `--episode_length_s` | 单回合最大时长（秒）    | 60     |
| `--score_t0_s`       | 效率分基准时间 t0（秒） | 60     |

最终评分参数可能根据实际情况调整，以官方说明为准。

执行指令后可以在终端查看输出的评分。

每回合输出示例：

```
[Score][Task1] Episode 0 | 完成性 20.00 (6/6), 成功率 60.00 (6/6), 效率 18.33 (耗时 5.0s / t0 60.0s), 总分 98.33
```

最终平均分输出示例：

```
[Score][Task1] 平均分 | 完成性 19.50, 成功率 58.00, 效率 17.20, 总分 94.70
```

### 5. 提交与打包说明（Docker 镜像）

最终提交物为 **Docker 镜像**，**仅通过赛事官网指定提交通道上传**。测评时主办方在本地运行评分脚本 `policy_scoring.py`，通过端口连接到你镜像内运行的推理服务进行打分。请严格按以下**命名与路径规范**打包，便于测评自动化。

#### 5.1 队伍编号与命名规范（必守）

- **队伍编号**：主办方将为每支队伍发放一个 **四位数编号**（如 `0001`、`0123`）。该编号用于唯一标识你的队伍，**所有提交物命名必须以该编号为前缀**。
- **提交通道**：仅通过 **赛事官网的提交通道** 上传镜像或镜像信息，不接受其他渠道。具体入口、截止时间及上传方式以官网说明为准。
- **命名格式**（以下均为强制要求，测评将按此约定执行）：

| 项目 | 格式 | 示例（假设队伍编号为 0001） |
|------|------|------------------------------|
| Docker 镜像名 | `dobot_{四位数}_submission` | `dobot_0001_submission` |
| 镜像 Tag | 按提交次数用 **`v1`、`v2`** 递增，便于区分多次提交；测评时以主办方通知为准 | `v1` |
| 镜像内模型路径（固定） | `/workspace/checkpoints/pretrained_model` | 同上，不可更改 |

- **模型路径**：为统一测评脚本参数，**镜像内模型必须且只能放在** `/workspace/checkpoints/pretrained_model`。测评时一律使用该路径作为 `--policy_checkpoint_path`，请勿使用其他路径。
- 若你需上传说明文件（如使用说明、依赖说明），请命名为：`{四位数}_readme.txt`（如 `0001_readme.txt`），便于测评方对应队伍。

#### 5.2 提交要求概述

- 镜像内需包含：**推理服务端**（与 LeRobot 异步推理接口兼容）、**模型权重与依赖**。
- 容器启动后必须 **监听 5555 端口**（默认），且服务需在 **0.0.0.0** 上监听，以便宿主机能连接。
- 测评时主办方会：从官网获取你的镜像 → 启动容器（映射 5555）→ 在宿主机运行 `policy_scoring.py`，使用 `--policy_host=localhost --policy_port=5555` 和 **固定路径** `--policy_checkpoint_path=/workspace/checkpoints/pretrained_model` 连接你的服务并评分。

#### 5.3 接口兼容性

评分脚本通过 **gRPC** 与推理服务通信，协议与 LeRobot 的 `lerobot.async_inference.policy_server` 一致。你的镜像内可以：

- **推荐**：直接使用 LeRobot 自带的推理服务端，在容器内放置好权重；客户端连接时会通过 `SendPolicyInstructions` 下发 `pretrained_name_or_path`，服务端在**容器内**从该路径加载模型。
- 或自行实现与 LeRobot 相同 gRPC 接口的服务（`Ready`、`SendPolicyInstructions`、`SendObservations`、`GetActions` 等）。

客户端（评分脚本）会向服务端发送 `PolicySetup`（包含 `policy_type`、`pretrained_name_or_path`、`actions_per_chunk`、观测/动作特征等）。**模型路径 `pretrained_name_or_path` 在服务端（容器内）解析**，因此镜像内模型必须放在你约定好的路径；测评时主办方会使用相同的路径作为 `--policy_checkpoint_path` 传入评分脚本。

#### 5.4 镜像内约定：模型路径与启动方式

为便于测评，**必须**在镜像内使用**固定模型路径**（见 5.1）：

- **模型目录（固定）**：`/workspace/checkpoints/pretrained_model`。权重文件须完整放置于该目录下，与 LeRobot ACT 等格式兼容。

容器启动时应：

1. 在该路径下放置好权重文件（与 LeRobot ACT 等格式兼容）；
2. 启动推理服务并监听 **0.0.0.0:5555**。

若使用 LeRobot 官方服务端，示例启动命令（在容器内）：

```bash
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=5555 \
    --fps=30
```

注意：`--host=0.0.0.0` 必须指定，否则宿主机无法连接。测评时使用的 `--policy_checkpoint_path` 固定为 `/workspace/checkpoints/pretrained_model`，服务端会在该路径加载模型。

#### 5.5 打包示例（Dockerfile 与运行）

以下示例中 **镜像名与模型路径均按 6.1 规范**（队伍编号以 `0001` 为例）。请将 `0001` 替换为你队编号。

```dockerfile
# 示例：基于带 Python 与 CUDA 的镜像，安装 LeRobot 并放入模型
# 镜像名须为 dobot_{四位数}_submission，模型路径须为 /workspace/checkpoints/pretrained_model
FROM nvidia/cuda:11.8-runtime-ubuntu22.04
# 或使用你已配好的 lerobot 环境镜像

WORKDIR /workspace

# 安装 Python、LeRobot 及依赖（此处省略具体 pip/conda 步骤，请按你本地环境编写）
# RUN pip install lerobot ...

# 将本地训练好的权重拷贝到镜像内固定路径（路径不可更改）
COPY ./ckpts/act_task1/checkpoints/200000/pretrained_model /workspace/checkpoints/pretrained_model

# 暴露推理端口
EXPOSE 5555

# 启动推理服务，监听 0.0.0.0
CMD ["python", "-m", "lerobot.async_inference.policy_server", "--host=0.0.0.0", "--port=5555", "--fps=30"]
```

构建与运行示例（**镜像名必须为 dobot_{队伍编号}_submission**）：

```bash
# 在 x-trainer 或你的项目目录构建，将 0001 替换为你的四位数队伍编号
docker build -t dobot_0001_submission:v1 -f Dockerfile .

# 运行容器并映射 5555 端口（自测用，Tag 如 v1 表示第一次提交）
docker run -it --rm -p 5555:5555 --gpus all dobot_0001_submission:v1
```


#### 5.6 自测方法

在提交前，建议在本地完整走通“宿主机评分脚本 + 容器内推理服务”的流程（自测时镜像名、路径须与 6.1 规范一致）：

1. **启动你的镜像**（映射 5555，镜像名与 Tag 按规范如 `dobot_0001_submission:v1`）：
   ```bash
   docker run -it --rm -p 5555:5555 --gpus all dobot_0001_submission:v1
   ```
   确认容器内服务已启动并监听 5555。

2. **在宿主机**打开另一个终端，进入已配置好的 **LeIsaac / x-trainer 环境**，运行评分脚本并指向容器端口（以 task1 为例）。**模型路径固定为 `/workspace/checkpoints/pretrained_model`**：
   ```bash
   cd ~/dobot/x-trainer/
   conda activate leisaac
   source ~/dobot/isaacsim/setup_conda_env.sh
   
   python scripts/evaluation/policy_scoring.py \
       --task=task1 \
       --eval_rounds=2 \
       --episode_length_s=60 \
       --score_t0_s=60 \
       --policy_type=xtrainer_act \
       --policy_host=localhost \
       --policy_port=5555 \
       --policy_timeout_ms=5000 \
       --policy_action_horizon=16 \
       --policy_language_instruction="Put good parts into the blue box and dirty ones into the red box." \
       --device=cuda \
       --enable_cameras \
       --policy_checkpoint_path=/workspace/checkpoints/pretrained_model
   ```

3. 若能在宿主机看到评分输出且无连接/超时错误，则说明镜像与接口兼容，可按 6.1 命名规范打包并在**官网提交通道**提交。若有报错，请检查：镜像名是否为 `dobot_{编号}_submission`、容器是否暴露 5555、服务是否监听 0.0.0.0、镜像内 `/workspace/checkpoints/pretrained_model` 下是否有完整权重文件。

#### 5.7 提交方式

- **唯一通道**：镜像（或按官网要求的镜像提交方式）**仅通过赛事官网的提交通道上传**，具体操作、截止时间以官网说明为准。
- 提交时请确保：
  - 镜像名严格为 **`dobot_{你的四位数编号}_submission`**；Tag 使用**版本号**以区分多次提交：第一次提交用 **`v1`**，第二次用 **`v2`**，依次递增（完整示例：`dobot_0001_submission:v1`）。
  - 镜像内模型已放在 **`/workspace/checkpoints/pretrained_model`**，无需再单独说明路径；

## 指令参数说明

### 通用参数

| 参数                     | 说明                          | 默认值  |
| ------------------------ | ----------------------------- | ------- |
| `--task`                 | 任务名称（task1/task2/task3） | 无      |
| `--device`               | 计算设备（cuda/cpu）          | `cuda`  |
| `--num_envs`             | 并行环境数量                  | `1`     |
| `--enable_cameras`       | 启用相机                      | `True`  |
| `--multi_view`           | 多视角显示                    | `True`  |
| `--enable_visualization` | 启用判定可视化                | `False` |

### 遥操作参数

| 参数              | 说明                                                         | 默认值        |
| ----------------- | ------------------------------------------------------------ | ------------- |
| `--teleop_device` | 遥操作设备可选键盘、主臂和 VR （bi_keyboard/xtrainerleader/xtrainer_vr） | `bi_keyboard` |

### 推理/评分参数

| 参数                            | 说明                   | 默认值         |
| ------------------------------- | ---------------------- | -------------- |
| `--policy_type`                 | 策略类型               | `xtrainer_act` |
| `--policy_host`                 | 策略服务端地址         | `localhost`    |
| `--policy_port`                 | 策略服务端端口         | `5555`         |
| `--policy_timeout_ms`           | 策略服务端超时（毫秒） | `5000`         |
| `--policy_action_horizon`       | 动作块大小             | `16`           |
| `--policy_language_instruction` | 策略语言指令           | 无             |
| `--policy_checkpoint_path`      | 模型检查点路径         | 无             |
| `--eval_rounds`                 | 评估回合数             | `10`           |
| `--episode_length_s`            | 单回合最大时长（秒）   | `60`           |
| `--score_t0_s`                  | 效率分基准时间（秒）   | `60`           |

------

## 鸣谢 / 引用

本项目基于以下开源工作构建：

- [Isaac Lab / LeIsaac](https://github.com/lightwheelai/leisaac)
- [LeRobot](https://github.com/huggingface/lerobot)
- [dobot_xtrainer](https://github.com/robotdexterity/dobot_xtrainer)

## 联系我们

如果你在参赛过程遇到任何问题，请点击加入[赛事钉钉交流群](https://www.dingtalk.com/download?action=joingroup&code=v1,k1,E1COHnERBDvBifM0WitPDmps1YyrNbGplBaQC+MF63SdR7ksupjDEA==&_dt_no_comment=1&origin=11)，或者联系我们 [challenge@dobot-robots.com](mailto:challenge@dobot-robots.com)
