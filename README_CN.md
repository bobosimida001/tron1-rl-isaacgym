# TRON1 强化学习训练框架（基于 Isaac Gym）

本项目是一个基于 NVIDIA Isaac Gym 的双足/轮足机器人强化学习（RL）训练框架，支持多种 TRON1 系列机器人的运动控制策略训练与部署。

---

## 目录

1. [项目概述](#项目概述)
2. [代码结构](#代码结构)
3. [机器人型号](#机器人型号)
4. [环境与算法](#环境与算法)
5. [配置系统](#配置系统)
6. [安装说明](#安装说明)
7. [使用方法](#使用方法)
8. [已知问题](#已知问题)
9. [致谢](#致谢)

---

## 项目概述

本框架专为 TRON1 系列机器人设计，利用 Isaac Gym GPU 加速物理仿真，可同时运行多达 **8192 个并行环境**，大幅加速强化学习训练。

**主要特性：**

- 基于 **PPO（近端策略优化）** 算法的高效 RL 训练
- 支持 **多种机器人形态**：点接触双足（PF）、面接触双足（SF）、轮足（WF）
- **领域随机化（Domain Randomization）**：摩擦力、质量、PD 增益、动作延迟等
- **课程学习（Curriculum Learning）**：地形难度和速度指令自适应递增
- **ONNX/JIT 模型导出**：方便在真实机器人上部署推理策略
- **TensorBoard / Weights & Biases** 训练可视化

---

## 代码结构

```
tron1-rl-isaacgym/
├── legged_gym/
│   ├── algorithm/              # RL 算法实现
│   │   ├── ppo.py              # PPO 算法（带 KL 自适应学习率调度）
│   │   ├── actor_critic.py     # Actor-Critic 神经网络
│   │   ├── mlp_encoder.py      # MLP 编码器（处理观测历史）
│   │   ├── on_policy_runner.py # 训练循环主控（rollout 收集、梯度更新、日志）
│   │   └── rollout_storage.py  # 经验缓冲区（支持 GAE 优势估计）
│   ├── envs/                   # 仿真环境
│   │   ├── __init__.py         # 机器人类型选择与任务注册
│   │   ├── vec_env.py          # 向量化环境抽象接口
│   │   ├── base/
│   │   │   ├── base_task.py    # 核心仿真环境（1600+ 行，含奖励/观测/复位逻辑）
│   │   │   └── base_config.py  # 基础配置类
│   │   ├── pointfoot_flat/     # PF 点接触双足环境
│   │   ├── solefoot_flat/      # SF 面接触双足环境
│   │   └── wheelfoot_flat/     # WF 轮足环境
│   ├── scripts/                # 可执行脚本
│   │   ├── train.py            # 训练入口
│   │   ├── play.py             # 推理/可视化入口
│   │   └── export_policy_as_onnx.py  # 导出 ONNX 模型
│   └── utils/                  # 工具模块
│       ├── task_registry.py    # 任务注册表（工厂方法）
│       ├── helpers.py          # 参数解析、配置读写、模型加载
│       ├── terrain.py          # 地形程序化生成
│       ├── logger.py           # 训练指标日志
│       ├── math.py             # 四元数/角度等数学工具
│       └── wandb_utils.py      # Weights & Biases 集成
├── resources/
│   └── robots/                 # 机器人 URDF 模型及网格文件
├── install.sh                  # 一键安装脚本（Ubuntu 20.04）
├── setup.py                    # Python 包配置
└── README.md                   # 英文说明文档
```

---

## 机器人型号

通过环境变量 `ROBOT_TYPE` 选择机器人型号：

| 变量值       | 类型       | 说明                  |
|-------------|------------|----------------------|
| `PF_TRON1A` | 点接触双足 | TRON1A 点足版（主型） |
| `PF_P441A`  | 点接触双足 | P441 A 款             |
| `PF_P441B`  | 点接触双足 | P441 B 款             |
| `PF_P441C`  | 点接触双足 | P441 C 款             |
| `PF_P441C2` | 点接触双足 | P441 C2 款            |
| `SF_TRON1A` | 面接触双足 | TRON1A 面足版         |
| `WF_TRON1A` | 轮足       | TRON1A 轮足版         |

机器人模型文件位于 `resources/robots/<ROBOT_TYPE>/urdf/robot.urdf`。

---

## 环境与算法

### 观测空间（Observation Space）

每个时间步的观测包含（30 维）：
- 机体基座线速度（3 维）
- 机体基座角速度（3 维）
- 重力向量（3 维）
- 速度指令（3 维：$v_x, v_y, \omega_{yaw}$）
- 关节位置（6 维）
- 关节速度（6 维）
- 上一步动作（6 维）

观测历史长度为 10 帧，由 MLP 编码器压缩为 3 维隐状态，再拼接到 Actor 输入。

### 动作空间（Action Space）

6 维连续动作，分别对应双腿各 3 个关节（髋外展、髋俯仰、膝关节）的目标位置偏移，通过 PD 控制器转换为力矩输出。

### 奖励函数（Reward Function）

奖励由配置文件中 `rewards.scales` 下的各分量加权求和：

| 奖励项                         | 作用                          |
|-------------------------------|-------------------------------|
| `keep_balance`                | 保持平衡（基础存活奖励）        |
| `tracking_lin_vel`            | 跟踪线速度指令                  |
| `tracking_ang_vel`            | 跟踪偏航角速度指令              |
| `base_height`                 | 惩罚机体高度偏差                |
| `orientation`                 | 惩罚机体倾斜                    |
| `lin_vel_z`                   | 惩罚垂直方向线速度              |
| `ang_vel_xy`                  | 惩罚侧滚/俯仰角速度            |
| `torques`                     | 惩罚过大力矩                    |
| `dof_acc`                     | 惩罚关节加速度                  |
| `action_rate`                 | 惩罚动作变化率                  |
| `collision`                   | 惩罚膝/髋碰撞                   |
| `feet_distance`               | 惩罚双脚过近                    |
| `tracking_contacts_shaped_*`  | 步态接触力/速度跟踪              |

### PPO 算法参数

| 参数                  | 默认值   | 说明                           |
|----------------------|---------|-------------------------------|
| `clip_param`         | 0.2     | PPO 裁剪系数                   |
| `gamma`              | 0.99    | 折扣因子                        |
| `lam`                | 0.95    | GAE λ 参数                     |
| `learning_rate`      | 1e-3    | 学习率（自适应调度）             |
| `num_mini_batches`   | 4       | 每次更新的 mini-batch 数量      |
| `num_learning_epochs`| 5       | 每个 rollout 的梯度更新轮数     |
| `desired_kl`         | 0.01    | 目标 KL 散度（用于自适应学习率） |
| `entropy_coef`       | 0.01    | 熵正则化系数                    |
| `max_iterations`     | 15000   | 最大训练迭代次数                |

### 网络结构

- **MLP 编码器**：输入 300 维（30 × 10 帧历史）→ [256, 128] → 输出 3 维隐状态
- **Actor 网络**：[512, 256, 128]，ELU 激活，输出 6 维动作分布均值
- **Critic 网络**：[512, 256, 128]，ELU 激活，输出标量状态价值

### 领域随机化

训练时对以下参数进行随机化以增强策略泛化能力：

| 随机化项目           | 范围                |
|---------------------|---------------------|
| 地面摩擦系数         | [0.0, 1.6]          |
| 地面弹性系数         | [0.0, 1.0]          |
| 机体附加质量         | [-0.5, 5.0] kg      |
| 机体质心偏移         | ±[0.03, 0.02, 0.03] m |
| PD 刚度 $K_p$        | [0.8, 1.2] × 标称值 |
| PD 阻尼 $K_d$        | [0.8, 1.2] × 标称值 |
| 电机力矩输出         | [0.8, 1.2] × 标称值 |
| 关节默认角度偏置     | [-0.05, 0.05] rad   |
| 动作延迟             | [0, 20] ms          |
| IMU 偏置             | [-1.2, 1.2] rad     |
| 随机外力推扰         | 每 7 s，最大 1.5 m/s |

---

## 配置系统

每个环境由两个类配置（以 `pointfoot_flat` 为例）：

- **`BipedCfgPF`**（继承自 `BaseConfig`）：仿真环境参数，包括地形、指令、控制、奖励、噪声、归一化等
- **`BipedCfgPPOPF`**（继承自 `BaseConfig`）：训练参数，包括网络结构、PPO 超参数、日志配置等

可通过命令行参数覆盖配置文件中的值（见[使用方法](#使用方法)）。

---

## 安装说明

### 系统要求

- Ubuntu 20.04，x86_64
- NVIDIA GPU（支持 CUDA 12.1）
- Python 3.8（推荐）

### 方式一：一键安装脚本

```bash
bash install.sh
```

脚本将依次安装：NVIDIA 驱动、Anaconda、PyTorch、Isaac Gym 及本项目。

### 方式二：手动安装

**1. 创建并激活 Conda 虚拟环境**

```bash
conda create -n legged_gym python=3.8
conda activate legged_gym
```

**2. 安装 PyTorch 2.2.2（CUDA 12.1）**

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

**3. 安装 Isaac Gym Preview 4**

从 [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) 下载安装包后：

```bash
cd isaacgym/python && pip install -e .
# 验证安装
cd examples && python 1080_balls_of_solitude.py
```

**4. 安装辅助依赖**

```bash
pip install onnx tensorboard==2.12.0 setuptools==59.5.0
```

**5. 安装本项目**

```bash
cd tron1-rl-isaacgym && pip install -e .
```

---

## 使用方法

**重要**：所有命令执行前必须设置机器人型号环境变量：

```bash
export ROBOT_TYPE=PF_TRON1A
# 可选值：PF_TRON1A, PF_P441A, PF_P441B, PF_P441C, PF_P441C2, SF_TRON1A, WF_TRON1A
```

### 训练

```bash
python legged_gym/scripts/train.py --task=pointfoot_flat --headless
```

**常用参数：**

| 参数                              | 说明                                    |
|----------------------------------|-----------------------------------------|
| `--task TASK`                    | 任务名称（如 `pointfoot_flat`）          |
| `--headless`                     | 无头模式（不渲染，加快训练速度）          |
| `--sim_device cpu`               | 在 CPU 上运行物理仿真                    |
| `--rl_device cpu`                | 在 CPU 上运行 RL 计算                    |
| `--num_envs NUM_ENVS`            | 并行环境数量                             |
| `--seed SEED`                    | 随机种子                                 |
| `--max_iterations MAX_ITERATIONS`| 最大训练迭代次数                         |
| `--resume`                       | 从检查点恢复训练                         |
| `--load_run LOAD_RUN`            | 指定恢复训练的 run 目录（`-1` 为最新）   |
| `--checkpoint CHECKPOINT`        | 指定检查点编号（`-1` 为最新）            |
| `--experiment_name NAME`         | 实验名称                                 |
| `--run_name NAME`                | 本次 run 名称                            |

训练模型保存路径：
```
logs/<experiment_name>/<ROBOT_TYPE>/<date_time>_<run_name>/model_<iteration>.pt
```

> **提示**：训练启动后按 `v` 键可暂停渲染，显著提升训练速度。

### 推理与可视化

```bash
python legged_gym/scripts/play.py \
    --task=pointfoot_flat \
    --load_run Apr18_15-48-46_ \
    --checkpoint 10000
```

- `--load_run`：训练结果目录名，例如 `Apr18_15-48-46_`
- `--checkpoint`：对应 `model_<checkpoint>.pt` 中的数字

运行后将：
1. 在 Isaac Gym 中可视化机器人运动
2. 记录关节位置、速度、力矩、接触力等数据
3. 自动将策略导出为 JIT（`.jit`）和 ONNX（`.onnx`）格式

### 导出 ONNX 模型

```bash
python legged_gym/scripts/export_policy_as_onnx.py \
    --task=pointfoot_flat \
    --load_run Apr18_15-48-46_ \
    --checkpoint 10000
```

---

## 已知问题

1. **GPU 上三角网格地形的接触力不准确**

   当使用 `trimesh` 类型地形并在 GPU 上仿真时，`net_contact_force_tensor` 报告的接触力不可靠。  
   解决方案：为末端执行器（足部）添加力传感器，并禁用重力分量：

   ```python
   sensor_pose = gymapi.Transform()
   for name in feet_names:
       sensor_options = gymapi.ForceSensorProperties()
       sensor_options.enable_forward_dynamics_forces = False  # 排除重力
       sensor_options.enable_constraint_solver_forces = True  # 包含接触力
       sensor_options.use_world_frame = True  # 以世界坐标系报告力
       index = self.gym.find_asset_rigid_body_index(robot_asset, name)
       self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

   sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
   self.gym.refresh_force_sensor_tensor(self.sim)
   force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
   self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

   # 判断接触
   self.gym.refresh_force_sensor_tensor(self.sim)
   contact = self.sensor_forces[:, :, 2] > 1.
   ```

---

## 致谢

本项目的实现参考了以下开源工作：

- [legged_gym](https://github.com/leggedrobotics/legged_gym)（苏黎世联邦理工学院机器人系统实验室）
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)（苏黎世联邦理工学院机器人系统实验室）

特别感谢 Nikita Rudin 及 NVIDIA CORPORATION 提供的基础框架与工具。

---

## 问题反馈

如有任何问题，请在本仓库创建 Issue：[提交 Issue](../../issues/new)
