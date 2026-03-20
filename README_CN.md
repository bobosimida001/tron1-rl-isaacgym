# TRON1-RL-ISAACGYM 项目分析文档

> 本文档为 tron1-rl-isaacgym 项目的中文综合分析，涵盖项目架构、算法原理、环境配置及使用方法。

---

## 目录

1. [项目概述](#1-项目概述)
2. [代码结构](#2-代码结构)
3. [支持的机器人类型](#3-支持的机器人类型)
4. [强化学习算法详解](#4-强化学习算法详解)
5. [环境与奖励函数分析](#5-环境与奖励函数分析)
6. [关键配置参数说明](#6-关键配置参数说明)
7. [安装指南](#7-安装指南)
8. [训练与使用方法](#8-训练与使用方法)
9. [已知问题](#9-已知问题)
10. [致谢](#10-致谢)

---

## 1. 项目概述

**项目名称：** tron1-rl-isaacgym（双足机器人腿部强化学习框架）

**核心目标：** 基于 NVIDIA Isaac Gym 仿真平台，使用近端策略优化（PPO）算法训练双足机器人（两足行走机器人）的运动控制策略，支持平足、整脚和轮足等多种机器人形态。

**核心技术栈：**

| 技术 | 说明 |
|------|------|
| NVIDIA Isaac Gym | 高性能 GPU 并行物理仿真平台 |
| PyTorch | 深度神经网络框架 |
| PPO 算法 | 近端策略优化（On-Policy 强化学习） |
| GAE | 广义优势估计（Generalized Advantage Estimation） |
| 领域随机化 | 提升策略鲁棒性与sim-to-real迁移能力 |

**许可证：** BSD-3-Clause

**作者：** Hongxi Wang、Nicolas Rudin（苏黎世联邦理工大学）

---

## 2. 代码结构

```
tron1-rl-isaacgym/
├── legged_gym/                    # 主程序包
│   ├── __init__.py                # 包级常量与路径定义
│   ├── algorithm/                 # 强化学习算法实现
│   │   ├── actor_critic.py        # Actor-Critic 神经网络架构
│   │   ├── mlp_encoder.py         # MLP 编码器（状态历史特征提取）
│   │   ├── on_policy_runner.py    # 训练主循环控制器
│   │   ├── ppo.py                 # PPO 算法核心实现
│   │   └── rollout_storage.py     # 轨迹回放存储缓冲区
│   ├── envs/                      # 机器人环境定义
│   │   ├── base/
│   │   │   ├── base_config.py     # 基础配置类
│   │   │   └── base_task.py       # 基础环境类（物理仿真、奖励、观测）
│   │   ├── pointfoot_flat/        # 平足机器人环境（点接触）
│   │   │   ├── pointfoot_flat.py
│   │   │   └── pointfoot_flat_config.py
│   │   ├── solefoot_flat/         # 整脚机器人环境（踝关节）
│   │   │   ├── solefoot_flat.py
│   │   │   └── solefoot_flat_config.py
│   │   ├── wheelfoot_flat/        # 轮足机器人环境（轮式足端）
│   │   │   ├── wheelfoot_flat.py
│   │   │   └── wheelfoot_flat_config.py
│   │   ├── vec_env.py             # 向量化环境包装器
│   │   └── __init__.py            # 环境注册系统
│   ├── scripts/                   # 可执行脚本
│   │   ├── train.py               # 训练入口
│   │   ├── play.py                # 策略可视化推理
│   │   └── export_policy_as_onnx.py  # 策略导出为 ONNX 格式
│   └── utils/                     # 工具函数库
│       ├── task_registry.py       # 环境与训练配置注册系统
│       ├── helpers.py             # 参数解析、检查点、配置工具
│       ├── terrain.py             # 程序化地形生成
│       ├── logger.py              # 日志系统（Tensorboard/W&B）
│       ├── math.py                # 数学工具（四元数、角度处理）
│       └── wandb_utils.py         # Weights & Biases 实验追踪
├── resources/                     # 机器人模型资源
│   └── robots/                    # 各机器人 URDF 与网格文件
│       ├── PF_P441A/              # 点足 P441 型号 A
│       ├── PF_P441B/              # 点足 P441 型号 B
│       ├── PF_P441C/              # 点足 P441 型号 C
│       ├── PF_P441C2/             # 点足 P441 型号 C2
│       ├── PF_TRON1A/             # 点足 TRON1 型号 A
│       ├── SF_TRON1A/             # 整脚 TRON1 型号 A（含踝关节）
│       └── WF_TRON1A/             # 轮足 TRON1 型号 A（轮式足端）
├── setup.py                       # 包安装配置
├── install.sh                     # 自动化安装脚本（Ubuntu 20.04）
├── README.md                      # 英文文档
└── LICENSE                        # BSD-3-Clause 许可证
```

---

## 3. 支持的机器人类型

通过环境变量 `ROBOT_TYPE` 选择机器人型号，支持三类形态：

### 3.1 平足机器人（Pointfoot，点接触）

| 型号 | 描述 | 驱动自由度 |
|------|------|----------|
| `PF_P441A` | P441 系列 A 型 | 6（每腿 3 个） |
| `PF_P441B` | P441 系列 B 型 | 6 |
| `PF_P441C` | P441 系列 C 型 | 6 |
| `PF_P441C2` | P441 系列 C2 型 | 6 |
| `PF_TRON1A` | TRON1 平足 A 型 | 6 |

每腿关节结构（从近端到远端）：
- `abad_*_Joint`：外展/内收关节
- `hip_*_Joint`：髋关节
- `knee_*_Joint`：膝关节
- `foot_*_Joint`：足端关节（被动，刚度=0）

### 3.2 整脚机器人（Solefoot，踝关节）

| 型号 | 描述 | 驱动自由度 |
|------|------|----------|
| `SF_TRON1A` | TRON1 整脚 A 型 | 8+（含踝关节） |

在平足基础上增加踝关节，脚掌面接触地面，具备更好的稳定性。

### 3.3 轮足机器人（Wheelfoot，轮式足端）

| 型号 | 描述 | 驱动自由度 |
|------|------|----------|
| `WF_TRON1A` | TRON1 轮足 A 型 | 6 |

每腿末端安装主动轮，可实现滚动与行走的混合运动模式。

---

## 4. 强化学习算法详解

### 4.1 整体训练框架

```
Isaac Gym 仿真（8192 个并行环境）
        ↓  观测 obs（30维）+ 历史obs（10帧）+ Critic观测
PPO Actor-Critic 网络
        ↓  动作 actions（6维关节角度目标）
PD 控制器 → 关节力矩
        ↓  奖励信号 reward
GAE 优势估计
        ↓
Mini-Batch SGD 更新网络参数
        ↓
循环至 max_iterations（默认 15000）
```

### 4.2 PPO（近端策略优化）

**核心目标函数：**

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 为概率比值，$\epsilon=0.2$ 为裁剪参数。

**关键超参数（来自 `BipedCfgPPOPF`）：**

| 参数 | 值 | 说明 |
|------|----|------|
| `clip_param` | 0.2 | PPO 策略更新裁剪范围 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE λ 参数 |
| `num_learning_epochs` | 5 | 每批数据训练轮次 |
| `num_mini_batches` | 4 | 小批次数量 |
| `learning_rate` | 1e-3 | 初始学习率 |
| `schedule` | adaptive | 自适应学习率调整（基于 KL 散度） |
| `desired_kl` | 0.01 | 目标 KL 散度 |
| `max_grad_norm` | 1.0 | 梯度裁剪上限 |
| `entropy_coef` | 0.01 | 探索熵正则化系数 |
| `value_loss_coef` | 1.0 | 价值函数损失权重 |

**自适应学习率规则：**
- 若 KL 散度 > `2 × desired_kl`：学习率 ÷ 1.5（下调）
- 若 KL 散度 < `0.5 × desired_kl`：学习率 × 1.5（上调）
- 学习率范围限制在 `[1e-5, 1e-2]`

### 4.3 Actor-Critic 神经网络架构

```
输入层
  ├── Actor输入：MLP编码器输出(3维) + 当前观测(30维) + 速度命令(3维) = 36维
  └── Critic输入：Critic观测(33维) + 编码器输出(3维) = 36维

Actor 网络（策略网络）：
  Linear(36) → ELU → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → Linear(6)
  输出：6维动作均值（服从正态分布采样）

Critic 网络（价值网络）：
  Linear(36) → ELU → Linear(512) → ELU → Linear(256) → ELU → Linear(128) → Linear(1)
  输出：状态价值估计 V(s)
```

动作分布：$a \sim \mathcal{N}(\mu_\theta(s),\ e^{\log\sigma})$，其中 `logstd` 为可学习参数。

### 4.4 MLP 编码器（历史状态特征提取）

```
输入：观测历史 obs_history（10帧 × 30维 = 300维）
  ↓
Linear(300) → ELU → Linear(256) → ELU → Linear(128) → Linear(3)
  ↓
输出：3维特征向量（编码当前速度状态等隐含信息）
```

编码器通过监督损失与真实速度对齐：
$$L_{encoder} = \|h_{0:3} - v_{base,0:3}\|^2$$

编码器参数由独立优化器（`est_learning_rate=1e-3`）更新，与主策略网络解耦。

### 4.5 GAE（广义优势估计）

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

- 结合了蒙特卡洛方法（低偏差）与时序差分（低方差）的优势
- `lam=0.95` 平衡了偏差-方差权衡

---

## 5. 环境与奖励函数分析

### 5.1 观测空间（30维）

| 维度 | 描述 | 缩放比例 |
|------|------|----------|
| 3 | 基座线速度 (x,y,z) | 2.0 |
| 3 | 基座角速度 (roll,pitch,yaw) | 0.25 |
| 3 | 重力投影向量 | 1.0 |
| 6 | 关节位置（相对默认值）| 1.0 |
| 6 | 关节速度 | 0.05 |
| 6 | 上一步动作 | 1.0 |
| 3 | 速度命令 (vx, vy, ωyaw) | — |

同时维护 **10帧观测历史**（300维），用于 MLP 编码器提取时序特征。

### 5.2 动作空间（6维）

输出 6 个关节角度目标（每腿3个：abad、hip、knee），通过 PD 控制器转换为力矩：

$$\tau = K_p \cdot (q_{target} - q_{current}) - K_d \cdot \dot{q}_{current}$$

- `action_scale = 0.25`（动作缩放）
- `stiffness = 42 N·m/rad`，`damping = 2.5 N·m·s/rad`
- 力矩上限：`user_torque_limit = 80 N·m`
- 最大功率：`max_power = 1000 W`
- 控制频率：仿真频率（200Hz）/ `decimation(4)` = **50Hz**

### 5.3 奖励函数分析（平足环境）

奖励由以下各项加权求和组成：

**正向奖励（激励期望行为）：**

| 奖励函数 | 权重 | 说明 |
|----------|------|------|
| `keep_balance` | +1.0 | 维持直立平衡，奖励未触发终止条件的时步 |
| `tracking_lin_vel` | +1.0 | 线速度跟踪精度 $\exp(-\|v_{cmd}-v\|^2/\sigma^2)$，$\sigma=0.2$ |
| `tracking_ang_vel` | +0.5 | 偏航角速度跟踪精度，$\sigma_{ang}=0.25$ |

**负向惩罚（抑制不良行为）：**

| 奖励函数 | 权重 | 说明 |
|----------|------|------|
| `orientation` | -10.0 | 机身俯仰/侧倾角惩罚（保持竖直） |
| `base_height` | -2.0 | 机身高度偏离目标高度(0.68m)的惩罚 |
| `feet_distance` | -100.0 | 两脚间距小于最小阈值(0.115m)的惩罚 |
| `collision` | -1.0 | 膝关节、髋关节碰撞惩罚 |
| `torques` | -8e-5 | 关节力矩能耗惩罚（鼓励省力运动） |
| `dof_acc` | -2.5e-7 | 关节加速度惩罚（平滑运动） |
| `action_rate` | -0.01 | 动作变化率惩罚（减少抖动） |
| `action_smooth` | -0.01 | 动作平滑性惩罚 |
| `lin_vel_z` | -0.5 | 垂直方向线速度惩罚 |
| `ang_vel_xy` | -0.05 | 水平面角速度惩罚 |
| `dof_pos_limits` | -2.0 | 关节角度超出软限位的惩罚 |
| `feet_regulation` | -0.05 | 步态调节惩罚 |
| `foot_landing_vel` | -0.15 | 落足速度惩罚（减少冲击） |
| `tracking_contacts_shaped_force` | -2.0 | 期望接触力形状跟踪 |
| `tracking_contacts_shaped_vel` | -2.0 | 期望接触速度形状跟踪 |

### 5.4 终止条件

- 机身与地面接触（abad、base 接触检测）
- 机身高度低于设定阈值
- 持续倾斜超过 `fail_to_terminal_time_s = 0.5` 秒
- 超过最大时步 `episode_length_s = 20` 秒

### 5.5 步态参数（Gait Parameters）

系统支持可参数化步态，4个步态参数：

| 参数 | 范围 | 说明 |
|------|------|------|
| `frequencies` | [1.5, 2.5] Hz | 步频 |
| `offsets` | [0, 1] | 相位偏移 |
| `durations` | [0.5, 0.5] | 支撑相时长比例 |
| `swing_height` | [0.0, 0.1] m | 摆动腿抬腿高度 |

---

## 6. 关键配置参数说明

### 6.1 领域随机化（Domain Randomization）

领域随机化是提升策略 sim-to-real 迁移能力的核心手段，本项目实现了全面的随机化：

| 随机化项目 | 范围/说明 |
|-----------|----------|
| 摩擦系数 | [0.0, 1.6]（均匀分布） |
| 恢复系数 | [0.0, 1.0] |
| 机身质量偏置 | [-0.5, 5.0] kg |
| 质心偏移 | ±[0.03, 0.02, 0.03] m（三轴） |
| 转动惯量 | [0.8, 1.2] 倍缩放 |
| 外力扰动 | 每7秒施加最大 1.5 m/s 速度冲击 |
| PD 增益 Kp | [0.8, 1.2] 倍随机化 |
| PD 阻尼 Kd | [0.8, 1.2] 倍随机化 |
| 电机力矩 | [0.8, 1.2] 倍随机化 |
| 默认关节角度 | [-0.05, 0.05] rad 偏置 |
| 动作延迟 | [0, 20] ms 随机 FIFO 延迟 |
| IMU 偏差 | [-1.2, 1.2] rad 随机偏移 |

### 6.2 课程学习（Curriculum Learning）

- **地形课程**：从平地逐步切换至斜面、台阶、崎岖地形
- **速度命令课程**：根据策略跟踪性能自动调整最大速度指令

```python
smooth_max_lin_vel_x = 2.0   # 最大前进速度 (m/s)
smooth_max_lin_vel_y = 1.0   # 最大侧向速度 (m/s)
max_ang_vel_yaw = 3.0        # 最大偏航角速度 (rad/s)
curriculum_threshold = 0.75  # 提升课程难度的跟踪精度阈值
```

### 6.3 仿真参数

| 参数 | 值 | 说明 |
|------|----|------|
| 仿真步长 `dt` | 0.005 s | 物理仿真频率 200 Hz |
| 控制抽取 `decimation` | 4 | 策略控制频率 50 Hz |
| 并行环境数 | 8192 | GPU 并行仿真 |
| 每次迭代步数 | 24 步 | 每环境每次 PPO 更新采集的数据量 |
| 最大训练迭代 | 15000 次 | 约等于 2.95M × 8192 = 24.2 亿步 |

---

## 7. 安装指南

### 7.1 手动安装

**步骤 1：创建 Python 虚拟环境**
```bash
conda create -n legged_gym python=3.8
conda activate legged_gym
```

**步骤 2：安装 PyTorch（CUDA 12.1）**
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121
```

**步骤 3：安装 Isaac Gym**
```bash
# 从 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 3
cd isaacgym/python && pip install -e .
# 验证安装
cd examples && python 1080_balls_of_solitude.py
```

**步骤 4：安装本框架**
```bash
git clone <此仓库>
cd tron1-rl-isaacgym
pip install -e .
```

### 7.2 自动化安装（Ubuntu 20.04 x86_64）

执行项目根目录的安装脚本（中文引导，自动处理驱动、Conda 等依赖）：
```bash
chmod +x install.sh && ./install.sh
```

脚本将自动完成：NVIDIA 驱动安装 → Conda 环境创建 → PyTorch 安装 → Isaac Gym 安装 → 本框架安装

---

## 8. 训练与使用方法

### 8.1 训练策略

```bash
# 设置机器人类型（必须）
export ROBOT_TYPE=PF_TRON1A

# 无渲染训练（推荐，速度更快）
python legged_gym/scripts/train.py --task=pointfoot_flat --headless

# 有渲染训练（可按 v 键暂停渲染提速）
python legged_gym/scripts/train.py --task=pointfoot_flat

# 在 CPU 上训练（不推荐，速度慢）
python legged_gym/scripts/train.py --task=pointfoot_flat \
    --sim_device=cpu --rl_device=cpu
```

**常用命令行参数：**

| 参数 | 说明 |
|------|------|
| `--task` | 任务名称（pointfoot_flat / solefoot_flat / wheelfoot_flat） |
| `--headless` | 无渲染模式（提升训练速度） |
| `--num_envs` | 并行环境数量 |
| `--max_iterations` | 最大训练迭代次数 |
| `--seed` | 随机种子 |
| `--resume` | 从检查点恢复训练 |
| `--load_run` | 加载的运行目录名（如 `Apr18_15-48-46_`） |
| `--checkpoint` | 模型迭代次数（如 `10000`） |
| `--experiment_name` | 实验名称 |
| `--run_name` | 运行名称 |

**模型保存路径：**
```
logs/<experiment_name>/<ROBOT_TYPE>/<日期时间>_<run_name>/model_<iteration>.pt
```

### 8.2 推理/可视化训练好的策略

```bash
export ROBOT_TYPE=PF_TRON1A
python legged_gym/scripts/play.py \
    --task=pointfoot_flat \
    --load_run Apr18_15-48-46_ \
    --checkpoint 10000
```

### 8.3 导出为 ONNX 格式（部署）

```bash
export ROBOT_TYPE=PF_TRON1A
python legged_gym/scripts/export_policy_as_onnx.py \
    --task=pointfoot_flat \
    --load_run <your_run_path> \
    --checkpoint <iteration_number>
```

### 8.4 不同机器人类型的训练命令

```bash
# 平足机器人（点接触）
export ROBOT_TYPE=PF_TRON1A
python legged_gym/scripts/train.py --task=pointfoot_flat --headless

# 整脚机器人（踝关节）
export ROBOT_TYPE=SF_TRON1A
python legged_gym/scripts/train.py --task=solefoot_flat --headless

# 轮足机器人（轮式足端）
export ROBOT_TYPE=WF_TRON1A
python legged_gym/scripts/train.py --task=wheelfoot_flat --headless
```

### 8.5 使用 Tensorboard 监控训练

训练过程中，日志自动写入 `logs/` 目录，可用 Tensorboard 实时监控：
```bash
tensorboard --logdir logs/
```

主要监控指标：
- `Train/mean_reward`：平均奖励
- `Train/mean_episode_length`：平均回合长度
- `Loss/value_function`：价值函数损失
- `Loss/surrogate`：代理策略损失
- `Loss/mean_noise_std`：动作噪声标准差
- `Perf/total_fps`：训练帧率

---

## 9. 已知问题

### GPU 三角网格地形接触力不可靠

在 GPU 上使用三角网格（trimesh）地形时，`net_contact_force_tensor` 报告的接触力不可靠。

**推荐解决方案：** 仅在足端/末端执行器添加力传感器：

```python
sensor_pose = gymapi.Transform()
for name in feet_names:
    sensor_options = gymapi.ForceSensorProperties()
    sensor_options.enable_forward_dynamics_forces = False  # 排除重力
    sensor_options.enable_constraint_solver_forces = True  # 保留接触力
    sensor_options.use_world_frame = True  # 世界坐标系报告力
    index = self.gym.find_asset_rigid_body_index(robot_asset, name)
    self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
self.gym.refresh_force_sensor_tensor(self.sim)
force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

# 接触判断
self.gym.refresh_force_sensor_tensor(self.sim)
contact = self.sensor_forces[:, :, 2] > 1.
```

---

## 10. 致谢

本项目的实现基于 [legged_gym](https://github.com/leggedrobotics/legged_gym) 和 [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 项目的资源，这两个项目由苏黎世联邦理工大学（ETH Zurich）机器人系统实验室（Robotic Systems Lab）创建。我们特别使用了其中的 `LeggedRobot` 实现。

---

## 问题反馈

如有任何疑问，欢迎在本仓库中创建 Issue。
