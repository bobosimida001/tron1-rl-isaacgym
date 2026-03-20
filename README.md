# tron1-rl-isaacgym

A reinforcement learning training framework for legged robots based on [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym). It supports training locomotion policies for multiple quadruped/biped robot morphologies (PointFoot, SoleFoot, WheelFoot) using on-policy PPO via massively parallel GPU simulation.

Built on and modified from [legged_gym](https://github.com/leggedrobotics/legged_gym) by the Robotic Systems Lab (ETH Zurich).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Supported Robots](#supported-robots)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration Guide](#configuration-guide)
- [Training Workflow](#training-workflow)
- [Known Issues](#known-issues)
- [Acknowledgment](#acknowledgment)

---

## Project Overview

This project enables researchers and engineers to train neural network locomotion policies for legged robots entirely within simulation, then deploy them to real hardware. Key features include:

- **Massively parallel simulation**: Up to 8,192 Isaac Gym environments running simultaneously on a single GPU.
- **Multiple robot morphologies**: PointFoot (tip-toe), SoleFoot (flat-foot), and WheelFoot variants of the TRON1A and P441x robot families.
- **PPO-based on-policy RL**: Proximal Policy Optimization with Generalized Advantage Estimation (GAE) and MLP encoder for observation history.
- **Terrain curriculum**: Procedurally generated flat, slope, stair, and rough-surface terrains with automatic curriculum progression.
- **Domain randomization**: Friction, mass, and perturbation randomization for sim-to-real transfer.
- **Export support**: Export trained policies to PyTorch JIT or ONNX format for deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Entry Points                         │
│          train.py / play.py / export_policy_as_onnx.py      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                     Task Registry                           │
│          task_registry.make_env / make_alg_runner           │
└──────────────────┬──────────────────┬───────────────────────┘
                   │                  │
         ┌─────────▼──────────┐  ┌───▼──────────────────────┐
         │   RL Environment   │  │    Training Config        │
         │  (BaseTask subclass)│  │  (BaseConfig subclass)   │
         │  pointfoot_flat.py │  │  pointfoot_flat_config.py│
         └─────────┬──────────┘  └──────────────────────────┘
                   │
          ┌────────▼─────────────────────────────────┐
          │            OnPolicyRunner                 │
          │  (Orchestrates training loop)             │
          └────┬────────────────┬────────────────────┘
               │                │
    ┌──────────▼──┐        ┌────▼────────────┐
    │  ActorCritic│        │   PPO Algorithm  │
    │  (Policy NN)│        │  (PPO + GAE)     │
    └──────────┬──┘        └────┬────────────┘
               │                │
    ┌──────────▼──┐        ┌────▼────────────┐
    │ MLP Encoder │        │ RolloutStorage  │
    │ (history→   │        │ (trajectory buf)│
    │  latent)    │        └─────────────────┘
    └─────────────┘
```

### Component Descriptions

| Component | File | Description |
|-----------|------|-------------|
| **BaseTask** | `envs/base/base_task.py` | Abstract base for all RL environments. Manages Isaac Gym simulation, terrain, buffers, domain randomization, and curriculum. |
| **PointFoot Env** | `envs/pointfoot_flat/pointfoot_flat.py` | Concrete env for PointFoot robots. Implements reward computation, observation extraction, torque control, and reset logic. |
| **BaseConfig** | `envs/base/base_config.py` | Recursive config base class. All environment and training configs inherit from this. |
| **ActorCritic** | `algorithm/actor_critic.py` | Actor-Critic neural network. Actor maps `(obs + latent + commands) → actions`; Critic maps `(obs + commands) → value`. Outputs a Gaussian policy for continuous action spaces. |
| **MLP Encoder** | `algorithm/mlp_encoder.py` | Encodes 300-dimensional observation history into a 3-dimensional latent vector for the policy. |
| **PPO** | `algorithm/ppo.py` | PPO algorithm with GAE, policy clipping (ε=0.2), value clipping, entropy regularization, and gradient clipping. |
| **RolloutStorage** | `algorithm/rollout_storage.py` | Circular buffer for on-policy trajectories. Supports mini-batch sampling and advantage computation. |
| **OnPolicyRunner** | `algorithm/on_policy_runner.py` | Orchestrates the training loop: rollout collection, PPO updates, logging, and checkpoint management. |
| **TaskRegistry** | `utils/task_registry.py` | Registry for environments. Maps task names to (EnvClass, EnvConfig, TrainConfig) tuples and instantiates them. |
| **Terrain** | `utils/terrain.py` | Procedural terrain generator (flat, slope, stairs, discrete obstacles) with configurable curriculum. |
| **Logger** | `utils/logger.py` | Tensorboard and Weights & Biases (WandB) integration for training metrics. |
| **Helpers** | `utils/helpers.py` | CLI argument parsing, seed management, simulation parameter setup, model export utilities. |

---

## File Structure

```
tron1-rl-isaacgym/
├── README.md
├── LICENSE                          # BSD-3-Clause
├── setup.py                         # pip-installable package (legged_gym)
├── install.sh                       # Automated setup script (Ubuntu 20.04)
├── legged_gym/
│   ├── __init__.py                  # Package root; defines LEGGED_GYM_ROOT_DIR
│   ├── scripts/
│   │   ├── train.py                 # Training entry point
│   │   ├── play.py                  # Inference / visualization entry point
│   │   └── export_policy_as_onnx.py # Export trained policy to ONNX
│   ├── algorithm/
│   │   ├── actor_critic.py          # Actor-Critic network (192 lines)
│   │   ├── mlp_encoder.py           # Observation history encoder (127 lines)
│   │   ├── on_policy_runner.py      # Training orchestration (376 lines)
│   │   ├── ppo.py                   # PPO algorithm (334 lines)
│   │   └── rollout_storage.py       # Trajectory buffer (312 lines)
│   ├── envs/
│   │   ├── __init__.py              # Task registration (registers all tasks)
│   │   ├── vec_env.py               # Abstract VecEnv interface
│   │   ├── base/
│   │   │   ├── base_task.py         # Base RL task (1631 lines)
│   │   │   └── base_config.py       # Base configuration class
│   │   ├── pointfoot_flat/
│   │   │   ├── pointfoot_flat.py    # PointFoot environment (419 lines)
│   │   │   └── pointfoot_flat_config.py  # PointFoot config (388 lines)
│   │   ├── solefoot_flat/
│   │   │   ├── solefoot_flat.py     # SoleFoot environment
│   │   │   └── solefoot_flat_config.py
│   │   └── wheelfoot_flat/
│   │       ├── wheelfoot_flat.py    # WheelFoot environment
│   │       └── wheelfoot_flat_config.py
│   └── utils/
│       ├── task_registry.py         # Environment registry (227 lines)
│       ├── helpers.py               # CLI, model I/O, utilities
│       ├── logger.py                # Training logger (Tensorboard / WandB)
│       ├── terrain.py               # Terrain generation
│       ├── math.py                  # Math utilities (quaternions, etc.)
│       └── wandb_utils.py           # WandB helper
└── resources/
    └── robots/                      # URDF files and STL meshes
        ├── PF_TRON1A/               # PointFoot TRON1A
        ├── PF_P441A/                # PointFoot P441A
        ├── PF_P441B/                # PointFoot P441B
        ├── PF_P441C/                # PointFoot P441C
        ├── PF_P441C2/               # PointFoot P441C2
        ├── SF_TRON1A/               # SoleFoot TRON1A
        ├── WF_TRON1A/               # WheelFoot TRON1A
        └── meshes_camera/           # Camera mesh files
```

---

## Supported Robots

| `ROBOT_TYPE` | Morphology | DOF | Description |
|---|---|---|---|
| `PF_TRON1A` | PointFoot | 6 | TRON1A biped with tip-point feet |
| `PF_P441A` | PointFoot | 6 | P441A variant |
| `PF_P441B` | PointFoot | 6 | P441B variant |
| `PF_P441C` | PointFoot | 6 | P441C variant |
| `PF_P441C2` | PointFoot | 6 | P441C2 variant |
| `SF_TRON1A` | SoleFoot | — | TRON1A biped with flat-sole feet |
| `WF_TRON1A` | WheelFoot | — | TRON1A biped with wheeled feet |

Each robot has 3 DOF per leg (hip abduction/adduction, hip flexion/extension, knee flexion/extension) and is loaded from a URDF under `resources/robots/<ROBOT_TYPE>/urdf/`.

---

## Installation

### Prerequisites
- Ubuntu 20.04, x86_64
- NVIDIA GPU with CUDA 12.1 support
- Anaconda / Miniconda

### Quick Setup (Ubuntu 20.04)

```bash
bash install.sh
```

### Manual Setup

1. **Create a conda environment (Python 3.8 recommended):**
   ```bash
   conda create -n legged_gym python=3.8
   conda activate legged_gym
   ```

2. **Install PyTorch with CUDA 12.1:**
   ```bash
   pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
       --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install Isaac Gym Preview 4:**
   - Download from https://developer.nvidia.com/isaac-gym
   ```bash
   cd isaacgym/python && pip install -e .
   # Verify: cd examples && python 1080_balls_of_solitude.py
   ```

4. **Install this package:**
   ```bash
   git clone <this-repo>
   cd tron1-rl-isaacgym
   pip install -e .
   ```

---

## Usage

### Training

```bash
export ROBOT_TYPE=PF_TRON1A
python legged_gym/scripts/train.py --task=pointfoot_flat --headless
```

**Common flags:**

| Flag | Description |
|------|-------------|
| `--task TASK` | Task name (e.g., `pointfoot_flat`) |
| `--sim_device cuda:0` | Physics simulation device |
| `--rl_device cuda:0` | RL computation device |
| `--num_envs N` | Number of parallel environments (default: 8192) |
| `--seed N` | Random seed |
| `--max_iterations N` | Max training iterations (default: 15000) |
| `--headless` | Disable rendering (recommended for speed) |
| `--resume` | Resume from a checkpoint |
| `--experiment_name NAME` | Experiment name for logging |
| `--run_name NAME` | Run name for logging |
| `--load_run NAME` | Run folder to resume from (use `-1` for latest) |
| `--checkpoint N` | Checkpoint iteration to load (use `-1` for latest) |

> **Tip:** After training starts, press `v` to toggle rendering off for maximum performance.

Checkpoints are saved to:
```
logs/<experiment_name>/<ROBOT_TYPE>/<date_time>_<run_name>/model_<iteration>.pt
```

### Inference / Visualization

```bash
python legged_gym/scripts/play.py \
    --task=pointfoot_flat \
    --load_run Apr18_15-48-46_ \
    --checkpoint 10000
```

### Export Policy to ONNX

```bash
python legged_gym/scripts/export_policy_as_onnx.py \
    --task=pointfoot_flat \
    --load_run Apr18_15-48-46_ \
    --checkpoint 10000
```

---

## Configuration Guide

Each task is defined by two configuration classes in its `*_config.py` file:

### Environment Config (`BipedCfgPF`)

| Section | Key Parameters |
|---------|---------------|
| `env` | `num_envs=8192`, `num_observations=30`, `num_actions=6`, `episode_length_s=20` |
| `terrain` | `mesh_type='plane'`, `friction_range=[0.4, 0.4]`, `curriculum=True` |
| `commands` | 3 command dims: `lin_vel_x`, `lin_vel_y`, `ang_vel_yaw` |
| `rewards` | Weighted reward scales for velocity tracking, energy, smoothness, stability |
| `normalization` | Observation clip values and scaling factors |
| `domain_rand` | Push intervals, friction randomization, mass randomization |
| `sim` | `dt=0.005`, PhysX solver, gravity, substeps |

### Training Config (`BipedCfgPPOPF`)

| Section | Key Parameters |
|---------|---------------|
| `MLP_Encoder` | Obs history → 3D latent: hidden dims `[256, 128]` |
| `policy` | Actor/Critic hidden dims `[512, 256, 128]`, ELU activation |
| `algorithm` | `gamma=0.99`, `lam=0.95`, `clip_param=0.2`, `learning_rate=1e-3` |
| `runner` | `max_iterations=15000`, `num_steps_per_env=24`, `save_interval=500` |

### Reward Functions

Each non-zero reward scale in `cfg.rewards.scales` automatically registers a reward function of the same name in the environment. Reward signals include:

- **Tracking**: `tracking_lin_vel`, `tracking_ang_vel`
- **Penalties**: `torques`, `dof_vel`, `dof_acc`, `action_rate`
- **Stability**: `base_height`, `lin_vel_z`, `ang_vel_xy`, `orientation`
- **Foot contact**: `feet_air_time`, `collision`, `stumble`
- **Smoothness**: `stand_still`

---

## Training Workflow

```
1. Initialization
   └─ Isaac Gym creates N parallel physics environments
   └─ URDF loaded, terrain generated, buffers allocated

2. Network Setup
   └─ MLP Encoder: observation_history (300D) → latent (3D)
   └─ Actor:  (obs + latent + commands) → actions (6D)
   └─ Critic: (obs + commands) → value (1D)
   └─ Adam optimizer initialized

3. Training Loop  (repeated for max_iterations)
   ├─ a) Rollout Collection  (num_steps_per_env=24 steps × 8192 envs)
   │      Actor samples actions → environment steps → rewards computed
   │      Transitions stored in RolloutStorage
   │
   ├─ b) Advantage Estimation
   │      GAE computed using Critic value predictions
   │
   ├─ c) PPO Update  (num_learning_epochs=5, num_mini_batches=4)
   │      Policy gradient loss with clipping
   │      Value loss with optional clipping
   │      Entropy bonus for exploration
   │      Gradient clipping (max_grad_norm=1.0)
   │
   └─ d) Logging & Checkpoints
          Metrics → Tensorboard / WandB
          Model saved every save_interval=500 iterations
```

---

## Known Issues

### 1. GPU Contact Force Inaccuracy

The contact forces reported by `net_contact_force_tensor` are unreliable when simulating on GPU with a triangle mesh terrain. A workaround is to use force sensors attached to the feet only, with gravity excluded:

```python
sensor_pose = gymapi.Transform()
for name in feet_names:
    sensor_options = gymapi.ForceSensorProperties()
    sensor_options.enable_forward_dynamics_forces = False  # exclude gravity
    sensor_options.enable_constraint_solver_forces = True  # include contacts
    sensor_options.use_world_frame = True
    index = self.gym.find_asset_rigid_body_index(robot_asset, name)
    self.gym.create_asset_force_sensor(robot_asset, index, sensor_pose, sensor_options)

sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
self.gym.refresh_force_sensor_tensor(self.sim)
force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

# Later in the loop:
self.gym.refresh_force_sensor_tensor(self.sim)
contact = self.sensor_forces[:, :, 2] > 1.
```

---

## Acknowledgment

The implementation relies on resources from [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl) projects, created by the Robotic Systems Lab (ETH Zurich). We specifically utilize the `LeggedRobot` implementation from their research to enhance our codebase.

## Any Questions?

If you have any more questions, please create an issue in this repository.
