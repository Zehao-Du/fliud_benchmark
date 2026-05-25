# FluidBench — 流体参与的机器人操作 Benchmark

> **Isaac Sim fork** | 基于 NVIDIA Isaac Sim 5.1.0，扩展了流体/气体参与的 Franka 机械臂操作场景、数据采集链路与策略学习基线。

---

## 项目计划

**目标**：构建一个涵盖流体（液体 + 气体）参与的机器人操作 Benchmark，验证流体感知对策略性能的影响，并提供完整的数据采集 → 训练 → 评估流程。

### 四条主线

| 主线 | 内容 |
|------|------|
| **仿真场景** | 两个双 backend 场景（mock 无需 Isaac；isaac 接真实物理引擎），覆盖液体（PhysX PBD 粒子）和气体（NvFlow 体积烟雾） |
| **FluidBench** | 12 个标准化任务（6 气体 + 6 液体，easy/medium/hard），统一评估接口，含 success/failure demo 视频 |
| **数据采集 + 策略学习** | 批量 rollout → HDF5 → Diffusion Policy 训练/推理全链路 |
| **基线对比** | Random / BC-NoFluid / DiffusionPolicy，量化展示流体感知的重要性 |

---

## 现有进展

### 仿真场景（双 Backend）

| 场景 | 文件 | 物理特性 |
|------|------|---------|
| 液体浸没抓取 | `source/.../franka_stir_liquid.py` | PhysX PBD 粒子；mock Python 粒子模拟 |
| 气体（BBQ 烟雾）烤架放置 | `source/.../franka_meat_on_grill.py` | NvFlow 体积烟雾；fallback 合成粒子云 |

两个场景均支持 `--backend mock`（无需 Isaac Sim，纯 Python，可跑测试/采集数据）和 `--backend isaac`（接真实物理引擎，RTX 渲染）。

```bash
python source/.../franka_stir_liquid.py --backend mock
python source/.../franka_meat_on_grill.py --backend mock --variation chicken
```

### FluidBench（12 任务，已验证）

**气体场景（烤架）：**

| 任务 ID | 难度 | 描述 |
|---------|------|------|
| `grill_chicken_standard` | Easy | 标准鸡肉放置到烤架 |
| `grill_steak_standard` | Easy | 标准牛排放置到烤架 |
| `grill_chicken_dense_smoke` | Medium | 浓烟遮挡下精确放置 |
| `grill_steak_far_grill` | Medium | 烤架距离更远 |
| `grill_chicken_offset_start` | Medium | 肉类初始位置偏移 |
| `grill_steak_narrow_target` | Hard | 目标区域缩小 40%，需精确放置 |

**液体场景（浸没抓取）：**

| 任务 ID | 难度 | 描述 |
|---------|------|------|
| `liquid_pick_standard` | Easy | 标准液位抓取 |
| `liquid_pick_shallow` | Easy | 浅液位（低阻力） |
| `liquid_pick_deep` | Medium | 深液位全浸没抓取 |
| `liquid_pick_offset_seed1` | Medium | 液体扰动导致物体偏移 |
| `liquid_pick_high_lift` | Medium | 需将物体提升超过容器边缘 |
| `liquid_pick_quick_grasp` | Hard | 极短抓取窗口 |

Oracle policy 在全部 12 个任务上成功率 **100%**（mock backend 验证）。

```bash
# 运行 benchmark（oracle 上界）
python benchmark/run_benchmark.py --policy oracle --backend mock

# 生成所有任务的 success/failure 演示视频
python tools/generate_demo_videos.py --backend mock --output-dir outputs/benchmark_videos
```

### 数据采集链路

```bash
python tools/collect_demo_data.py --scene liquid --num-episodes 100 --backend mock
python tools/collect_demo_data.py --scene grill  --num-episodes 100 --backend mock
# 输出: outputs/demo_liquid.h5, outputs/demo_grill.h5
```

HDF5 结构：`/episode_XXXX/observations/obs_vector (T,D)` + `actions/action_vector (T,4)` + `task_info`

- `obs_dim`：liquid=10，grill=14
- `action_dim`：4（EE position×3 + gripper×1）

### Diffusion Policy 基线

| 组件 | 文件 |
|------|------|
| 1D UNet DDPM（2.4M 参数） | `policy/diffusion_policy.py` |
| HDF5 数据集加载 + 归一化 | `policy/dataset.py` |
| 训练脚本（AdamW + cosine LR） | `policy/train.py` |
| 在线推理评估 | `policy/eval_env.py` |

已验证（液体场景，5 episodes，5 epochs）：loss 0.164 → 0.016，eval 3/3 成功。

```bash
conda run -n env_isaaclab python policy/train.py \
    --data outputs/demo_liquid.h5 --scene liquid --epochs 200

conda run -n env_isaaclab python policy/eval_env.py \
    --scene liquid --checkpoint policy/checkpoints/liquid/best.pt \
    --num-episodes 10 --diffusion-steps 10
```

### 高质量 3D 渲染

```bash
# 单场景渲染
xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \
  conda run -n env_isaaclab python tools/render_franka_meat_on_grill_cinematic.py

# 批量渲染所有 12 个任务（tmux 后台）
tmux new -s fluidbench_render
xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \
  conda run -n env_isaaclab python tools/render_benchmark_cinematic.py \
    --output-dir outputs/benchmark_videos_cinematic 2>&1 | tee /tmp/render_benchmark.log
```

---

## TODO（RoadMap）

### Phase 4 — 基线对比实验（优先级：高）

目标：量化展示「无流体感知」vs「有流体感知」策略的性能差距。

```
策略                        液体成功率    气体成功率    平均
─────────────────────────────────────────────────────
Random Policy                   0%          0%        0%
BC-NoFluid (无流体观测)          ~0%         ~0%      ~0%
DiffusionPolicy (50 demos)      TBD         TBD      TBD
DiffusionPolicy (200 demos)     TBD         TBD      TBD
Oracle (上界)                  100%        100%      100%
```

- [ ] 采集 200 episodes/scene 训练数据
- [ ] 实现 `policy/bc_baseline.py`（仅用 robot state，不含流体观测）
- [ ] 运行对比实验，输出成绩表

### Phase 5 — FluidDiffusionTransformer（优先级：中）

针对流体动态的时序建模：

- [ ] Fluid Dynamics LSTM：对过去 T 步流体状态建模，捕捉速度/加速度信息
- [ ] Fluid-Conditioned Cross-Attention：动作去噪网络与流体嵌入做交叉注意力
- [ ] 联合预测：同时预测机器人动作序列 + 下一步流体状态变化（辅助监督）

### Phase 6 — 泛化与迁移（优先级：低）

- [ ] 跨流体泛化：水 → 油/蜂蜜（不同粘度）策略迁移
- [ ] 部分可观测性：纯 RGB-D 图像输入（无精确流体状态）
- [ ] Sim-to-Real：Isaac PhysX PBD → 真实液体的 domain gap
- [ ] 双臂协作：两臂在流体环境中协调操作

---

## 文件结构

```
benchmark/
  tasks.py            ← 12 个任务定义
  run_benchmark.py    ← 评估 oracle/random/diffusion，输出成绩表

tools/
  collect_demo_data.py              ← 批量采集 → HDF5
  generate_demo_videos.py           ← mock 2D 演示视频
  render_benchmark_cinematic.py     ← 批量真实感 3D 渲染（tmux 后台）
  render_franka_meat_on_grill_cinematic.py  ← 单场景气体渲染
  render_franka_stir_cinematic.py           ← 单场景液体渲染
  bbq_smoke.usda                    ← NvFlow BBQ 烟雾资产

policy/
  diffusion_policy.py  ← 1D UNet DDPM
  dataset.py           ← HDF5 加载 + 归一化
  train.py             ← 训练循环
  eval_env.py          ← 在线推理评估

source/standalone_examples/api/isaacsim.robot.manipulators/franka/
  franka_meat_on_grill.py  ← 气体场景（mock + Isaac 双 backend）
  franka_stir_liquid.py    ← 液体场景（mock + Isaac 双 backend）
```

---

## 环境构建

### 1. Mock Backend（无需 Isaac Sim，仅用于快速测试/数据采集）

只需系统 Python 3.8+，安装以下依赖即可：

```bash
pip install numpy imageio imageio-ffmpeg h5py torch
```

验证：

```bash
python source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py --backend mock
# 应输出 episode summary，无报错
```

### 2. Isaac Backend + Policy 训练（conda 环境）

推荐使用独立 conda 环境（Python 3.11）：

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.0 h5py imageio imageio-ffmpeg
```

验证 torch + CUDA：

```bash
conda run -n env_isaaclab python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 期望输出: 2.x.x+cu128  True
```

训练/评估：

```bash
conda run -n env_isaaclab python policy/train.py --data outputs/demo_liquid.h5 --scene liquid --epochs 200
conda run -n env_isaaclab python policy/eval_env.py --scene liquid \
    --checkpoint policy/checkpoints/liquid/best.pt --num-episodes 10 --diffusion-steps 10
```

### 3. Isaac Sim 完整安装（用于 isaac backend + 3D 渲染）

Isaac Sim 5.1.0 需要独立安装，参考官方文档：[docs.isaacsim.omniverse.nvidia.com](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

GPU 要求：RTX 4080+（本地）或 A40+（数据中心）

NvFlow 烟雾渲染额外需要：
- Xvfb（headless 环境）：`apt-get install xvfb`
- `omni.flowusd` 扩展（Isaac Sim 内置，需在 Extension Manager 中启用）

```bash
# 安装 Xvfb
sudo apt-get install -y xvfb git-lfs

# 克隆并构建（首次约 30 分钟）
git clone https://github.com/Zehao-Du/fliud_benchmark.git isaacsim
cd isaacsim
git lfs install && git lfs pull
./build.sh
```

---

> 本仓库是 [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim) 的 fork，Isaac Sim 原始构建说明见下方。

---

<details>
<summary>Isaac Sim 原始 README（构建 / 安装说明）</summary>

### Prerequisites

- **OS**: Ubuntu 22.04 / Windows 10/11
- **GPU**: RTX 4080+ (local) or A40+ (datacenter)
- **Driver**: see [NVIDIA Driver Requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)

### Build

```bash
git clone https://github.com/isaac-sim/IsaacSim.git isaacsim
cd isaacsim
git lfs install && git lfs pull
./build.sh          # Linux
```

### Run

```bash
cd _build/linux-x86_64/release
./isaac-sim.sh
```

Full documentation: [docs.isaacsim.omniverse.nvidia.com](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)

</details>
