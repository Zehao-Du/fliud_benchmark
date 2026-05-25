# FluidBench RoadMap

> **项目**：流体参与的机器人操作 — Benchmark、数据采集与策略学习  
> **代码库**：`isaacsim` (fork of NVIDIA Isaac Sim)  
> **最后更新**：2026-05

---

## 已完成

### Phase 0 — 仿真场景基础设施

两个完整的双 Backend 场景（`mock` 无需 Isaac；`isaac` 接入真实物理引擎）：

| 场景 | 文件 | 物理特性 |
|------|------|---------|
| 气体（BBQ 烟雾）烤架放置 | `source/.../franka_meat_on_grill.py` | NvFlow 体积烟雾；fallback 合成粒子云 |
| 液体（PhysX PBD）浸没抓取 | `source/.../franka_stir_liquid.py` | PhysX 粒子系统；fallback Python 粒子模拟 |

**NvFlow 烟雾渲染说明**：NvFlow 只在原生 cm 坐标系（metersPerUnit=0.01）下正确运行；将其 reference 进 meters 舞台会产生 10m 半径均匀灰雾。当前方案：独立 cm 舞台渲染烟雾帧 → 加法合成到场景帧（见 `render_franka_meat_on_grill_cinematic.py`）。

### Phase 1 — FluidBench（已验证）

| 组件 | 文件 | 说明 |
|------|------|------|
| 任务定义 | `benchmark/tasks.py` | 12 个任务（6 气体 + 6 液体），easy/medium/hard |
| 评估运行器 | `benchmark/run_benchmark.py` | 支持 oracle / random / diffusion policy |
| 演示视频生成 | `tools/generate_demo_videos.py` | 每任务输出 success.mp4 + failure.mp4 |

**Oracle 验证**：12/12 = 100% 成功率（mock backend 确定性脚本策略上界）。

#### 任务列表

**气体参与（烤架，6 个任务）：**

| 任务 ID | 难度 | 描述 |
|---------|------|------|
| `grill_chicken_standard` | Easy | 标准鸡肉放置到烤架 |
| `grill_steak_standard` | Easy | 标准牛排放置到烤架 |
| `grill_chicken_dense_smoke` | Medium | 浓烟遮挡下精确放置 |
| `grill_steak_far_grill` | Medium | 烤架距离更远（较长运输路径） |
| `grill_chicken_offset_start` | Medium | 肉类初始位置横向偏移 |
| `grill_steak_narrow_target` | Hard | 目标区域缩小 40%，需精确放置 |

**液体参与（浸没抓取，6 个任务）：**

| 任务 ID | 难度 | 描述 |
|---------|------|------|
| `liquid_pick_standard` | Easy | 标准液位抓取 |
| `liquid_pick_shallow` | Easy | 浅液位（低阻力） |
| `liquid_pick_deep` | Medium | 深液位全浸没抓取 |
| `liquid_pick_offset_seed1` | Medium | 液体扰动导致物体偏移 |
| `liquid_pick_high_lift` | Medium | 需将物体提升超过容器边缘 |
| `liquid_pick_quick_grasp` | Hard | 极短抓取窗口 |

**演示视频**（已生成，mock 2D 俯视渲染）：
```
outputs/benchmark_videos/
  grill/<task_id>/success.mp4  failure.mp4   (×6)
  liquid/<task_id>/success.mp4 failure.mp4   (×6)
```
高质量 3D 渲染视频：见 Phase 1.5。

### Phase 2 — 数据采集链路

```bash
python tools/collect_demo_data.py --scene liquid --num-episodes 100 --backend mock
python tools/collect_demo_data.py --scene grill  --num-episodes 100 --backend mock
# 输出: outputs/demo_liquid.h5, outputs/demo_grill.h5
```

HDF5 结构：`/episode_XXXX/observations/obs_vector (T,D)` + `actions/action_vector (T,4)` + `task_info`  
obs_dim: liquid=10，grill=14；action_dim=4（EE position×3 + gripper×1）

### Phase 3 — Diffusion Policy 基线

| 组件 | 文件 | 参数量 |
|------|------|--------|
| 1D UNet DDPM | `policy/diffusion_policy.py` | 2.4M |
| HDF5 数据集加载 | `policy/dataset.py` | — |
| 训练脚本 | `policy/train.py` | AdamW + cosine LR |
| 在线推理评估 | `policy/eval_env.py` | 10步去噪 ≈ 1.7s/action (CPU) |

**已验证**（液体场景，5 episodes，5 epochs）：loss 0.164→0.016，eval 3/3 成功。

---

## 进行中

### Phase 1.5 — 高质量 3D 渲染视频（后台运行中）

脚本 `tools/render_benchmark_cinematic.py` 在 tmux 后台批量渲染所有任务的真实感视频：

- 分辨率：1920×1080，RTX RaytracedLighting，spp=16
- NvFlow 体积烟雾合成（气体场景）
- 估算时间：~2 小时（所有 12 个任务 × success+failure，单次 Isaac Sim 会话）
- 输出：`outputs/benchmark_videos_cinematic/<task_id>/success.mp4`

查看进度：
```bash
tmux attach -t fluidbench_render
# 或查看日志
tail -f /tmp/render_benchmark.log
```

---

## 待完成

### Phase 4 — 基线对比实验（优先级：高）

**目标**：量化展示"无流体感知"策略 vs "有流体感知"策略的性能差距。

```
策略                        液体成功率    气体成功率    平均
─────────────────────────────────────────────────────
Random Policy                   0%          0%        0%
BC-NoFluid (无流体观测)          ~0%         ~0%      ~0%
DiffusionPolicy (50 demos)      TBD         TBD      TBD
DiffusionPolicy (200 demos)     TBD         TBD      TBD
Oracle (上界)                  100%        100%      100%
```

实现步骤：
```bash
# 1. 采集大规模数据
python tools/collect_demo_data.py --scene liquid --num-episodes 200 --backend mock
python tools/collect_demo_data.py --scene grill  --num-episodes 200 --backend mock

# 2. 训练
conda run -n env_isaaclab python policy/train.py --data outputs/demo_liquid.h5 --scene liquid --epochs 200
conda run -n env_isaaclab python policy/train.py --data outputs/demo_grill.h5  --scene grill  --epochs 200

# 3. 跑 benchmark 对比
python benchmark/run_benchmark.py --policy random    --output outputs/bench_random.json
python benchmark/run_benchmark.py --policy diffusion \
    --liquid-ckpt policy/checkpoints/liquid/best.pt \
    --grill-ckpt  policy/checkpoints/grill/best.pt  \
    --output outputs/bench_diffusion.json
```

还需实现 `policy/bc_baseline.py`（仅用 robot state 观测，不含流体信息），对比展示流体感知的重要性。

### Phase 5 — FluidDiffusionTransformer（优先级：中）

**动机**：当前 Diffusion Policy 将流体状态作为平坦向量（`fluid_centroid + fill_ratio`）处理，缺乏时序动态建模。

**架构设计**：

```
观测输入:
  robot_joints (9) + ee_pos (3)      → Linear projection
  fluid_centroid (3) + fill_ratio (1) ──┐
  [可选] particle_cloud (N×3)          ├─ Fluid Encoder (LSTM + MLP)
  [可选] smoke_density_voxel (8³)      ──┘
                                         ↓
                                 Fused Embedding (D=256)
                                         │
                               1D UNet Diffusion (条件去噪)
                                         │
                                 Action Sequence (Ta×4)
```

关键创新：
- **Fluid Dynamics LSTM**：对过去 To 步的流体状态建模，捕捉速度/加速度信息
- **Fluid-Conditioned Cross-Attention**：动作去噪网络与流体嵌入做交叉注意力
- **联合预测**：同时预测机器人动作序列和下一步流体状态变化（辅助监督）

### Phase 6 — 泛化与迁移（优先级：低）

| 方向 | 描述 | 依赖 |
|------|------|------|
| 跨流体泛化 | 水→油/蜂蜜（不同粘度）策略迁移 | Isaac PhysX 不同流体材质配置 |
| 部分可观测性 | 纯 RGB-D 图像输入（无精确流体状态） | 视觉编码器（ResNet/ViT） |
| Sim-to-Real | Isaac PhysX PBD → 真实液体的 domain gap | 真实机器人硬件 |
| 双臂协作 | 两臂在流体环境中协调操作 | 双臂场景设计 |

---

## 快速上手

```bash
# 运行 benchmark（oracle 上界）
python benchmark/run_benchmark.py --policy oracle --backend mock

# 生成演示视频（mock 2D 俯视，快速）
python tools/generate_demo_videos.py --backend mock --output-dir outputs/benchmark_videos

# 高质量 3D 渲染（Isaac + NvFlow，在 tmux 后台）
tmux new -s fluidbench_render
xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \
  conda run -n env_isaaclab python tools/render_benchmark_cinematic.py \
    --output-dir outputs/benchmark_videos_cinematic 2>&1 | tee /tmp/render_benchmark.log

# 采集数据 + 训练
python tools/collect_demo_data.py --scene liquid --num-episodes 100 --backend mock
conda run -n env_isaaclab python policy/train.py \
    --data outputs/demo_liquid.h5 --scene liquid --epochs 200

# 评估
conda run -n env_isaaclab python policy/eval_env.py \
    --scene liquid --checkpoint policy/checkpoints/liquid/best.pt \
    --num-episodes 10 --diffusion-steps 10
```

---

## 文件结构

```
benchmark/
  tasks.py            ← 12 个任务定义（success + failure config）
  run_benchmark.py    ← 评估 oracle/random/diffusion，输出成绩表

tools/
  collect_demo_data.py              ← 批量采集 → HDF5
  generate_demo_videos.py           ← mock 2D 演示视频（fast）
  render_benchmark_cinematic.py     ← 批量真实感 3D 渲染（Isaac + NvFlow）
  render_franka_meat_on_grill_cinematic.py  ← 单场景气体渲染
  render_franka_stir_cinematic.py           ← 单场景液体渲染
  bbq_smoke.usda                    ← NvFlow BBQ 烟雾资产（cm 坐标系）

policy/
  diffusion_policy.py  ← 1D UNet DDPM，predict_action(num_steps=)
  dataset.py           ← HDF5 加载 + 归一化
  train.py             ← 训练循环
  eval_env.py          ← 在线推理评估

source/standalone_examples/api/isaacsim.robot.manipulators/franka/
  franka_meat_on_grill.py  ← 气体场景（mock + Isaac 双 backend）
  franka_stir_liquid.py    ← 液体场景（mock + Isaac 双 backend）

outputs/
  benchmark_videos/               ← mock 2D 演示视频（已生成）
  benchmark_videos_cinematic/     ← 3D 渲染视频（渲染中）
  franka_meat_on_grill_cinematic/ ← 单场景样例视频（已生成）
```
