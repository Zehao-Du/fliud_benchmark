# STATUS.md — 已完成内容

> 最后更新：2026-05

---

## 已完成模块

### 1. 仿真场景（双 Backend）

| 文件 | 功能 | 状态 |
|------|------|------|
| `source/.../franka_meat_on_grill.py` | 气体参与的烤架场景（NvFlow 烟雾 + mock fallback） | ✅ 完成 |
| `source/.../franka_stir_liquid.py` | 液体参与的搅拌场景（PhysX PBD + mock Python 粒子） | ✅ 完成 |
| `tools/bbq_smoke.usda` | NvFlow BBQ 烟雾资产（原生 cm 坐标系，可直接使用） | ✅ 完成 |

两个场景均实现 `MockBackend` + `IsaacBackend` 双路，`--backend mock` 无需 Isaac Sim 即可运行。

---

### 2. 高质量渲染

| 文件 | 功能 | 状态 |
|------|------|------|
| `tools/render_franka_meat_on_grill_cinematic.py` | 1920×1080 RTX 渲染，NvFlow 烟雾后期合成 | ✅ 完成 |
| `tools/render_franka_stir_cinematic.py` | 1920×1080 RTX Path Tracing，液体 isosurface | ✅ 完成 |
| `tools/render_franka_meat_on_grill_video.py` | 快速 mock 渲染（测试用） | ✅ 完成 |
| `tools/render_franka_stir_multiview_video.py` | 多视角液体场景视频 | ✅ 完成 |
| `outputs/franka_meat_on_grill_cinematic/` | 已渲染的烤架场景电影级视频（含烟雾合成） | ✅ 产出 |

**NvFlow 烟雾说明**：NvFlow 只在原生 cm 坐标系（metersPerUnit=0.01）下正确渲染；在 meters 坐标系中 reference 会产生 10m 半径均匀雾。当前方案：独立渲染烟雾帧（`/tmp/composite_smoke_frames/`）+ 加法合成到烤架场景帧。

---

### 3. Benchmark（FluidBench）

| 文件 | 功能 | 状态 |
|------|------|------|
| `benchmark/tasks.py` | 12 个任务定义（6 烤架 + 6 液体，easy/medium/hard） | ✅ 完成 |
| `benchmark/run_benchmark.py` | 评估脚本（oracle / random / diffusion policy） | ✅ 完成 |
| `tools/generate_demo_videos.py` | 为每个任务生成 success + failure 演示视频 | ✅ 完成 |

**验证结果**：Oracle policy 在所有 12 个任务上成功率 100%（mock backend）。

#### FluidBench 任务列表

**气体参与（烤架）：**
1. `grill_chicken_standard` — easy：标准烤鸡任务
2. `grill_steak_standard` — easy：标准烤牛排任务
3. `grill_chicken_dense_smoke` — medium：浓烟遮挡下的放置
4. `grill_steak_far_grill` — medium：烤架距离较远
5. `grill_chicken_offset_start` — medium：肉类初始位置偏移
6. `grill_steak_narrow_target` — hard：精确放置到缩小目标区域

**液体参与：**
1. `liquid_pick_standard` — easy：标准液位抓取
2. `liquid_pick_shallow` — easy：浅液位（低阻力）
3. `liquid_pick_deep` — medium：深液位全浸没抓取
4. `liquid_pick_offset_seed1` — medium：液体扰动导致物体偏移
5. `liquid_pick_high_lift` — medium：需将物体提升超过容器边缘
6. `liquid_pick_quick_grasp` — hard：极短抓取窗口

---

### 4. 数据采集链路

| 文件 | 功能 | 状态 |
|------|------|------|
| `tools/collect_demo_data.py` | 批量 rollout → HDF5，支持两个场景 | ✅ 完成 |

**验证结果**：
- 液体场景：5 episodes，5/5 成功，~7s/episode（mock，particle_spacing=0.018）
- 烤架场景：5 episodes，5/5 成功，~15s/episode（mock）

HDF5 结构：
```
demo_<scene>.h5
  /metadata      — scene_type, obs_dim, action_dim, norm stats
  /episode_XXXX/
    observations/obs_vector   (T, D) float32
    actions/action_vector     (T, 4) float32
    task_info/                phases, success, seed
```

---

### 5. Diffusion Policy

| 文件 | 功能 | 状态 |
|------|------|------|
| `policy/diffusion_policy.py` | 1D UNet DDPM（2.4M 参数），`predict_action(num_steps=)` | ✅ 完成 |
| `policy/dataset.py` | HDF5 数据集加载，obs/action 归一化 | ✅ 完成 |
| `policy/train.py` | AdamW + cosine LR，best checkpoint 保存 | ✅ 完成 |
| `policy/eval_env.py` | 在线推理，`--diffusion-steps 10`（CPU 1.7s/次） | ✅ 完成 |

**验证结果**（液体场景，5 episodes 训练数据，5 epochs）：
- 训练损失：0.164 → 0.016
- Eval 成功率：3/3（100%，与 oracle 相同轨迹）

**注**：当前 eval 成功是因为训练数据覆盖了测试 seed 轨迹（脚本化 policy 确定性）。Phase 4 将用 seed offset 测试真正的泛化能力。

---

## 待完成（RoadMap）

详见 `docs/ROADMAP.md`。

| 优先级 | 任务 | 预估工作量 |
|--------|------|-----------|
| 高 | 运行 Phase 4 baseline 对比实验（200 demos，random vs diffusion） | 1-2 天 |
| 高 | 实现 BC-NoFluid baseline（无流体信息的 BC 策略） | 0.5 天 |
| 中 | 扩大训练数据集（500+ demos/scene） | 计算时间 |
| 中 | FluidDiffusionTransformer（流体状态编码 + 时序 LSTM） | 1 周 |
| 低 | VisionFluidBench（RGB-D 输入） | 2 周 |
| 低 | Sim-to-Real transfer 实验 | 硬件依赖 |

---

## 运行命令速查

```bash
# 生成所有演示视频（12 任务 × success+failure）
python tools/generate_demo_videos.py --backend mock --output-dir outputs/benchmark_videos

# 运行 benchmark（oracle 上界）
python benchmark/run_benchmark.py --policy oracle --backend mock

# 采集数据
python tools/collect_demo_data.py --scene liquid --num-episodes 100 --backend mock
python tools/collect_demo_data.py --scene grill  --num-episodes 100 --backend mock

# 训练 policy
conda run -n env_isaaclab python policy/train.py \
    --data outputs/demo_liquid.h5 --scene liquid --epochs 200

# 评估（快速推理：10 diffusion steps ≈ 1.7s/action on CPU）
conda run -n env_isaaclab python policy/eval_env.py \
    --scene liquid --checkpoint policy/checkpoints/liquid/best.pt \
    --num-episodes 10 --diffusion-steps 10
```
