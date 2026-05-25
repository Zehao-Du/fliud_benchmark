# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a fork of the NVIDIA Isaac Sim repository, extended with a robotics course project: **fluid/gas-involved Franka manipulation** using IsaacSim as the physics backend. The custom code lives in `source/standalone_examples/api/isaacsim.robot.manipulators/franka/` and `tools/`.

## Running the Custom Scenes

Both custom scenes are self-contained standalone Python scripts with no build step. Run them directly with a system Python (for mock backend) or with Isaac Sim's Python runtime (for the Isaac backend):

```bash
# Liquid pick-and-place (mock backend, no Isaac required)
python source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_stir_liquid.py --backend mock

# Meat-on-grill / smoke scene (mock backend)
python source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_meat_on_grill.py --backend mock --variation chicken

# Isaac backend (requires Isaac Sim runtime)
python franka_stir_liquid.py --backend isaac --headless
python franka_meat_on_grill.py --backend isaac --preset cinematic_smoke
```

Common flags: `--preset`, `--seed`, `--backend {auto|isaac|mock}`, `--output-root`, `--disable-video`.

## Running Tests

The custom tests use standard `unittest` and require only system Python (no Isaac Sim):

```bash
# Run all custom tests
python -m pytest source/standalone_examples/testing/python_sh/test_franka_stir_liquid.py -v
python -m pytest source/standalone_examples/testing/python_sh/test_franka_meat_on_grill.py -v

# Run a single test
python -m pytest source/standalone_examples/testing/python_sh/test_franka_stir_liquid.py::FrankaStirLiquidTests::test_observation_structure -v
```

Tests use `importlib.util.spec_from_file_location` to load the scene modules directly by path — no package install needed.

## Render Tools

```bash
# Cinematic renders — 1920×1080 RTX Path Tracing (require Isaac Sim + Xvfb for smoke)
xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" conda run -n env_isaaclab \
  python tools/render_franka_meat_on_grill_cinematic.py --frames 480 --spp 16
conda run -n env_isaaclab \
  python tools/render_franka_stir_cinematic.py --frames 240 --spp 16

# NvFlow smoke requires Xvfb + headless=False. The grill cinematic script sets headless=False automatically.
# tools/bbq_smoke.usda — Fire.usda physics clone with grey BBQ smoke colormap (metersPerUnit=0.01).

# Lower quality for fast iteration
python tools/render_franka_stir_multiview_video.py --preset default --seed 0 --frames 180
python tools/render_franka_meat_on_grill_video.py --preset default --seed 0
```

## Data Collection

```bash
# Collect demo rollouts to HDF5 (requires h5py: pip install h5py)
python tools/collect_demo_data.py --scene liquid --num-episodes 50 --backend mock
python tools/collect_demo_data.py --scene grill  --num-episodes 50 --backend mock
# Output: outputs/demo_liquid.h5, outputs/demo_grill.h5
# obs_dim: liquid=10 (tool_pos+target_pos+fluid_centroid+fill_ratio), grill=14 (+ meat_pos+smoke)
# action_dim: 4 (target_ee_position + gripper_closed)
```

## Diffusion Policy

```bash
# Train
conda run -n env_isaaclab python policy/train.py --data outputs/demo_liquid.h5 --scene liquid --epochs 200
# Eval
conda run -n env_isaaclab python policy/eval_env.py --scene liquid --checkpoint policy/checkpoints/liquid/best.pt
```

`policy/` contains: `dataset.py` (HDF5 loader + normalization), `diffusion_policy.py` (1D UNet DDPM), `train.py`, `eval_env.py`.

## Architecture: Custom Scenes

Both scenes follow the same dual-backend pattern:

```
MockXXXBackend          — pure Python, zero dependencies, deterministic
    └── IsaacXXXBackend — inherits Mock, overrides _ensure_runtime(), create_scene(),
                          step_sim() to drive the real Isaac physics engine
```

`FrankaXXXEnv` is the public API that wraps either backend and exposes:
- `create_scene()` / `reset_scene(seed)` / `step_sim(n)` / `get_obs()` / `get_metrics()` / `get_events()`
- `run_episode(seed, output_dir)` — runs a full episode and writes all artifacts

**Episode artifacts** written to `outputs/<name>/run_<timestamp>_<seed>/`:
- `config.json`, `obs_schema.json`, `metrics.json`, `episode_summary.json`
- `trajectory.csv` (one row per step), `events.jsonl`, `video.mp4`, `stdout.log`

### Scene 1: `franka_stir_liquid.py`
- Franka picks a cube submerged in liquid inside a container
- Phases: `approach → descend → grasp → lift → done`
- Fluid: PhysX Particle System with CUDA (Isaac) or Python particle simulation (mock)
- Key metrics: `spill_count`, `fill_ratio`, `stirring_intensity_proxy`, `tool_contact_proxy`, `fluid_centroid_displacement`
- Config entry point: `get_preset_config(name)` — presets: `default`, `stress`

### Scene 2: `franka_meat_on_grill.py`
- Franka picks meat from a table and places it on a grill (with smoke)
- Phases: `approach → descend → grasp → lift → move_to_grill → place → retract → hold_done`
- Gas: NvFlow volumetric smoke via `tools/bbq_smoke.usda` (Isaac backend); falls back to sphere-prim particle cloud if NvFlow unavailable
- NvFlow requires `omni.flowusd` extension + `headless=False` + Xvfb for rendering
- Variations: `chicken`, `steak`; presets: `default`, `cinematic_smoke`
- Success = meat landed in grill target zone + gripper released

### Observation contract
Both scenes expose a structured `obs` dict validated by `validate_obs(obs)`. The schema is documented in `get_obs_schema()` within each scene file. Observations must not contain NaN/Inf.

## Key Design Constraints

- **Mock backend is always runnable** — never add Isaac-only imports at module top level; use `importlib.util.find_spec("isaacsim")` for runtime detection and lazy imports inside `_ensure_runtime()`.
- **Determinism**: same seed → same trajectory in mock backend. Tests rely on this.
- **Config is a plain dict** — `get_preset_config()` returns a deep-copyable dict; `_deep_update(base, overrides)` merges without mutation.
- Videos are rendered from trajectory rows using a pure-Python PPM canvas + ffmpeg. If ffmpeg is unavailable, a placeholder binary is written so artifact paths always exist.
