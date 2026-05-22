# A Line Implementation Plan: `franka_stir_liquid`

## Goal

Build a stable, measurable, and reproducible minimum viable Isaac Sim scene named `franka_stir_liquid`.

The scene should:

- use a Franka arm with a thin stirring rod,
- interact with liquid-like PBD particles in an open square container,
- execute a fixed `approach -> dip -> stir -> lift -> done` routine,
- export structured observations, metrics, trajectories, events, and a video artifact.

The main implementation target is a standalone Python script, not an extension or plugin.

## Scope

### Main line

- PhysX PBD particles
- Franka manipulator
- open square container
- horizontal sweep stirring only
- structured observation and metric interface

### Comparison line

- a separate Flow demo for visualization and discussion only
- not part of the main observation schema or evaluation interface

## Fixed Task Definition

### Geometry

- container inner size: `0.18 x 0.18 x 0.12 m`
- wall thickness: `0.008 m`
- initial liquid height: `0.05 m`
- tool radius: `0.006 m`
- dip depth below surface: `0.015 m`

### Motion

- sweep axis: container-local `x`
- sweep half amplitude: `0.045 m`
- sweep frequency: `1.2 Hz`
- task phases: `approach -> dip -> stir -> lift -> done`

## Required Environment Interface

The standalone scene should expose the following methods:

```python
create_scene(config)
reset_scene(seed)
step_sim(num_steps=1)
apply_tool_action(action)
apply_robot_action(action)
get_obs()
close()
```

`apply_tool_action(action)` is the v1 control path. The action represents a target tool pose that is converted internally into end-effector tracking behavior.

## Observation Contract

`get_obs()` returns:

```python
obs = {
    "robot": {
        "joint_positions": ...,
        "joint_velocities": ...,
        "ee_position": ...,
        "ee_orientation": ...,
    },
    "tool": {
        "position": ...,
        "orientation": ...,
        "linear_velocity": ...,
        "angular_velocity": ...,
    },
    "container": {
        "position": ...,
        "orientation": ...,
        "inner_bounds": {
            "min": [...],
            "max": [...],
        },
    },
    "fluid": {
        "centroid": ...,
        "bbox_min": ...,
        "bbox_max": ...,
        "fill_ratio": ...,
        "spill_count": ...,
        "tool_min_dist": ...,
    },
    "task": {
        "phase": ...,
        "target_position": ...,
        "target_orientation": ...,
    },
}
```

Conventions:

- `container.inner_bounds` is always axis-aligned in the container local frame.
- spill checks are always done in the container local frame.
- `fill_ratio` is approximated with a fixed 3D occupancy grid inside the container.

## Metrics

The first version exports metrics only, not reward or success labels.

Required metrics:

- `spill_count`
- `spill_ratio`
- `in_container_particle_count`
- `fill_ratio`
- `fluid_centroid_displacement`
- `fluid_height_mean`
- `fluid_height_std`
- `tool_contact_proxy`
- `tool_path_length_in_fluid`
- `stirring_intensity_proxy`

Definitions:

- `spill_count`: particles outside `inner_bounds`
- `spill_ratio`: `spill_count / total_particle_count`
- `fluid_centroid_displacement`: distance from current centroid to reset centroid
- `fluid_height_mean/std`: statistics over in-container particle `z`
- `tool_contact_proxy`: particles within a fixed distance threshold of the tool
- `tool_path_length_in_fluid`: tool path accumulated only while in contact with the fluid
- `stirring_intensity_proxy`: mean particle speed for in-container particles

## Output Layout

Each run should write:

```text
outputs/franka_stir_liquid/
  run_<timestamp>_<seed>/
    config.json
    obs_schema.json
    metrics.json
    episode_summary.json
    trajectory.csv
    events.jsonl
    video.mp4
    stdout.log
```

`episode_summary.json.status` is restricted to:

- `completed`
- `aborted`
- `error`

## Presets

### `default`

- primary development preset
- target particle count: `2k-4k`
- moderate density
- stable stepping
- video enabled
- medium sweep parameters

### `stress`

- higher response and failure visibility
- target particle count: `4k-8k`
- slightly higher liquid level
- larger sweep amplitude and frequency
- deeper dip
- video enabled
- should produce more splashing or stronger stirring metrics than `default`

## Testing Target

The implementation is considered complete after covering:

1. headless scene startup and one full episode path
2. deterministic reset under the same seed
3. correct phase ordering and event emission
4. fixed observation schema with no NaN values
5. correct spill counting
6. contact proxy increasing when the tool enters the fluid
7. stronger stress metrics than default
8. complete output artifact generation
9. independent Flow comparison demo output

## Current Implementation Note

This repository checkout is source-oriented and does not include the generated Isaac Python launcher environment. The implementation therefore keeps two execution paths:

- an Isaac-facing standalone scene entrypoint for real runtime use,
- a pure-Python mock backend for deterministic validation of the interface, metrics, and artifact pipeline in source-only environments.
