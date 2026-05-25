# RLBench `meat_on_grill` -> Isaac Sim Mapping

This demo is a semantic recreation of RLBench `meat_on_grill`, not a 1:1 asset remake.

- `chicken` / `steak` graspable object -> configurable `scene.meat_variations` entry.
- RLBench success sensor -> grill target zone centered at `scene.grill.target_position` with half extents `scene.grill.target_extent`.
- RLBench waypoint1 grill approach -> fixed `task.grill_approach_offset` waypoint above and in front of the grill.

Current scope:

- Default variation is `chicken`.
- `steak` is wired as a config entry only.
- Success means the meat is inside the grill target zone and the gripper has opened.
- Smoke is visual-only. It is attempted through `omni.flowusd` when importable, with a clear proxy fallback when scripted plume authoring is not available in the current runtime.
