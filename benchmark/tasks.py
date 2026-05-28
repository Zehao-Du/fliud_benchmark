"""Benchmark task definitions for fluid-involved robot manipulation.

Each BenchmarkTask specifies a scene, config overrides, and expected success criteria.
Three groups cover gas-involved (grill), liquid-involved, and steam/spray manipulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkTask:
    task_id: str
    scene: str                  # "liquid" | "grill" | "steam"
    description: str
    difficulty: str             # "easy" | "medium" | "hard"
    preset: str = "default"
    variation: str = "chicken"  # grill only
    task_type: str = ""         # steam only: one of TASK_TYPES in franka_steam_tasks.py
    seed: int = 0
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    # Config overrides that make the scripted policy FAIL (for failure demo videos)
    failure_overrides: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.task_id.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Gas-involved tasks (grill / smoke)
# ---------------------------------------------------------------------------

GRILL_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="grill_chicken_standard",
        scene="grill",
        description="Pick raw chicken from table and place it on the BBQ grill under smoke.",
        difficulty="easy",
        variation="chicken",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Robot descends 50 cm above meat → grasp distance > 0.055 → never grasps
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="grill_steak_standard",
        scene="grill",
        description="Pick raw steak from table and place it on the BBQ grill under smoke.",
        difficulty="easy",
        variation="steak",
        seed=0,
        config_overrides={},
        failure_overrides={
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="grill_chicken_dense_smoke",
        scene="grill",
        description="Place chicken on grill with dense BBQ smoke obscuring the workspace.",
        difficulty="medium",
        variation="chicken",
        preset="cinematic_smoke",
        seed=0,
        config_overrides={
            "scene": {"smoke": {"density": 0.80, "blob_count": 18, "blob_radius": 0.065}},
        },
        failure_overrides={
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="grill_steak_far_grill",
        scene="grill",
        description="Place steak on grill positioned farther from the robot (longer transport).",
        difficulty="medium",
        variation="steak",
        seed=0,
        config_overrides={
            "scene": {"grill": {"position": [0.82, 0.14, 0.1]}},
            "task": {"phase_steps": {"move_to_grill": 95}},
        },
        failure_overrides={
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="grill_chicken_offset_start",
        scene="grill",
        description="Chicken placed at an offset starting position; robot must adapt approach.",
        difficulty="medium",
        variation="chicken",
        seed=2,
        config_overrides={
            "scene": {"meat_variations": {"chicken": {"initial_position": [0.40, -0.26, 0.059]}}},
        },
        failure_overrides={
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="grill_steak_narrow_target",
        scene="grill",
        description="Steak must be placed in a smaller target zone on the grill (precise placement).",
        difficulty="hard",
        variation="steak",
        seed=0,
        config_overrides={
            "scene": {"grill": {"target_extent": [0.045, 0.035, 0.020]}},
        },
        failure_overrides={
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
]


# ---------------------------------------------------------------------------
# Liquid-involved tasks
# ---------------------------------------------------------------------------

LIQUID_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="liquid_pick_standard",
        scene="liquid",
        description="Pick an object submerged in a standard-depth liquid container.",
        difficulty="easy",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Lift only 1 cm above surface (< 2 cm object_lifted threshold)
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
    BenchmarkTask(
        task_id="liquid_pick_shallow",
        scene="liquid",
        description="Pick object from a shallow liquid layer (low fill ratio, easier reach).",
        difficulty="easy",
        seed=0,
        config_overrides={
            "scene": {"fluid": {"initial_height": 0.045}},
        },
        failure_overrides={
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
    BenchmarkTask(
        task_id="liquid_pick_deep",
        scene="liquid",
        description="Pick object fully submerged under deep liquid (high fluid resistance).",
        difficulty="medium",
        seed=0,
        config_overrides={
            "scene": {"fluid": {"initial_height": 0.088}},
            "task": {"phase_steps": {"descend": 80, "grasp": 90}},
        },
        failure_overrides={
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
    BenchmarkTask(
        task_id="liquid_pick_offset_seed1",
        scene="liquid",
        description="Object displaced laterally by initial fluid dynamics (seed variation).",
        difficulty="medium",
        seed=1,
        config_overrides={},
        failure_overrides={
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
    BenchmarkTask(
        task_id="liquid_pick_high_lift",
        scene="liquid",
        description="Object must be lifted well above the liquid surface to clear the container rim.",
        difficulty="medium",
        seed=0,
        config_overrides={
            "task": {"lift_height_above_surface": 0.30},
        },
        failure_overrides={
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
    BenchmarkTask(
        task_id="liquid_pick_quick_grasp",
        scene="liquid",
        description="Time-limited grasp window – robot must secure object with fewer grasp steps.",
        difficulty="hard",
        seed=0,
        config_overrides={
            "task": {"phase_steps": {"grasp": 35}},
        },
        failure_overrides={
            "task": {"lift_height_above_surface": 0.01},
        },
    ),
]

# ---------------------------------------------------------------------------
# Steam/spray tasks (new gas scenes — Phase 3.5)
# ---------------------------------------------------------------------------

STEAM_TASKS: List[BenchmarkTask] = [
    BenchmarkTask(
        task_id="steam_pot_place",
        scene="steam",
        task_type="steam_pot_place",
        description="Pick object from table and place it inside a steaming pot.",
        difficulty="easy",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Robot descends 50 cm above object → never within grasp distance
            "task": {"grasp_depth_offset": 0.50},
        },
    ),
    BenchmarkTask(
        task_id="steam_lid_open_close",
        scene="steam",
        task_type="steam_lid_open_close",
        description="Lift pot lid (releasing steam), move aside, then replace it precisely.",
        difficulty="medium",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Barely descend + 1-step grasp → tool never reaches handle → lid not grasped
            "task": {"phase_steps": {"approach": 50, "descend": 1, "grasp_lid": 1, "lift_lid": 50,
                                     "hold_open": 80, "return_to_pot": 55, "lower_lid": 50,
                                     "release": 20, "retract": 40, "hold_done": 120}},
        },
    ),
    BenchmarkTask(
        task_id="spray_chamber_tray",
        scene="steam",
        task_type="spray_chamber_tray",
        description="Insert a tray into a spray-filled processing chamber.",
        difficulty="medium",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Zero grasp threshold → impossible to satisfy dist_xy ≤ 0 → tray never grasped
            "task": {"grasp_threshold": 0.0},
        },
    ),
    BenchmarkTask(
        task_id="steam_valve_close",
        scene="steam",
        task_type="steam_valve_close",
        description="Rotate a leaking steam valve to its closed position.",
        difficulty="hard",
        seed=0,
        config_overrides={},
        failure_overrides={
            # Only 5 rotate steps → valve barely moves, steam keeps flowing
            "task": {"phase_steps": {"approach": 55, "contact": 35, "rotate": 5,
                                     "hold": 30, "retract": 40, "hold_done": 120}},
        },
    ),
]

ALL_TASKS: List[BenchmarkTask] = GRILL_TASKS + LIQUID_TASKS + STEAM_TASKS


def get_task(task_id: str) -> BenchmarkTask:
    for t in ALL_TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id!r}")
