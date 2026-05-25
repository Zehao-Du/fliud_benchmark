"""Benchmark task definitions for fluid-involved robot manipulation.

Each BenchmarkTask specifies a scene, config overrides, and expected success criteria.
Two groups of 6 tasks each cover gas-involved (grill) and liquid-involved manipulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class BenchmarkTask:
    task_id: str
    scene: str                  # "liquid" | "grill"
    description: str
    difficulty: str             # "easy" | "medium" | "hard"
    preset: str = "default"
    variation: str = "chicken"  # grill only
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

ALL_TASKS: List[BenchmarkTask] = GRILL_TASKS + LIQUID_TASKS


def get_task(task_id: str) -> BenchmarkTask:
    for t in ALL_TASKS:
        if t.task_id == task_id:
            return t
    raise KeyError(f"Unknown task_id: {task_id!r}")
