# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import sys
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE_ORDER = ["approach", "descend", "grasp", "lift", "done"]
STATUS_VALUES = {"completed", "aborted", "error"}


def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _round(value: float, ndigits: int = 6) -> float:
    return round(float(value), ndigits)


def _is_nan(value: Any) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _vec3(x: float, y: float, z: float) -> List[float]:
    return [_round(x), _round(y), _round(z)]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[0] + b[0]), _round(a[1] + b[1]), _round(a[2] + b[2])]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[0] - b[0]), _round(a[1] - b[1]), _round(a[2] - b[2])]


def _vec_mul(a: Sequence[float], scalar: float) -> List[float]:
    return [_round(a[0] * scalar), _round(a[1] * scalar), _round(a[2] * scalar)]


def _vec_lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> List[float]:
    return [_round(a[i] + (b[i] - a[i]) * alpha) for i in range(3)]


def _vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def _vec_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return _vec_norm(_vec_sub(a, b))


def _vec_min(points: Sequence[Sequence[float]]) -> List[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    return [_round(min(p[i] for p in points)) for i in range(3)]


def _vec_max(points: Sequence[Sequence[float]]) -> List[float]:
    if not points:
        return [0.0, 0.0, 0.0]
    return [_round(max(p[i] for p in points)) for i in range(3)]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return _round(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    return _round(math.sqrt(sum((v - mu) * (v - mu) for v in values) / len(values)))


def _quat_identity() -> List[float]:
    return [1.0, 0.0, 0.0, 0.0]


def _quat_tool_down() -> List[float]:
    return [0.0, 1.0, 0.0, 0.0]


def _deep_update(base: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _base_config() -> Dict[str, Any]:
    container_inner = {"x": 0.28, "y": 0.24, "z": 0.16}
    liquid_height = 0.065
    container_position = [0.55, 0.0, container_inner["z"] * 0.5]
    surface_z = container_position[2] - container_inner["z"] * 0.5 + liquid_height
    cube_size = 0.042
    cube_position = [container_position[0], container_position[1], surface_z - 0.010]
    return {
        "name": "franka_pick_cube_liquid",
        "preset": "default",
        "backend": "auto",
        "seed": 0,
        "headless": True,
        "video": {"enabled": True, "fps": 30, "min_duration_seconds": 3.0},
        "output": {"root_dir": "outputs/franka_pick_cube_liquid"},
        "scene": {
            "container": {
                "position": container_position,
                "orientation": _quat_identity(),
                "inner_size": container_inner,
                "wall_thickness": 0.008,
            },
            "fluid": {
                "initial_height": liquid_height,
                "occupancy_grid": [18, 18, 12],
                "contact_threshold": 0.04,
                "particle_spacing": 0.0055,
                "surface_z": surface_z,
                "particle_count_goal": [12000, 22000],
                "visual": {
                    "particle_radius_scale": 1.2,
                    "color": [0.05, 0.42, 0.95],
                    "opacity": 1.0,
                    "roughness": 0.03,
                    "metallic": 0.0,
                    "emissive": [0.0, 0.0, 0.0],
                    "enable_anisotropy": True,
                    "enable_smoothing": True,
                    "enable_isosurface": True,
                    "proxy_opacity": 0.06,
                    "proxy_surface_opacity": 0.08,
                    "proxy_min_fill_x": 0.10,
                    "proxy_min_fill_y": 0.10,
                },
                "physx": {
                    "enabled": True,
                    "require_cuda": True,
                    "density": 1000.0,
                    "particle_mass": 0.0,
                    "solver_position_iterations": 32,
                },
            },
            "tool": {
                "radius": 0.018,
                "length": 0.02,
                "orientation": _quat_tool_down(),
                "approach_clearance": 0.08,
                "tip_offset_from_control_frame": [0.0, 0.0, -0.105],
            },
            "pick_object": {
                "size": cube_size,
                "color": [1.0, 0.24, 0.08],
                "initial_position": cube_position,
                "attach_offset": [0.0, 0.0, 0.010],
                "grasp_snap_distance": 0.075,
            },
            "robot": {
                "asset_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "prim_path": "/World/Franka",
                "ee_name": "panda_hand",
                "ik_frame_name": "panda_hand",
            },
        },
        "task": {
            "approach_start_offset": [-0.18, -0.10, 0.16],
            "pregrasp_height": 0.065,
            "grasp_depth_offset": -0.004,
            "lift_height_above_surface": 0.20,
            "lift_x_offset": -0.24,
            "lift_y_offset": 0.08,
            "phase_steps": {
                "approach": 45,
                "descend": 65,
                "grasp": 70,
                "lift": 150,
            },
        },
        "simulation": {
            "physics_dt": 1.0 / 60.0,
            "render_every": 1,
            "particle_damping": 0.96,
            "tool_follow_alpha": 0.24,
            "robot_settle_substeps": 12,
            "robot_reset_settle_steps": 140,
            "tool_influence_radius": 0.03,
            "tool_stir_gain": 0.18,
            "surface_wave_gain": 0.08,
            "gravity_scale": 0.08,
        },
        "logging": {"emit_stdout_log": True},
    }


def get_preset_config(name: str = "default", overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = _base_config()
    if name == "stress":
        config["preset"] = "stress"
        config["scene"]["fluid"]["initial_height"] = 0.075
        config["scene"]["fluid"]["surface_z"] = _round(config["scene"]["container"]["position"][2] - config["scene"]["container"]["inner_size"]["z"] * 0.5 + config["scene"]["fluid"]["initial_height"])
        config["scene"]["fluid"]["occupancy_grid"] = [20, 20, 14]
        config["scene"]["fluid"]["particle_spacing"] = 0.005
        config["scene"]["fluid"]["particle_count_goal"] = [18000, 30000]
        config["scene"]["pick_object"]["initial_position"][2] = _round(config["scene"]["fluid"]["surface_z"] - 0.018)
        config["task"]["pregrasp_height"] = 0.09
        config["task"]["lift_height_above_surface"] = 0.14
        config["simulation"]["tool_stir_gain"] = 2.0
        config["simulation"]["surface_wave_gain"] = 0.6
        config["simulation"]["gravity_scale"] = 0.05
        config["simulation"]["particle_damping"] = 0.978
        config["simulation"]["tool_influence_radius"] = 0.05
    elif name != "default":
        raise ValueError(f"Unknown preset: {name}")
    config = _deep_update(config, overrides)
    config["preset"] = name
    return config


def get_obs_schema() -> Dict[str, Any]:
    return {
        "robot": {
            "joint_positions": {"type": "list[float]", "shape": [9]},
            "joint_velocities": {"type": "list[float]", "shape": [9]},
            "ee_position": {"type": "list[float]", "shape": [3]},
            "ee_orientation": {"type": "list[float]", "shape": [4]},
        },
        "tool": {
            "position": {"type": "list[float]", "shape": [3]},
            "orientation": {"type": "list[float]", "shape": [4]},
            "linear_velocity": {"type": "list[float]", "shape": [3]},
            "angular_velocity": {"type": "list[float]", "shape": [3]},
        },
        "container": {
            "position": {"type": "list[float]", "shape": [3]},
            "orientation": {"type": "list[float]", "shape": [4]},
            "inner_bounds": {
                "min": {"type": "list[float]", "shape": [3], "frame": "container_local"},
                "max": {"type": "list[float]", "shape": [3], "frame": "container_local"},
            },
        },
        "object": {
            "position": {"type": "list[float]", "shape": [3]},
            "size": {"type": "float"},
            "grasped": {"type": "bool"},
        },
        "fluid": {
            "centroid": {"type": "list[float]", "shape": [3]},
            "bbox_min": {"type": "list[float]", "shape": [3]},
            "bbox_max": {"type": "list[float]", "shape": [3]},
            "fill_ratio": {"type": "float"},
            "spill_count": {"type": "int"},
            "tool_min_dist": {"type": "float"},
        },
        "task": {
            "phase": {"type": "str", "enum": PHASE_ORDER},
            "target_position": {"type": "list[float]", "shape": [3]},
            "target_orientation": {"type": "list[float]", "shape": [4]},
        },
    }


def _flatten_numbers(node: Any) -> Iterable[float]:
    if isinstance(node, dict):
        for value in node.values():
            yield from _flatten_numbers(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _flatten_numbers(value)
    elif isinstance(node, (int, float)):
        yield float(node)


def validate_obs(obs: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    required_keys = {"robot", "tool", "container", "object", "fluid", "task"}
    missing = required_keys.difference(obs.keys())
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")
    if len(obs.get("robot", {}).get("joint_positions", [])) != 9:
        errors.append("robot.joint_positions must have length 9")
    if len(obs.get("robot", {}).get("joint_velocities", [])) != 9:
        errors.append("robot.joint_velocities must have length 9")
    for key in ("ee_position",):
        if len(obs.get("robot", {}).get(key, [])) != 3:
            errors.append(f"robot.{key} must have length 3")
    for group in ("tool",):
        for key in ("position", "linear_velocity", "angular_velocity"):
            if len(obs.get(group, {}).get(key, [])) != 3:
                errors.append(f"{group}.{key} must have length 3")
        if len(obs.get(group, {}).get("orientation", [])) != 4:
            errors.append(f"{group}.orientation must have length 4")
    if len(obs.get("container", {}).get("position", [])) != 3:
        errors.append("container.position must have length 3")
    if len(obs.get("container", {}).get("orientation", [])) != 4:
        errors.append("container.orientation must have length 4")
    if len(obs.get("container", {}).get("inner_bounds", {}).get("min", [])) != 3:
        errors.append("container.inner_bounds.min must have length 3")
    if len(obs.get("container", {}).get("inner_bounds", {}).get("max", [])) != 3:
        errors.append("container.inner_bounds.max must have length 3")
    if len(obs.get("object", {}).get("position", [])) != 3:
        errors.append("object.position must have length 3")
    for key in ("centroid", "bbox_min", "bbox_max"):
        if len(obs.get("fluid", {}).get(key, [])) != 3:
            errors.append(f"fluid.{key} must have length 3")
    if obs.get("task", {}).get("phase") not in PHASE_ORDER:
        errors.append("task.phase must be one of the defined task phases")
    if len(obs.get("task", {}).get("target_position", [])) != 3:
        errors.append("task.target_position must have length 3")
    if len(obs.get("task", {}).get("target_orientation", [])) != 4:
        errors.append("task.target_orientation must have length 4")
    for value in _flatten_numbers(obs):
        if math.isnan(value) or math.isinf(value):
            errors.append("observation contains NaN or Inf")
            break
    return len(errors) == 0, errors


def _container_inner_bounds(config: Dict[str, Any]) -> Tuple[List[float], List[float], List[float], List[float]]:
    container = config["scene"]["container"]
    position = list(container["position"])
    inner = container["inner_size"]
    local_min = [-inner["x"] * 0.5, -inner["y"] * 0.5, -inner["z"] * 0.5]
    local_max = [inner["x"] * 0.5, inner["y"] * 0.5, inner["z"] * 0.5]
    world_min = _vec_add(position, local_min)
    world_max = _vec_add(position, local_max)
    return position, local_min, local_max, world_min, world_max


def _world_to_container_local(config: Dict[str, Any], point: Sequence[float]) -> List[float]:
    position, _, _, _, _ = _container_inner_bounds(config)
    return _vec_sub(point, position)


def _container_surface_z(config: Dict[str, Any]) -> float:
    container_position, _, _, _, _ = _container_inner_bounds(config)
    return _round(container_position[2] - config["scene"]["container"]["inner_size"]["z"] * 0.5 + config["scene"]["fluid"]["initial_height"])


def _build_initial_particle_cloud(config: Dict[str, Any], seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    _, _, local_max, world_min, world_max = _container_inner_bounds(config)
    spacing = config["scene"]["fluid"]["particle_spacing"]
    radius = spacing * 0.35
    fluid_height = config["scene"]["fluid"]["initial_height"]
    x = world_min[0] + spacing * 0.5
    points: List[List[float]] = []
    while x <= world_max[0] - spacing * 0.5 + 1e-9:
        y = world_min[1] + spacing * 0.5
        while y <= world_max[1] - spacing * 0.5 + 1e-9:
            z = world_min[2] + spacing * 0.5
            while z <= world_min[2] + fluid_height - spacing * 0.5 + 1e-9:
                jitter = [
                    rng.uniform(-radius, radius),
                    rng.uniform(-radius, radius),
                    rng.uniform(-radius, radius * 0.5),
                ]
                px = min(max(x + jitter[0], world_min[0] + spacing * 0.25), world_max[0] - spacing * 0.25)
                py = min(max(y + jitter[1], world_min[1] + spacing * 0.25), world_max[1] - spacing * 0.25)
                pz = min(max(z + jitter[2], world_min[2] + spacing * 0.25), world_min[2] + fluid_height)
                points.append(_vec3(px, py, pz))
                z += spacing
            y += spacing
        x += spacing
    return points


def compute_fluid_metrics(
    config: Dict[str, Any],
    particle_positions: Sequence[Sequence[float]],
    particle_velocities: Sequence[Sequence[float]],
    tool_position: Sequence[float],
    initial_centroid: Sequence[float],
    tool_path_length_in_fluid: float,
) -> Dict[str, Any]:
    _, local_min, local_max, _, _ = _container_inner_bounds(config)
    occupancy = config["scene"]["fluid"]["occupancy_grid"]
    in_container_positions: List[List[float]] = []
    in_container_velocities: List[List[float]] = []
    spill_count = 0
    occupied_cells = set()
    tool_min_dist = None
    tool_contact_proxy = 0
    threshold = config["scene"]["fluid"]["contact_threshold"]
    dims = [local_max[i] - local_min[i] for i in range(3)]

    tool_radius = float(config["scene"]["tool"].get("radius", 0.018))

    for position, velocity in zip(particle_positions, particle_velocities):
        local = _world_to_container_local(config, position)
        inside = (
            local_min[0] <= local[0] <= local_max[0]
            and local_min[1] <= local[1] <= local_max[1]
            and local_min[2] <= local[2] <= local_max[2]
        )
        if inside:
            in_container_positions.append(list(position))
            in_container_velocities.append(list(velocity))
            indices = []
            for axis in range(3):
                scale = 0.0 if dims[axis] == 0.0 else (local[axis] - local_min[axis]) / dims[axis]
                raw = int(scale * occupancy[axis])
                indices.append(min(max(raw, 0), occupancy[axis] - 1))
            occupied_cells.add(tuple(indices))
        else:
            spill_count += 1

        dist = max(0.0, _vec_distance(position, tool_position) - tool_radius)
        if tool_min_dist is None or dist < tool_min_dist:
            tool_min_dist = dist
        if dist <= threshold:
            tool_contact_proxy += 1

    total_particles = max(len(particle_positions), 1)
    centroid = _vec3(
        _mean([p[0] for p in particle_positions]),
        _mean([p[1] for p in particle_positions]),
        _mean([p[2] for p in particle_positions]),
    )
    height_values = [p[2] for p in in_container_positions]
    speed_values = [_vec_norm(v) for v in in_container_velocities]
    bbox_min = _vec_min(particle_positions)
    bbox_max = _vec_max(particle_positions)
    fill_ratio = len(occupied_cells) / float(max(occupancy[0] * occupancy[1] * occupancy[2], 1))

    return {
        "spill_count": spill_count,
        "spill_ratio": _round(spill_count / total_particles),
        "in_container_particle_count": len(in_container_positions),
        "fill_ratio": _round(fill_ratio),
        "fluid_centroid_displacement": _round(_vec_distance(centroid, initial_centroid)),
        "fluid_height_mean": _mean(height_values),
        "fluid_height_std": _std(height_values),
        "tool_contact_proxy": int(tool_contact_proxy),
        "tool_path_length_in_fluid": _round(tool_path_length_in_fluid),
        "stirring_intensity_proxy": _mean(speed_values),
        "centroid": centroid,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "tool_min_dist": _round(tool_min_dist if tool_min_dist is not None else 0.0),
        "total_particle_count": len(particle_positions),
    }


def _placeholder_video_bytes() -> bytes:
    return b"placeholder video artifact for source-only workspace\n"


def write_placeholder_video(path: Path) -> None:
    path.write_bytes(_placeholder_video_bytes())


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(value)))


def _rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (_clamp_byte(r), _clamp_byte(g), _clamp_byte(b))


def _new_canvas(width: int, height: int, color: Tuple[int, int, int]) -> bytearray:
    return bytearray(color * (width * height))


def _put_pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int]) -> None:
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    index = (y * width + x) * 3
    canvas[index : index + 3] = bytes(color)


def _blend_pixel(canvas: bytearray, width: int, height: int, x: int, y: int, color: Tuple[int, int, int], alpha: float) -> None:
    if x < 0 or x >= width or y < 0 or y >= height:
        return
    index = (y * width + x) * 3
    inv_alpha = 1.0 - alpha
    canvas[index] = _clamp_byte(canvas[index] * inv_alpha + color[0] * alpha)
    canvas[index + 1] = _clamp_byte(canvas[index + 1] * inv_alpha + color[1] * alpha)
    canvas[index + 2] = _clamp_byte(canvas[index + 2] * inv_alpha + color[2] * alpha)


def _fill_rect(
    canvas: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    alpha: float = 1.0,
) -> None:
    left = max(0, min(x0, x1))
    right = min(width, max(x0, x1))
    top = max(0, min(y0, y1))
    bottom = min(height, max(y0, y1))
    for y in range(top, bottom):
        for x in range(left, right):
            if alpha >= 1.0:
                _put_pixel(canvas, width, height, x, y, color)
            else:
                _blend_pixel(canvas, width, height, x, y, color, alpha)


def _draw_line(
    canvas: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    thickness: int = 1,
) -> None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        radius = max(thickness // 2, 0)
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                _put_pixel(canvas, width, height, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _draw_circle(
    canvas: bytearray,
    width: int,
    height: int,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
    fill: bool = True,
) -> None:
    if radius <= 0:
        _put_pixel(canvas, width, height, cx, cy, color)
        return
    r2 = radius * radius
    inner = max(radius - 1, 0) * max(radius - 1, 0)
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            dist2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if fill and dist2 <= r2:
                _put_pixel(canvas, width, height, x, y, color)
            elif not fill and inner <= dist2 <= r2:
                _put_pixel(canvas, width, height, x, y, color)


def _write_binary_ppm(path: Path, width: int, height: int, canvas: bytearray) -> None:
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(canvas))


def _resolve_ffmpeg_binary() -> Optional[str]:
    candidates = [shutil.which("ffmpeg")]
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        candidates.append(str(Path(conda_exe).resolve().with_name("ffmpeg")))
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if Path(candidate).exists():
            return candidate
    return None


def encode_mp4_from_ppm_sequence(frames_dir: Path, output_path: Path, fps: int = 30) -> bool:
    ffmpeg = _resolve_ffmpeg_binary()
    if ffmpeg is not None:
        input_pattern = frames_dir / "frame_%05d.ppm"
        env = os.environ.copy()
        for key in ("LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH", "LD_PRELOAD"):
            env.pop(key, None)
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(input_pattern),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output_path)],
            capture_output=True, text=True, env=env,
        )
        if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
    # Fallback: use OpenCV if available
    try:
        import cv2
        ppm_files = sorted(frames_dir.glob("frame_*.ppm"))
        if not ppm_files:
            return False
        first = cv2.imread(str(ppm_files[0]))
        if first is None:
            return False
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        for p in ppm_files:
            frame = cv2.imread(str(p))
            if frame is not None:
                writer.write(frame)
        writer.release()
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False


def _phase_color(phase: str) -> Tuple[int, int, int]:
    return {
        "approach": _rgb(88, 110, 117),
        "dip": _rgb(38, 139, 210),
        "stir": _rgb(220, 50, 47),
        "lift": _rgb(42, 161, 152),
        "done": _rgb(133, 153, 0),
    }.get(phase, _rgb(120, 120, 120))


def write_episode_video(path: Path, config: Dict[str, Any], trajectory_rows: Sequence[Dict[str, Any]], fps: int = 30) -> bool:
    if not trajectory_rows:
        write_placeholder_video(path)
        return False

    container_position, local_min, local_max, _, _ = _container_inner_bounds(config)
    width = 480
    height = 360
    panel_left = 24
    panel_top = 24
    panel_size = 264
    side_left = 324
    side_width = 120

    def project(world_x: float, world_y: float) -> Tuple[int, int]:
        local_x = world_x - container_position[0]
        local_y = world_y - container_position[1]
        nx = 0.0 if local_max[0] == local_min[0] else (local_x - local_min[0]) / (local_max[0] - local_min[0])
        ny = 0.0 if local_max[1] == local_min[1] else (local_y - local_min[1]) / (local_max[1] - local_min[1])
        px = int(panel_left + max(0.0, min(1.0, nx)) * panel_size)
        py = int(panel_top + panel_size - max(0.0, min(1.0, ny)) * panel_size)
        return px, py

    max_contact = max(float(row.get("contact_proxy", 0.0)) for row in trajectory_rows) or 1.0
    max_stir = max(float(row.get("stirring_intensity_proxy", 0.0)) for row in trajectory_rows) or 1.0
    max_spill = max(float(row.get("spill_count", 0.0)) for row in trajectory_rows) or 1.0
    sampled_rows = list(trajectory_rows[::2])
    if sampled_rows[-1] is not trajectory_rows[-1]:
        sampled_rows.append(trajectory_rows[-1])
    video_cfg = config.get("video", {})
    min_duration_seconds = max(0.0, float(video_cfg.get("min_duration_seconds", 3.0)))
    min_frame_count = max(1, int(round(fps * min_duration_seconds)))
    if len(sampled_rows) < min_frame_count:
        sampled_rows.extend([sampled_rows[-1]] * (min_frame_count - len(sampled_rows)))

    with tempfile.TemporaryDirectory(prefix="franka_stir_video_") as tmpdir:
        frames_dir = Path(tmpdir)
        tool_trail: List[Tuple[int, int]] = []
        centroid_trail: List[Tuple[int, int]] = []

        for frame_index, row in enumerate(sampled_rows):
            canvas = _new_canvas(width, height, _rgb(247, 244, 236))
            phase = str(row.get("phase", "approach"))
            phase_color = _phase_color(phase)
            _fill_rect(canvas, width, height, 0, 0, width, 18, phase_color)
            _fill_rect(canvas, width, height, panel_left - 8, panel_top - 8, panel_left + panel_size + 8, panel_top + panel_size + 8, _rgb(232, 226, 213))
            _fill_rect(canvas, width, height, panel_left, panel_top, panel_left + panel_size, panel_top + panel_size, _rgb(252, 249, 242))

            _draw_line(canvas, width, height, panel_left, panel_top, panel_left + panel_size, panel_top, _rgb(70, 70, 70), 2)
            _draw_line(canvas, width, height, panel_left + panel_size, panel_top, panel_left + panel_size, panel_top + panel_size, _rgb(70, 70, 70), 2)
            _draw_line(canvas, width, height, panel_left + panel_size, panel_top + panel_size, panel_left, panel_top + panel_size, _rgb(70, 70, 70), 2)
            _draw_line(canvas, width, height, panel_left, panel_top + panel_size, panel_left, panel_top, _rgb(70, 70, 70), 2)

            bbox_min_x = float(row.get("bbox_min_x", row.get("centroid_x", container_position[0])))
            bbox_min_y = float(row.get("bbox_min_y", row.get("centroid_y", container_position[1])))
            bbox_max_x = float(row.get("bbox_max_x", row.get("centroid_x", container_position[0])))
            bbox_max_y = float(row.get("bbox_max_y", row.get("centroid_y", container_position[1])))
            fluid_min = project(bbox_min_x, bbox_min_y)
            fluid_max = project(bbox_max_x, bbox_max_y)
            _fill_rect(
                canvas,
                width,
                height,
                fluid_min[0],
                fluid_max[1],
                fluid_max[0],
                fluid_min[1],
                _rgb(133, 188, 224),
                alpha=0.45,
            )

            target = project(float(row["target_x"]), float(row["target_y"]))
            tool = project(float(row["tool_x"]), float(row["tool_y"]))
            centroid = project(float(row["centroid_x"]), float(row["centroid_y"]))
            tool_trail.append(tool)
            centroid_trail.append(centroid)
            tool_trail = tool_trail[-30:]
            centroid_trail = centroid_trail[-45:]

            for trail_index in range(1, len(centroid_trail)):
                _draw_line(
                    canvas,
                    width,
                    height,
                    centroid_trail[trail_index - 1][0],
                    centroid_trail[trail_index - 1][1],
                    centroid_trail[trail_index][0],
                    centroid_trail[trail_index][1],
                    _rgb(88, 170, 214),
                    1,
                )
            for trail_index in range(1, len(tool_trail)):
                _draw_line(
                    canvas,
                    width,
                    height,
                    tool_trail[trail_index - 1][0],
                    tool_trail[trail_index - 1][1],
                    tool_trail[trail_index][0],
                    tool_trail[trail_index][1],
                    _rgb(177, 59, 48),
                    2,
                )

            _draw_circle(canvas, width, height, target[0], target[1], 4, _rgb(230, 196, 73), fill=False)
            _draw_circle(canvas, width, height, tool[0], tool[1], 5, _rgb(200, 64, 52), fill=True)
            _draw_circle(canvas, width, height, centroid[0], centroid[1], 4, _rgb(58, 137, 201), fill=True)

            fill_ratio = max(0.0, min(1.0, float(row.get("fill_ratio", 0.0))))
            contact_ratio = max(0.0, min(1.0, float(row.get("contact_proxy", 0.0)) / max_contact))
            stir_ratio = max(0.0, min(1.0, float(row.get("stirring_intensity_proxy", 0.0)) / max_stir))
            spill_ratio = max(0.0, min(1.0, float(row.get("spill_count", 0.0)) / max_spill)) if max_spill > 0 else 0.0

            bars = [
                (_rgb(74, 144, 226), fill_ratio),
                (_rgb(220, 50, 47), contact_ratio),
                (_rgb(181, 137, 0), stir_ratio),
                (_rgb(147, 161, 161), spill_ratio),
            ]
            for bar_index, (bar_color, ratio) in enumerate(bars):
                top = 42 + bar_index * 64
                _fill_rect(canvas, width, height, side_left, top, side_left + side_width, top + 32, _rgb(226, 220, 209))
                _fill_rect(canvas, width, height, side_left, top, side_left + int(side_width * ratio), top + 32, bar_color)

            progress = frame_index / max(len(sampled_rows) - 1, 1)
            _fill_rect(canvas, width, height, 24, height - 24, width - 24, height - 12, _rgb(220, 214, 202))
            _fill_rect(canvas, width, height, 24, height - 24, 24 + int((width - 48) * progress), height - 12, phase_color)

            _write_binary_ppm(frames_dir / f"frame_{frame_index:05d}.ppm", width, height, canvas)

        if encode_mp4_from_ppm_sequence(frames_dir, path, fps=fps):
            return True

    write_placeholder_video(path)
    return False



def _default_output_dir(config: Dict[str, Any], seed: int) -> Path:
    root = Path(config["output"]["root_dir"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    candidate = root / f"run_{timestamp}_{seed}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"run_{timestamp}_{seed}_{suffix}"
        suffix += 1
    return candidate


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


@dataclass
class EpisodeArtifacts:
    output_dir: Path
    config_path: Path
    obs_schema_path: Path
    metrics_path: Path
    episode_summary_path: Path
    trajectory_path: Path
    events_path: Path
    video_path: Path
    stdout_path: Path


class MockFluidBackend:
    def __init__(self, config: Dict[str, Any]):
        self.config = deepcopy(config)
        self.seed = int(config.get("seed", 0))
        self._rng = random.Random(self.seed)
        self._created = False
        self._step_index = 0
        self._phase = "approach"
        self._phase_events: List[Dict[str, Any]] = []
        self._stdout_lines: List[str] = []
        self._particles: List[List[float]] = []
        self._velocities: List[List[float]] = []
        self._tool_position = [0.0, 0.0, 0.0]
        self._tool_orientation = _quat_tool_down()
        self._tool_linear_velocity = [0.0, 0.0, 0.0]
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._target_position = [0.0, 0.0, 0.0]
        self._target_orientation = _quat_tool_down()
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._initial_centroid = [0.0, 0.0, 0.0]
        self._latest_metrics: Dict[str, Any] = {}
        self._tool_path_length_in_fluid = 0.0
        self._user_tool_action: Optional[Dict[str, Any]] = None
        self._phase_boundaries = self._compute_phase_boundaries()
        self._object_position = [0.0, 0.0, 0.0]
        self._object_pick_position = [0.0, 0.0, 0.0]
        self._object_grasped = False
        self._gripper_closed = False
        self._ee_position = [0.0, 0.0, 0.0]
        self._ee_orientation = _quat_tool_down()

    def _log(self, message: str) -> None:
        self._stdout_lines.append(message)

    def _compute_phase_boundaries(self) -> List[Tuple[str, int]]:
        steps = self.config["task"]["phase_steps"]
        boundary = 0
        result = []
        for phase in PHASE_ORDER:
            if phase == "done":
                continue
            boundary += int(steps[phase])
            result.append((phase, boundary))
        result.append(("done", boundary))
        return result

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self.seed = int(self.config.get("seed", 0))
            self._phase_boundaries = self._compute_phase_boundaries()
        self._created = True
        self._log("mock scene created")
        self.reset_scene(self.seed)

    def _object_initial_position(self) -> List[float]:
        return [float(v) for v in self.config["scene"]["pick_object"]["initial_position"]]

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.seed = int(self.seed if seed is None else seed)
        self._rng = random.Random(self.seed)
        self._particles = _build_initial_particle_cloud(self.config, self.seed)
        self._velocities = [[0.0, 0.0, 0.0] for _ in self._particles]
        self._step_index = 0
        self._phase = "approach"
        self._phase_events = [
            {"step": 0, "event": "reset", "phase": "approach", "seed": self.seed},
            {"step": 0, "event": "phase_enter", "phase": "approach"},
        ]
        self._tool_path_length_in_fluid = 0.0
        self._tool_orientation = list(self.config["scene"]["tool"]["orientation"])
        self._target_orientation = list(self._tool_orientation)
        self._object_position = self._object_initial_position()
        self._object_pick_position = list(self._object_position)
        self._object_grasped = False
        self._apply_gripper_command(False, force=True)
        self._target_position = self._compute_planned_target(0)
        self._tool_position = list(self._target_position)
        self._tool_linear_velocity = [0.0, 0.0, 0.0]
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._ee_position = list(self._tool_position)
        self._ee_orientation = list(self._tool_orientation)
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._initial_centroid = self._compute_centroid()
        self._latest_metrics = compute_fluid_metrics(
            self.config,
            self._particles,
            self._velocities,
            self._tool_position,
            self._initial_centroid,
            self._tool_path_length_in_fluid,
        )
        self._log(f"mock scene reset with seed={self.seed} particles={len(self._particles)}")
        self._update_fluid_visual_from_metrics(self._latest_metrics)
        return self.get_obs()

    def _compute_centroid(self) -> List[float]:
        if not self._particles:
            return [0.0, 0.0, 0.0]
        return _vec3(
            _mean([p[0] for p in self._particles]),
            _mean([p[1] for p in self._particles]),
            _mean([p[2] for p in self._particles]),
        )

    def _determine_phase(self, step_index: int) -> str:
        for phase, boundary in self._phase_boundaries:
            if step_index < boundary:
                return phase
        return "done"

    def _phase_start(self, phase: str) -> int:
        boundary = 0
        for name in PHASE_ORDER:
            if name == "done":
                continue
            if name == phase:
                return boundary
            boundary += int(self.config["task"]["phase_steps"][name])
        return boundary

    def _phase_progress(self, phase: str, step_index: int) -> float:
        if phase == "done":
            return 1.0
        start = self._phase_start(phase)
        duration = max(int(self.config["task"]["phase_steps"][phase]), 1)
        return min(max((step_index - start) / duration, 0.0), 1.0)

    def _attached_object_position(self) -> List[float]:
        return _vec_add(self._tool_position, self.config["scene"]["pick_object"].get("attach_offset", [0.0, 0.0, 0.0]))

    def _compute_planned_target(self, step_index: int) -> List[float]:
        surface_z = _container_surface_z(self.config)
        task = self.config["task"]
        phase = self._determine_phase(step_index)
        object_position = list(self._object_pick_position if self._object_pick_position else self._object_initial_position())
        pregrasp = [object_position[0], object_position[1], object_position[2] + float(task.get("pregrasp_height", 0.08))]
        grasp = [object_position[0], object_position[1], object_position[2] + float(task.get("grasp_depth_offset", 0.002))]
        lift = [
            object_position[0] + float(task.get("lift_x_offset", -0.12)),
            object_position[1] + float(task.get("lift_y_offset", 0.0)),
            surface_z + float(task.get("lift_height_above_surface", 0.12)),
        ]
        start_offset = task.get("approach_start_offset", [-0.16, -0.08, 0.12])
        start = [
            pregrasp[0] + float(start_offset[0]),
            pregrasp[1] + float(start_offset[1]),
            pregrasp[2] + float(start_offset[2]),
        ]
        if phase == "approach":
            return _vec_lerp(start, pregrasp, self._phase_progress("approach", step_index))
        if phase == "descend":
            return _vec_lerp(pregrasp, grasp, self._phase_progress("descend", step_index))
        if phase == "grasp":
            return list(grasp)
        if phase == "lift":
            return _vec_lerp(grasp, lift, self._phase_progress("lift", step_index))
        return list(lift)

    def _update_target_state(self) -> None:
        planned = self._compute_planned_target(self._step_index)
        if self._user_tool_action is not None and "target_position" in self._user_tool_action:
            planned = list(self._user_tool_action["target_position"])
        self._target_position = [float(v) for v in planned]

    def apply_tool_action(self, action: Dict[str, Any]) -> None:
        self._user_tool_action = deepcopy(action)

    def apply_robot_action(self, action: Dict[str, Any]) -> None:
        self._log(f"robot action accepted but unused in v2: {action}")

    def _advance_phase(self) -> None:
        new_phase = self._determine_phase(self._step_index)
        if new_phase != self._phase:
            self._phase = new_phase
            self._phase_events.append({"step": self._step_index, "event": "phase_enter", "phase": self._phase})

    def _apply_gripper_command(self, closed: bool, force: bool = False) -> None:
        if force or closed != self._gripper_closed:
            self._gripper_closed = bool(closed)

    def _maybe_grasp_object(self) -> None:
        if self._object_grasped or not self._gripper_closed:
            return
        object_cfg = self.config["scene"]["pick_object"]
        object_size = float(object_cfg["size"])
        snap_distance = float(object_cfg.get("grasp_snap_distance", max(object_size * 1.2, 0.06)))
        horizontal = math.sqrt((self._tool_position[0] - self._object_position[0]) ** 2 + (self._tool_position[1] - self._object_position[1]) ** 2)
        vertical = abs(self._tool_position[2] - self._object_position[2])
        close_enough = _vec_distance(self._tool_position, self._object_position) <= snap_distance
        aligned_enough = horizontal <= max(object_size * 0.7, 0.022) and vertical <= max(object_size * 1.6, 0.05)
        late_grasp = self._phase == "lift" and _vec_distance(self._target_position, self._object_position) <= max(object_size * 1.6, 0.08)
        if close_enough or aligned_enough or late_grasp:
            self._object_grasped = True
            self._phase_events.append({"step": self._step_index, "event": "object_grasped", "phase": self._phase})
            self._log("pick cube attached to gripper")

    def _update_object_state(self) -> None:
        if self._object_grasped:
            self._object_position = self._attached_object_position()
        elif self._phase in {"grasp", "lift"}:
            self._maybe_grasp_object()
            if self._object_grasped:
                self._object_position = self._attached_object_position()

    def _update_tool_state(self) -> None:
        self._update_target_state()
        previous = list(self._tool_position)
        alpha = float(self.config["simulation"]["tool_follow_alpha"])
        self._tool_position = _vec_lerp(self._tool_position, self._target_position, alpha)
        dt = self.config["simulation"]["physics_dt"]
        delta = _vec_sub(self._tool_position, previous)
        self._tool_linear_velocity = _vec_mul(delta, 1.0 / dt)
        self._tool_angular_velocity = [0.0, 0.0, 0.0]
        self._update_robot_state(delta, dt)

    def _update_robot_state(self, tool_delta: Sequence[float], dt: float) -> None:
        container_position, _, _, _, _ = _container_inner_bounds(self.config)
        relative = _vec_sub(self._tool_position, container_position)
        previous = list(self._robot_joint_positions)
        finger_joint = 0.0 if self._gripper_closed else 0.04
        arm = [
            relative[0] * 5.0,
            relative[1] * 5.0,
            (self._tool_position[2] - 0.25) * 4.0,
            -1.2 + relative[0] * 2.0,
            1.0 + relative[1] * 1.5,
            1.6 - relative[0] * 1.2,
            0.6,
            finger_joint,
            finger_joint,
        ]
        self._robot_joint_positions = [_round(v) for v in arm]
        self._robot_joint_velocities = [_round((a - b) / dt) for a, b in zip(self._robot_joint_positions, previous)]
        self._ee_position = list(self._tool_position)
        self._ee_orientation = list(self._tool_orientation)

    def _update_fluid_visual_from_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        return

    def _simulate_particles(self) -> None:
        _, _, local_max, world_min, world_max = _container_inner_bounds(self.config)
        dt = self.config["simulation"]["physics_dt"]
        influence_radius = self.config["simulation"]["tool_influence_radius"]
        stir_gain = self.config["simulation"]["tool_stir_gain"]
        damping = self.config["simulation"]["particle_damping"]
        gravity = self.config["simulation"]["gravity_scale"]
        surface_z = _container_surface_z(self.config)
        tool_speed = _vec_norm(self._tool_linear_velocity)

        new_positions: List[List[float]] = []
        new_velocities: List[List[float]] = []
        for idx, (position, velocity) in enumerate(zip(self._particles, self._velocities)):
            px, py, pz = position
            vx, vy, vz = velocity
            dx = px - self._tool_position[0]
            dy = py - self._tool_position[1]
            dz = pz - self._tool_position[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            influence = 0.0
            if dist < influence_radius:
                influence = max(0.0, 1.0 - dist / influence_radius)
                vx += self._tool_linear_velocity[0] * stir_gain * influence
                vy += self._tool_linear_velocity[1] * stir_gain * influence
                vz += abs(self._tool_linear_velocity[2]) * 0.16 * influence

            if pz > surface_z - 0.01 and tool_speed > 0.01 and self._phase in {"descend", "grasp", "lift"}:
                wave = math.sin((self._step_index * 0.18) + idx * 0.01)
                vx += wave * self.config["simulation"]["surface_wave_gain"]
                vz += abs(self._tool_linear_velocity[0]) * self.config["simulation"]["surface_wave_gain"] * 0.22

            vz -= 9.81 * gravity * dt
            vx *= damping
            vy *= damping
            vz *= damping

            px += vx * dt
            py += vy * dt
            pz += vz * dt

            if px < world_min[0]:
                px = world_min[0] + (world_min[0] - px) * 0.2
                vx = abs(vx) * 0.2
            if px > world_max[0]:
                px = world_max[0] - (px - world_max[0]) * 0.2
                vx = -abs(vx) * 0.2
            if py < world_min[1]:
                py = world_min[1] + (world_min[1] - py) * 0.2
                vy = abs(vy) * 0.2
            if py > world_max[1]:
                py = world_max[1] - (py - world_max[1]) * 0.2
                vy = -abs(vy) * 0.2

            floor_z = world_min[2]
            if pz < floor_z:
                pz = floor_z + (floor_z - pz) * 0.1
                vz = abs(vz) * 0.1
            if pz > local_max[2] + world_min[2]:
                pz = local_max[2] + world_min[2]
                vz = 0.0

            new_positions.append(_vec3(px, py, pz))
            new_velocities.append(_vec3(vx, vy, vz))

        self._particles = new_positions
        self._velocities = new_velocities

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        for _ in range(num_steps):
            prev_tool = list(self._tool_position)
            self._advance_phase()
            if self._phase == "done":
                break
            self._apply_gripper_command(self._phase in {"grasp", "lift"})
            self._update_tool_state()
            self._update_object_state()
            self._simulate_particles()
            metrics = compute_fluid_metrics(
                self.config,
                self._particles,
                self._velocities,
                self._tool_position,
                self._initial_centroid,
                self._tool_path_length_in_fluid,
            )
            if metrics["tool_contact_proxy"] > 0:
                self._tool_path_length_in_fluid += _vec_distance(prev_tool, self._tool_position)
                metrics["tool_path_length_in_fluid"] = _round(self._tool_path_length_in_fluid)
            metrics["object_height"] = _round(self._object_position[2])
            metrics["object_lifted"] = bool(self._object_position[2] > _container_surface_z(self.config) + 0.02)
            self._latest_metrics = metrics
            self._update_fluid_visual_from_metrics(metrics)
            self._step_index += 1
            self._advance_phase()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        container_position, local_min, local_max, _, _ = _container_inner_bounds(self.config)
        self._latest_metrics = compute_fluid_metrics(
            self.config,
            self._particles,
            self._velocities,
            self._tool_position,
            self._initial_centroid,
            self._tool_path_length_in_fluid,
        )
        self._latest_metrics["object_height"] = _round(self._object_position[2])
        self._latest_metrics["object_lifted"] = bool(self._object_position[2] > _container_surface_z(self.config) + 0.02)
        return {
            "robot": {
                "joint_positions": list(self._robot_joint_positions),
                "joint_velocities": list(self._robot_joint_velocities),
                "ee_position": list(getattr(self, "_ee_position", self._tool_position)),
                "ee_orientation": list(getattr(self, "_ee_orientation", self._tool_orientation)),
            },
            "tool": {
                "position": list(self._tool_position),
                "orientation": list(self._tool_orientation),
                "linear_velocity": list(self._tool_linear_velocity),
                "angular_velocity": list(self._tool_angular_velocity),
            },
            "container": {
                "position": list(container_position),
                "orientation": list(self.config["scene"]["container"]["orientation"]),
                "inner_bounds": {"min": list(local_min), "max": list(local_max)},
            },
            "object": {
                "position": list(self._object_position),
                "size": float(self.config["scene"]["pick_object"]["size"]),
                "grasped": bool(self._object_grasped),
            },
            "fluid": {
                "centroid": list(self._latest_metrics["centroid"]),
                "bbox_min": list(self._latest_metrics["bbox_min"]),
                "bbox_max": list(self._latest_metrics["bbox_max"]),
                "fill_ratio": self._latest_metrics["fill_ratio"],
                "spill_count": self._latest_metrics["spill_count"],
                "tool_min_dist": self._latest_metrics["tool_min_dist"],
            },
            "task": {
                "phase": self._phase,
                "target_position": list(self._target_position),
                "target_orientation": list(self._target_orientation),
            },
        }

    def get_metrics(self) -> Dict[str, Any]:
        return deepcopy(self._latest_metrics)

    def get_events(self) -> List[Dict[str, Any]]:
        return deepcopy(self._phase_events)

    def get_stdout_log(self) -> str:
        return "\n".join(self._stdout_lines) + ("\n" if self._stdout_lines else "")

    def close(self) -> None:
        self._log("mock scene closed")


class IsaacFluidBackend(MockFluidBackend):
    _PARTICLE_SYSTEM_PATH = "/World/FluidParticleSystem"
    _PARTICLE_MATERIAL_PATH = "/World/FluidParticleMaterial"
    _PARTICLE_SET_PATH = "/World/FluidParticles"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._simulation_app = None
        self._runtime_initialized = False
        self._runtime_error: Optional[str] = None
        self._world = None
        self._stage = None
        self._UsdGeom = None
        self._UsdPhysics = None
        self._PhysxSchema = None
        self._Gf = None
        self._Sdf = None
        self._UsdShade = None
        self._UsdLux = None
        self._Usd = None
        self._Articulation = None
        self._SingleParticleSystem = None
        self._ParticleMaterial = None
        self._particleUtils = None
        self._add_reference_to_stage = None
        self._get_assets_root_path = None
        self._Franka = None
        self._FrankaKinematicsSolver = None
        self._np = None
        self._franka = None
        self._ik_solver = None
        self._assets_root_path = None
        self._uses_synthetic_fluid = True
        self._physx_ready = False
        self._physx_failure_reason: Optional[str] = None
        self._particle_system = None
        self._particle_material = None
        self._particle_set_prim = None
        self._visual_particle_material = None

    def _ensure_runtime(self) -> None:
        if self._runtime_initialized:
            return
        from isaacsim import SimulationApp

        headless = bool(self.config.get("headless", True))
        self._simulation_app = SimulationApp(
            {
                "headless": headless,
                "extra_args": ["--/app/useFabricSceneDelegate=0"],
            }
        )
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.materials.particle_material import ParticleMaterial
        from isaacsim.core.prims import Articulation, SingleParticleSystem
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver as FrankaKinematicsSolver
        from isaacsim.storage.native import get_assets_root_path
        from omni.physx.scripts import particleUtils
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

        self._World = World
        self._Articulation = Articulation
        self._SingleParticleSystem = SingleParticleSystem
        self._ParticleMaterial = ParticleMaterial
        self._Franka = Franka
        self._FrankaKinematicsSolver = FrankaKinematicsSolver
        self._np = np
        self._particleUtils = particleUtils
        self._add_reference_to_stage = add_reference_to_stage
        self._get_assets_root_path = get_assets_root_path
        self._UsdGeom = UsdGeom
        self._UsdPhysics = UsdPhysics
        self._PhysxSchema = PhysxSchema
        self._Gf = Gf
        self._Sdf = Sdf
        self._UsdShade = UsdShade
        self._UsdLux = UsdLux
        self._Usd = Usd
        self._runtime_initialized = True
        self._log("Isaac runtime initialized")

    def _set_translate(self, prim_path: str, position: Sequence[float]) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._UsdGeom.XformCommonAPI(prim).SetTranslate(tuple(float(v) for v in position))

    def _set_scale(self, prim_path: str, scale: Sequence[float]) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._UsdGeom.XformCommonAPI(prim).SetScale(tuple(float(v) for v in scale))

    def _define_cube(self, prim_path: str, position: Sequence[float], scale: Sequence[float], color: Sequence[float]) -> None:
        cube = self._UsdGeom.Cube.Define(self._stage, prim_path)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(prim_path, position)
        self._set_scale(prim_path, scale)

    def _define_sphere(self, prim_path: str, position: Sequence[float], radius: float, color: Sequence[float]) -> None:
        sphere = self._UsdGeom.Sphere.Define(self._stage, prim_path)
        sphere.CreateRadiusAttr(float(radius))
        sphere.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(prim_path, position)

    def _define_cylinder(self, prim_path: str, center: Sequence[float], radius: float, height: float, color: Sequence[float]) -> None:
        cylinder = self._UsdGeom.Cylinder.Define(self._stage, prim_path)
        cylinder.CreateRadiusAttr(float(radius))
        cylinder.CreateHeightAttr(float(height))
        cylinder.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(prim_path, center)

    def _create_preview_surface_material(
        self,
        material_path: str,
        color: Sequence[float],
        opacity: float = 1.0,
        roughness: float = 0.2,
        metallic: float = 0.0,
        emissive: Optional[Sequence[float]] = None,
    ):
        material = self._UsdShade.Material.Define(self._stage, material_path)
        shader = self._UsdShade.Shader.Define(self._stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(
            self._Gf.Vec3f(*[float(v) for v in color])
        )
        shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", self._Sdf.ValueTypeNames.Float).Set(float(metallic))
        shader.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(float(opacity))
        if emissive is not None:
            shader.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set(
                self._Gf.Vec3f(*[float(v) for v in emissive])
            )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _bind_material(self, prim_path: str, material) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or material is None:
            return
        self._UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

    def _add_scene_lighting(self, center: Sequence[float]) -> None:
        dome = self._UsdLux.DomeLight.Define(self._stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(1100.0)
        dome.CreateColorAttr(self._Gf.Vec3f(0.96, 0.97, 1.0))

        key = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Key")
        key.CreateIntensityAttr(42000.0)
        key.CreateRadiusAttr(0.12)
        key.CreateColorAttr(self._Gf.Vec3f(1.0, 0.97, 0.93))
        self._set_translate("/World/Lights/Key", [center[0] + 0.18, center[1] - 0.22, center[2] + 0.45])

        fill = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Fill")
        fill.CreateIntensityAttr(14000.0)
        fill.CreateRadiusAttr(0.10)
        fill.CreateColorAttr(self._Gf.Vec3f(0.84, 0.90, 1.0))
        self._set_translate("/World/Lights/Fill", [center[0] - 0.30, center[1] + 0.28, center[2] + 0.26])

        rim = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Rim")
        rim.CreateIntensityAttr(20000.0)
        rim.CreateRadiusAttr(0.06)
        rim.CreateColorAttr(self._Gf.Vec3f(0.94, 0.96, 1.0))
        self._set_translate("/World/Lights/Rim", [center[0] - 0.14, center[1] + 0.36, center[2] + 0.40])

    def _apply_static_collider(self, prim_path: str) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._UsdPhysics.CollisionAPI.Apply(prim)
        self._PhysxSchema.PhysxCollisionAPI.Apply(prim)

    def _apply_kinematic_collider(self, prim_path: str, mass: float = 1.0) -> None:
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return
        self._UsdPhysics.CollisionAPI.Apply(prim)
        rigid_body = self._UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_body.CreateRigidBodyEnabledAttr(True)
        rigid_body.CreateKinematicEnabledAttr(True)
        self._UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        self._PhysxSchema.PhysxCollisionAPI.Apply(prim)

    def _tool_tip_offset_from_control_frame(self) -> List[float]:
        offset = self.config["scene"]["tool"].get("tip_offset_from_control_frame")
        if offset is None:
            return [0.0, 0.0, -float(self.config["scene"]["tool"]["length"])]
        return [_round(float(v)) for v in offset]

    def _control_frame_target_from_tool_tip(self, tool_tip_position: Sequence[float]) -> List[float]:
        offset = self._tool_tip_offset_from_control_frame()
        return [
            _round(tool_tip_position[0] - offset[0]),
            _round(tool_tip_position[1] - offset[1]),
            _round(tool_tip_position[2] - offset[2]),
        ]

    def _tool_tip_to_ee_target(self, tool_tip_position: Sequence[float]) -> List[float]:
        return self._control_frame_target_from_tool_tip(tool_tip_position)

    def _get_robot_frame_world_pose(self, frame_name: Optional[str]) -> Optional[Tuple[List[float], List[float]]]:
        if self._stage is None or self._Usd is None or not frame_name:
            return None
        prim_path = f"{self.config['scene']['robot']['prim_path']}/{frame_name}"
        prim = self._stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            return None
        try:
            world_transform = self._UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(self._Usd.TimeCode.Default())
            translation = world_transform.ExtractTranslation()
            rotation = world_transform.ExtractRotationQuat()
            imag = rotation.GetImaginary()
            position = _vec3(translation[0], translation[1], translation[2])
            orientation = [_round(rotation.GetReal()), _round(imag[0]), _round(imag[1]), _round(imag[2])]
            return position, orientation
        except Exception as exc:
            self._log(f"Robot frame pose warning for {frame_name}: {exc}")
            return None

    def _refresh_robot_state_from_runtime(self, previous_tool_position: Optional[Sequence[float]] = None) -> None:
        if self._franka is None:
            return
        try:
            joint_positions = self._franka.get_joint_positions()
            joint_velocities = self._franka.get_joint_velocities()
            if joint_positions is not None:
                flat_positions = self._np.asarray(joint_positions, dtype=float).reshape(-1)
                self._robot_joint_positions = [_round(v) for v in flat_positions.tolist()]
            if joint_velocities is not None:
                flat_velocities = self._np.asarray(joint_velocities, dtype=float).reshape(-1)
                self._robot_joint_velocities = [_round(v) for v in flat_velocities.tolist()]
            tcp_pose = self._get_robot_frame_world_pose(self.config["scene"]["robot"].get("ik_frame_name"))
            if tcp_pose is None:
                ee_position, ee_orientation = self._franka.end_effector.get_world_pose()
                ee_position = self._np.asarray(ee_position, dtype=float).reshape(-1)
                ee_orientation = self._np.asarray(ee_orientation, dtype=float).reshape(-1)
                current_tool_position = [_round(v) for v in ee_position.tolist()]
                current_tool_orientation = [_round(v) for v in ee_orientation.tolist()]
            else:
                current_tool_position, current_tool_orientation = tcp_pose
            self._ee_position = list(current_tool_position)
            self._ee_orientation = list(current_tool_orientation)
            tool_tip_position = _vec_add(current_tool_position, self._tool_tip_offset_from_control_frame())
            previous = list(self._tool_position) if previous_tool_position is None else list(previous_tool_position)
            dt = float(self.config["simulation"]["physics_dt"])
            delta = _vec_sub(tool_tip_position, previous)
            self._tool_position = list(tool_tip_position)
            self._tool_linear_velocity = _vec_mul(delta, 1.0 / dt)
            self._tool_angular_velocity = [0.0, 0.0, 0.0]
            self._tool_orientation = list(current_tool_orientation)
        except Exception as exc:
            self._log(f"Robot runtime sync warning: {exc}")
            self._franka = None
            self._ik_solver = None

    def _command_robot_to_target(self) -> None:
        if self._franka is None or self._ik_solver is None:
            return
        try:
            target_position = self._np.asarray(self._tool_tip_to_ee_target(self._target_position), dtype=float)
            target_orientation = self._np.asarray(self.config["scene"]["tool"]["orientation"], dtype=float)
            action, success = self._ik_solver.compute_inverse_kinematics(
                target_position,
                target_orientation=target_orientation,
                position_tolerance=0.005,
                orientation_tolerance=0.2,
            )
            if success:
                self._franka.apply_action(action)
            else:
                self._log(f"IK warning: failed to converge to target {self._target_position}")
        except Exception as exc:
            self._log(f"IK command warning: {exc}")
            self._ik_solver = None

    def _settle_robot_to_current_target(self, max_steps: Optional[int] = None) -> None:
        if self._world is None or self._franka is None or self._ik_solver is None:
            return
        settle_steps = int(self.config["simulation"].get("robot_reset_settle_steps", 140) if max_steps is None else max_steps)
        for _ in range(settle_steps):
            previous = list(self._tool_position)
            self._command_robot_to_target()
            self._world.step(render=False)
            self._refresh_robot_state_from_runtime(previous)
            self._sync_runtime_from_state()
            if _vec_distance(self._tool_position, self._target_position) < 0.012:
                break

    def _update_robot_state(self, tool_delta: Sequence[float], dt: float) -> None:
        return

    def _cuda_particle_support_available(self) -> bool:
        physx_cfg = self.config["scene"]["fluid"].get("physx", {})
        if not physx_cfg.get("enabled", True):
            self._physx_failure_reason = "physx particle path disabled by config"
            return False
        if not physx_cfg.get("require_cuda", True):
            return True
        try:
            import torch

            if not torch.cuda.is_available():
                self._physx_failure_reason = "torch.cuda reports no usable CUDA device"
                return False
            return True
        except Exception:
            pass
        nvidia_smi = shutil.which("nvidia-smi")
        if nvidia_smi is None:
            self._physx_failure_reason = "nvidia-smi not available on this machine"
            return False
        try:
            result = subprocess.run([nvidia_smi, "-L"], capture_output=True, text=True, timeout=5)
        except Exception as exc:
            self._physx_failure_reason = f"failed to probe CUDA GPU availability: {exc}"
            return False
        if result.returncode != 0 or "GPU" not in result.stdout:
            self._physx_failure_reason = "no CUDA GPU reported by nvidia-smi"
            return False
        return True

    def _build_physx_particle_data(self) -> Tuple[List[Any], List[Any]]:
        positions = [self._Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in self._particles]
        velocities = [self._Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])) for v in self._velocities]
        return positions, velocities

    def _set_particle_prototype_radius(self, radius: float) -> None:
        prototype_path = f"{self._PARTICLE_SET_PATH}/particlePrototype0"
        prim = self._stage.GetPrimAtPath(prototype_path)
        if prim.IsValid():
            sphere = self._UsdGeom.Sphere(prim)
            sphere.CreateRadiusAttr(float(radius))

    def _set_particle_prototype_visuals(self) -> None:
        visual_cfg = self.config["scene"]["fluid"].get("visual", {})
        prototype_path = f"{self._PARTICLE_SET_PATH}/particlePrototype0"
        prim = self._stage.GetPrimAtPath(prototype_path)
        if not prim.IsValid():
            return
        color = [float(v) for v in visual_cfg.get("color", [0.05, 0.42, 0.95])]
        self._UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([self._Gf.Vec3f(*color)])
        material = self._create_preview_surface_material(
            "/World/Looks/FluidParticles",
            color=color,
            opacity=float(visual_cfg.get("opacity", 1.0)),
            roughness=float(visual_cfg.get("roughness", 0.18)),
            metallic=float(visual_cfg.get("metallic", 0.0)),
            emissive=visual_cfg.get("emissive"),
        )
        self._visual_particle_material = material
        self._bind_material(prototype_path, material)

    def _update_fluid_visual_from_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        if self._stage is None:
            return
        metrics = self._latest_metrics if metrics is None else metrics
        if not metrics:
            return
        visual_cfg = self.config["scene"]["fluid"].get("visual", {})
        _, _, _, world_min, world_max = _container_inner_bounds(self.config)
        inner = self.config["scene"]["container"]["inner_size"]
        centroid = list(metrics.get("centroid", self._latest_metrics.get("centroid", [0.0, 0.0, 0.0])))
        bbox_min = list(metrics.get("bbox_min", world_min))
        bbox_max = list(metrics.get("bbox_max", world_max))
        min_fill_x = float(visual_cfg.get("proxy_min_fill_x", 0.84)) * float(inner["x"])
        min_fill_y = float(visual_cfg.get("proxy_min_fill_y", 0.84)) * float(inner["y"])
        x_size = max(float(bbox_max[0] - bbox_min[0]) + 0.02, min_fill_x)
        y_size = max(float(bbox_max[1] - bbox_min[1]) + 0.02, min_fill_y)
        x_size = min(x_size, float(inner["x"]) - 0.008)
        y_size = min(y_size, float(inner["y"]) - 0.008)
        fluid_height = max(float(bbox_max[2] - world_min[2]), float(self.config["scene"]["fluid"]["initial_height"]) * 0.9, 0.012)
        fluid_height = min(fluid_height, float(inner["z"]) - 0.004)
        x_center = min(max(float(centroid[0]), world_min[0] + x_size * 0.5), world_max[0] - x_size * 0.5)
        y_center = min(max(float(centroid[1]), world_min[1] + y_size * 0.5), world_max[1] - y_size * 0.5)
        volume_center = [x_center, y_center, world_min[2] + fluid_height * 0.5]
        surface_thickness = min(max(fluid_height * 0.08, 0.004), 0.012)
        surface_center = [x_center, y_center, world_min[2] + fluid_height - surface_thickness * 0.5]
        self._set_translate("/World/FluidVisual/Volume", volume_center)
        self._set_scale("/World/FluidVisual/Volume", [x_size, y_size, fluid_height])
        self._set_translate("/World/FluidVisual/Surface", surface_center)
        self._set_scale("/World/FluidVisual/Surface", [x_size * 0.98, y_size * 0.98, surface_thickness])

    def _set_physx_particles_from_state(self) -> None:
        if not self._physx_ready or self._particle_set_prim is None:
            return
        instancer = self._UsdGeom.PointInstancer(self._particle_set_prim)
        positions, velocities = self._build_physx_particle_data()
        instancer.GetPositionsAttr().Set(positions)
        instancer.GetVelocitiesAttr().Set(velocities)

    def _read_physx_particle_state(self) -> bool:
        if not self._physx_ready or self._particle_set_prim is None:
            return False
        instancer = self._UsdGeom.PointInstancer(self._particle_set_prim)
        positions = instancer.GetPositionsAttr().Get() or []
        velocities = instancer.GetVelocitiesAttr().Get() or []
        if not positions:
            return False
        self._particles = [_vec3(p[0], p[1], p[2]) for p in positions]
        if velocities:
            self._velocities = [_vec3(v[0], v[1], v[2]) for v in velocities]
        else:
            self._velocities = [[0.0, 0.0, 0.0] for _ in positions]
        return True

    def _enable_physx_particles(self) -> None:
        self._uses_synthetic_fluid = True
        self._physx_ready = False
        self._particle_set_prim = None
        if not self._cuda_particle_support_available():
            return

        spacing = float(self.config["scene"]["fluid"]["particle_spacing"])
        physx_cfg = self.config["scene"]["fluid"].get("physx", {})
        visual_cfg = self.config["scene"]["fluid"].get("visual", {})
        rest_offset = spacing * 0.5
        contact_offset = rest_offset * 1.5
        system_path = self._Sdf.Path(self._PARTICLE_SYSTEM_PATH)
        particle_group = 0

        try:
            self._particleUtils.add_physx_particle_system(
                self._stage,
                system_path,
                simulation_owner=self._world.get_physics_context().prim_path,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
                particle_contact_offset=contact_offset,
                solid_rest_offset=rest_offset,
                fluid_rest_offset=rest_offset,
                solver_position_iterations=int(physx_cfg.get("solver_position_iterations", 16)),
                max_depenetration_velocity=50.0,
                max_neighborhood=96,
                max_velocity=20.0,
                global_self_collision_enabled=True,
                non_particle_collision_enabled=True,
            )
            self._particle_system = self._SingleParticleSystem(
                prim_path=self._PARTICLE_SYSTEM_PATH,
                simulation_owner=self._world.get_physics_context().prim_path,
                contact_offset=contact_offset,
                rest_offset=rest_offset,
                particle_contact_offset=contact_offset,
                solid_rest_offset=rest_offset,
                fluid_rest_offset=rest_offset,
                solver_position_iteration_count=int(physx_cfg.get("solver_position_iterations", 16)),
                max_depenetration_velocity=50.0,
                max_neighborhood=96,
                max_velocity=20.0,
                global_self_collision_enabled=True,
                non_particle_collision_enabled=True,
            )
            self._particle_material = self._ParticleMaterial(
                prim_path=self._PARTICLE_MATERIAL_PATH,
                friction=0.02,
                damping=0.0,
                viscosity=0.0009,
                cohesion=0.002,
                surface_tension=0.0015,
                gravity_scale=1.0,
                lift=0.0,
                drag=0.0,
            )
            self._particle_system.apply_particle_material(self._particle_material)
            positions, velocities = self._build_physx_particle_data()
            self._particleUtils.add_physx_particleset_pointinstancer(
                self._stage,
                self._Sdf.Path(self._PARTICLE_SET_PATH),
                positions,
                velocities,
                system_path,
                True,
                True,
                particle_group,
                float(physx_cfg.get("particle_mass", 0.0)),
                float(physx_cfg.get("density", 1000.0)),
            )
            self._particle_set_prim = self._stage.GetPrimAtPath(self._PARTICLE_SET_PATH)
            particle_radius = spacing * float(visual_cfg.get("particle_radius_scale", 2.4))
            self._set_particle_prototype_radius(particle_radius)
            self._set_particle_prototype_visuals()
            self._particleUtils.add_physx_particle_anisotropy(
                self._stage,
                system_path,
                enabled=bool(visual_cfg.get("enable_anisotropy", False)),
                scale=1.0,
                min=0.2,
                max=2.0,
            )
            self._particleUtils.add_physx_particle_smoothing(
                self._stage,
                system_path,
                enabled=bool(visual_cfg.get("enable_smoothing", False)),
                strength=0.5,
            )
            self._particleUtils.add_physx_particle_isosurface(
                self._stage,
                system_path,
                enabled=bool(visual_cfg.get("enable_isosurface", False)),
                grid_spacing=spacing * 1.5,
                surface_distance=rest_offset * 1.6,
                max_vertices=1024 * 1024,
                max_triangles=2 * 1024 * 1024,
                max_subgrids=4096,
            )
            self._physx_ready = True
            self._uses_synthetic_fluid = False
            self._physx_failure_reason = None
        except Exception as exc:
            self._physx_failure_reason = f"PhysX particle setup failed: {exc}"
            self._physx_ready = False
            self._uses_synthetic_fluid = True
            self._particle_set_prim = None

    def _configure_gpu_particle_physics(self) -> None:
        if self._world is None:
            return
        try:
            physics_context = self._world.get_physics_context()
            physics_context.enable_gpu_dynamics(True)
            physics_context.set_broadphase_type("GPU")
            physics_context.set_solver_type("TGS")
            self._log("Enabled GPU dynamics for PhysX particle scene")
        except Exception as exc:
            self._log(f"GPU dynamics setup warning: {exc}")

    def _build_runtime_scene(self) -> None:
        self._world = self._World(stage_units_in_meters=1.0)
        self._stage = self._simulation_app.context.get_stage()
        self._configure_gpu_particle_physics()
        self._world.scene.add_default_ground_plane()

        container_position, _, _, _, _ = _container_inner_bounds(self.config)
        inner = self.config["scene"]["container"]["inner_size"]
        wall = float(self.config["scene"]["container"]["wall_thickness"])
        center = [float(v) for v in container_position]
        base_z = center[2] - inner["z"] * 0.5 - wall * 0.5
        self._add_scene_lighting(center)

        table_z = base_z - wall * 0.5 - 0.020
        self._define_cube("/World/Table", [center[0], center[1], table_z - 0.020], [1.10, 0.78, 0.040], [0.62, 0.48, 0.32])
        table_mat = self._create_preview_surface_material("/World/Looks/Table", [0.62, 0.48, 0.32], opacity=1.0, roughness=0.75, metallic=0.0)
        self._bind_material("/World/Table", table_mat)

        self._define_cube(
            "/World/FluidContainer/Base",
            [center[0], center[1], base_z],
            [inner["x"] + 2.0 * wall, inner["y"] + 2.0 * wall, wall],
            [0.84, 0.85, 0.88],
        )
        self._define_cube(
            "/World/FluidContainer/WallPosX",
            [center[0] + inner["x"] * 0.5 + wall * 0.5, center[1], center[2]],
            [wall, inner["y"] + 2.0 * wall, inner["z"] + wall],
            [0.74, 0.76, 0.80],
        )
        self._define_cube(
            "/World/FluidContainer/WallNegX",
            [center[0] - inner["x"] * 0.5 - wall * 0.5, center[1], center[2]],
            [wall, inner["y"] + 2.0 * wall, inner["z"] + wall],
            [0.74, 0.76, 0.80],
        )
        self._define_cube(
            "/World/FluidContainer/WallPosY",
            [center[0], center[1] + inner["y"] * 0.5 + wall * 0.5, center[2]],
            [inner["x"], wall, inner["z"] + wall],
            [0.74, 0.76, 0.80],
        )
        self._define_cube(
            "/World/FluidContainer/WallNegY",
            [center[0], center[1] - inner["y"] * 0.5 - wall * 0.5, center[2]],
            [inner["x"], wall, inner["z"] + wall],
            [0.74, 0.76, 0.80],
        )
        for collider_path in (
            "/World/FluidContainer/Base",
            "/World/FluidContainer/WallPosX",
            "/World/FluidContainer/WallNegX",
            "/World/FluidContainer/WallPosY",
            "/World/FluidContainer/WallNegY",
        ):
            self._apply_static_collider(collider_path)

        glass_mat = self._create_preview_surface_material(
            "/World/Looks/ContainerGlass",
            color=[0.88, 0.92, 0.98],
            opacity=0.16,
            roughness=0.01,
            metallic=0.0,
        )
        base_mat = self._create_preview_surface_material(
            "/World/Looks/ContainerBase",
            color=[0.82, 0.86, 0.90],
            opacity=0.88,
            roughness=0.08,
            metallic=0.0,
        )
        for _wall_path in (
            "/World/FluidContainer/WallPosX",
            "/World/FluidContainer/WallNegX",
            "/World/FluidContainer/WallPosY",
            "/World/FluidContainer/WallNegY",
        ):
            self._bind_material(_wall_path, glass_mat)
        self._bind_material("/World/FluidContainer/Base", base_mat)

        tool_radius = float(self.config["scene"]["tool"]["radius"])
        pick_size = float(self.config["scene"]["pick_object"]["size"])
        pick_color = self.config["scene"]["pick_object"].get("color", [1.0, 0.78, 0.18])
        self._define_sphere("/World/ToolViz/TcpTip", center, tool_radius * 0.36, [1.0, 0.12, 0.12])
        self._define_cylinder("/World/ToolViz/TcpBeacon", [center[0], center[1], center[2] + 0.03], radius=0.0035, height=0.06, color=[1.0, 0.12, 0.12])
        self._define_cube(
            "/World/PickCube",
            self.config["scene"]["pick_object"]["initial_position"],
            [pick_size, pick_size, pick_size],
            pick_color,
        )
        self._apply_kinematic_collider("/World/PickCube")
        self._define_cube(
            "/World/FluidVisual/Volume",
            [center[0], center[1], center[2] - inner["z"] * 0.20],
            [inner["x"] * 0.84, inner["y"] * 0.84, self.config["scene"]["fluid"]["initial_height"]],
            [0.00, 0.46, 1.00],
        )
        self._define_cube(
            "/World/FluidVisual/Surface",
            [center[0], center[1], center[2] - inner["z"] * 0.11],
            [inner["x"] * 0.82, inner["y"] * 0.82, 0.006],
            [0.46, 0.82, 1.00],
        )
        volume_material = self._create_preview_surface_material(
            "/World/Looks/FluidVolume",
            color=[0.00, 0.46, 1.00],
            opacity=float(self.config["scene"]["fluid"]["visual"].get("proxy_opacity", 0.72)),
            roughness=0.08,
            metallic=0.0,
            emissive=[0.02, 0.08, 0.16],
        )
        surface_material = self._create_preview_surface_material(
            "/World/Looks/FluidSurface",
            color=[0.46, 0.82, 1.0],
            opacity=float(self.config["scene"]["fluid"]["visual"].get("proxy_surface_opacity", 0.96)),
            roughness=0.02,
            metallic=0.0,
            emissive=[0.05, 0.14, 0.24],
        )
        self._bind_material("/World/FluidVisual/Volume", volume_material)
        self._bind_material("/World/FluidVisual/Surface", surface_material)
        tcp_material = self._create_preview_surface_material(
            "/World/Looks/TcpMarker",
            color=[1.0, 0.12, 0.12],
            opacity=1.0,
            roughness=0.02,
            metallic=0.0,
            emissive=[0.35, 0.02, 0.02],
        )
        pick_material = self._create_preview_surface_material(
            "/World/Looks/PickCube",
            color=pick_color,
            opacity=1.0,
            roughness=0.22,
            metallic=0.0,
            emissive=[0.28, 0.05, 0.01],
        )
        self._bind_material("/World/ToolViz/TcpTip", tcp_material)
        self._bind_material("/World/ToolViz/TcpBeacon", tcp_material)
        self._bind_material("/World/PickCube", pick_material)

        try:
            self._assets_root_path = self._get_assets_root_path()
            if self._assets_root_path:
                robot_usd = self._assets_root_path + self.config["scene"]["robot"]["asset_path"]
                self._franka = self._Franka(
                    prim_path=self.config["scene"]["robot"]["prim_path"],
                    name="franka_stir_liquid_robot",
                    usd_path=robot_usd,
                    end_effector_prim_name=self.config["scene"]["robot"].get("ee_name"),
                )
                self._world.scene.add(self._franka)
                self._log(f"Loaded Franka asset from {robot_usd}")
            else:
                self._log("Assets root unavailable; robot asset load skipped")
        except Exception as exc:
            self._franka = None
            self._log(f"Robot asset load skipped: {exc}")

        self._enable_physx_particles()
        self._world.reset()
        if self._franka is not None:
            try:
                self._ik_solver = self._FrankaKinematicsSolver(
                    self._franka,
                    end_effector_frame_name=self.config["scene"]["robot"].get("ik_frame_name"),
                )
                self._refresh_robot_state_from_runtime()
            except Exception as exc:
                self._ik_solver = None
                self._log(f"Franka IK setup skipped: {exc}")
        if self._uses_synthetic_fluid:
            reason = self._physx_failure_reason or "no usable PhysX particle runtime available"
            self._log(f"Isaac backend uses real scene skeleton with synthetic fluid state because {reason}")
        else:
            self._log("Isaac backend enabled PhysX PBD particle fluid")

    def _apply_gripper_command(self, closed: bool, force: bool = False) -> None:
        if self._franka is not None and getattr(self._franka, "gripper", None) is not None and (force or closed != self._gripper_closed):
            try:
                if closed:
                    self._franka.gripper.close()
                else:
                    self._franka.gripper.open()
            except Exception as exc:
                self._log(f"Gripper command warning: {exc}")
        super()._apply_gripper_command(closed, force=force)

    def _sync_runtime_from_state(self) -> None:
        if self._stage is None:
            return
        self._set_translate("/World/ToolViz/TcpTip", self._tool_position)
        self._set_translate("/World/ToolViz/TcpBeacon", [self._tool_position[0], self._tool_position[1], self._tool_position[2] + 0.03])
        self._set_translate("/World/PickCube", self._object_position)
        self._update_fluid_visual_from_metrics(self._latest_metrics)

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self.seed = int(self.config.get("seed", 0))
            self._phase_boundaries = self._compute_phase_boundaries()
        if not importlib.util.find_spec("isaacsim"):
            raise RuntimeError(
                "The Isaac backend requires an Isaac Sim Python runtime. Use --backend mock in source-only workspaces."
            )
        self._ensure_runtime()
        self._build_runtime_scene()
        self._created = True
        self.reset_scene(self.seed)

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        MockFluidBackend.reset_scene(self, seed)
        if self._world is not None:
            try:
                self._world.reset()
            except Exception as exc:
                self._log(f"World reset warning: {exc}")
        if not self._uses_synthetic_fluid:
            try:
                self._set_physx_particles_from_state()
            except Exception as exc:
                self._physx_failure_reason = f"PhysX particle reset failed: {exc}"
                self._physx_ready = False
                self._uses_synthetic_fluid = True
                self._log(f"Falling back to synthetic fluid after reset failure: {exc}")
        self._phase_events.append(
            {
                "step": 0,
                "event": "runtime_mode",
                "backend": "isaac",
                "fluid_mode": "synthetic_fluid" if self._uses_synthetic_fluid else "physx_particles",
            }
        )
        self._apply_gripper_command(False, force=True)
        self._settle_robot_to_current_target()
        self._sync_runtime_from_state()
        if self._world is not None:
            try:
                self._world.step(render=False)
                self._refresh_robot_state_from_runtime()
                self._sync_runtime_from_state()
                if not self._uses_synthetic_fluid:
                    self._read_physx_particle_state()
            except Exception as exc:
                self._log(f"World step warning after reset: {exc}")
        return self.get_obs()

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        for _ in range(num_steps):
            prev_tool = list(self._tool_position)
            self._advance_phase()
            if self._phase == "done":
                break
            self._apply_gripper_command(self._phase in {"grasp", "lift"})
            self._update_target_state()
            if self._world is not None:
                try:
                    settle_substeps = int(self.config["simulation"].get("robot_settle_substeps", 4))
                    for substep in range(settle_substeps):
                        prior = list(self._tool_position) if substep == 0 else prev_tool
                        self._command_robot_to_target()
                        self._world.step(render=False)
                        self._refresh_robot_state_from_runtime(prior)
                        self._update_object_state()
                        self._sync_runtime_from_state()
                        if _vec_distance(self._tool_position, self._target_position) < 0.012:
                            break
                    if not self._uses_synthetic_fluid and not self._read_physx_particle_state():
                        self._log("PhysX particle readback returned no particles; retaining previous state")
                except Exception as exc:
                    self._log(f"World step warning: {exc}")
                    if not self._uses_synthetic_fluid:
                        self._physx_failure_reason = f"PhysX particle step failed: {exc}"
                        self._physx_ready = False
                        self._uses_synthetic_fluid = True
                        self._log(f"Falling back to synthetic fluid after runtime step failure: {exc}")
            if self._uses_synthetic_fluid:
                self._simulate_particles()
            self._update_object_state()
            self._sync_runtime_from_state()
            metrics = compute_fluid_metrics(
                self.config,
                self._particles,
                self._velocities,
                self._tool_position,
                self._initial_centroid,
                self._tool_path_length_in_fluid,
            )
            if metrics["tool_contact_proxy"] > 0:
                self._tool_path_length_in_fluid += _vec_distance(prev_tool, self._tool_position)
                metrics["tool_path_length_in_fluid"] = _round(self._tool_path_length_in_fluid)
            metrics["object_height"] = _round(self._object_position[2])
            metrics["object_lifted"] = bool(self._object_position[2] > _container_surface_z(self.config) + 0.02)
            self._latest_metrics = metrics
            self._step_index += 1
            self._advance_phase()
        return self.get_obs()

    def close(self) -> None:
        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
            self._simulation_app = None
        super().close()


class FrankaStirLiquidEnv:
    def __init__(self, config: Optional[Dict[str, Any]] = None, preset: str = "default", backend: str = "auto"):
        self.config = get_preset_config(preset, config)
        self.config["backend"] = backend
        self.backend_name = self._resolve_backend_name(backend)
        if self.backend_name == "isaac":
            self._backend: Any = IsaacFluidBackend(self.config)
        else:
            self._backend = MockFluidBackend(self.config)

    def _resolve_backend_name(self, backend: str) -> str:
        if backend == "auto":
            return "isaac" if importlib.util.find_spec("isaacsim") else "mock"
        if backend not in {"isaac", "mock"}:
            raise ValueError(f"Unsupported backend: {backend}")
        return backend

    def create_scene(self, config: Optional[Dict[str, Any]] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
        self._backend.create_scene(config or self.config)

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        return self._backend.reset_scene(seed)

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        return self._backend.step_sim(num_steps)

    def apply_tool_action(self, action: Dict[str, Any]) -> None:
        self._backend.apply_tool_action(action)

    def apply_robot_action(self, action: Dict[str, Any]) -> None:
        self._backend.apply_robot_action(action)

    def get_obs(self) -> Dict[str, Any]:
        return self._backend.get_obs()

    def get_metrics(self) -> Dict[str, Any]:
        return self._backend.get_metrics()

    def get_events(self) -> List[Dict[str, Any]]:
        return self._backend.get_events()

    def close(self) -> None:
        self._backend.close()

    def run_episode(
        self,
        seed: Optional[int] = None,
        output_dir: Optional[Path] = None,
        max_steps: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], EpisodeArtifacts]:
        run_seed = int(self.config.get("seed", 0) if seed is None else seed)
        self.config["seed"] = run_seed
        if not getattr(self._backend, "_created", False):
            self.create_scene(self.config)
        self.reset_scene(run_seed)
        output_dir = Path(output_dir) if output_dir is not None else _default_output_dir(self.config, run_seed)
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts = EpisodeArtifacts(
            output_dir=output_dir,
            config_path=output_dir / "config.json",
            obs_schema_path=output_dir / "obs_schema.json",
            metrics_path=output_dir / "metrics.json",
            episode_summary_path=output_dir / "episode_summary.json",
            trajectory_path=output_dir / "trajectory.csv",
            events_path=output_dir / "events.jsonl",
            video_path=output_dir / "video.mp4",
            stdout_path=output_dir / "stdout.log",
        )
        _write_json(artifacts.config_path, self.config)
        _write_json(artifacts.obs_schema_path, get_obs_schema())

        total_budget = max_steps
        if total_budget is None:
            total_budget = sum(self.config["task"]["phase_steps"].values()) + 5

        trajectory_rows: List[Dict[str, Any]] = []
        status = "completed"
        error_text = None
        try:
            for _ in range(total_budget):
                obs = self.get_obs()
                valid, errors = validate_obs(obs)
                if not valid:
                    raise RuntimeError("Invalid observation: " + "; ".join(errors))
                metrics = self.get_metrics()
                trajectory_rows.append(
                    {
                        "step": getattr(self._backend, "_step_index", 0),
                        "phase": obs["task"]["phase"],
                        "target_x": obs["task"]["target_position"][0],
                        "target_y": obs["task"]["target_position"][1],
                        "target_z": obs["task"]["target_position"][2],
                        "tool_x": obs["tool"]["position"][0],
                        "tool_y": obs["tool"]["position"][1],
                        "tool_z": obs["tool"]["position"][2],
                        "centroid_x": metrics["centroid"][0],
                        "centroid_y": metrics["centroid"][1],
                        "centroid_z": metrics["centroid"][2],
                        "spill_count": metrics["spill_count"],
                        "fill_ratio": metrics["fill_ratio"],
                        "contact_proxy": metrics["tool_contact_proxy"],
                        "stirring_intensity_proxy": metrics["stirring_intensity_proxy"],
                        "bbox_min_x": metrics["bbox_min"][0],
                        "bbox_min_y": metrics["bbox_min"][1],
                        "bbox_max_x": metrics["bbox_max"][0],
                        "bbox_max_y": metrics["bbox_max"][1],
                    }
                )
                if obs["task"]["phase"] == "done":
                    break
                self.step_sim(1)
            else:
                status = "aborted"
        except Exception as exc:
            status = "error"
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        metrics = self.get_metrics()
        phase_sequence = [event["phase"] for event in self.get_events() if event["event"] == "phase_enter"]
        final_summary = {
            "status": status,
            "backend": self.backend_name,
            "preset": self.config["preset"],
            "seed": run_seed,
            "step_count": getattr(self._backend, "_step_index", 0),
            "phase_sequence": phase_sequence,
            "fluid_mode": "synthetic_fluid" if getattr(self._backend, "_uses_synthetic_fluid", False) else "physx_particles",
            "error": error_text,
            "metrics": metrics,
        }

        with artifacts.trajectory_path.open("w", encoding="utf-8", newline="") as stream:
            fieldnames = list(trajectory_rows[0].keys()) if trajectory_rows else ["step", "phase"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for row in trajectory_rows:
                writer.writerow(row)

        _append_jsonl(artifacts.events_path, self.get_events())
        _write_json(
            artifacts.metrics_path,
            {
                "backend": self.backend_name,
                "preset": self.config["preset"],
                "seed": run_seed,
                "final": metrics,
                "phase_sequence": phase_sequence,
            },
        )
        _write_json(artifacts.episode_summary_path, final_summary)
        if self.config.get("video", {}).get("enabled", True):
            write_episode_video(artifacts.video_path, self.config, trajectory_rows)
        else:
            write_placeholder_video(artifacts.video_path)
        artifacts.stdout_path.write_text(self._backend.get_stdout_log(), encoding="utf-8")
        return final_summary, artifacts


def run_episode_cli(args: argparse.Namespace) -> int:
    config = get_preset_config(args.preset)
    config["seed"] = args.seed
    config["headless"] = args.headless
    config["video"]["enabled"] = not args.disable_video
    if args.output_root is not None:
        config["output"]["root_dir"] = args.output_root
    env = FrankaStirLiquidEnv(config=config, preset=args.preset, backend=args.backend)
    try:
        summary, artifacts = env.run_episode(seed=args.seed, output_dir=None, max_steps=args.max_steps)
        print(json.dumps({"summary": summary, "output_dir": str(artifacts.output_dir)}, indent=2, sort_keys=True))
        return 0 if summary["status"] == "completed" else 1
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimum viable Franka stirring liquid scene.")
    parser.add_argument("--preset", choices=["default", "stress"], default="default")
    parser.add_argument("--backend", choices=["auto", "isaac", "mock"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_episode_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
