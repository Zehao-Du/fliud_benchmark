# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four steam/spray manipulation tasks for FluidBench.

Tasks
-----
  steam_pot_place      — pick object from tray, place inside steaming pot
  steam_lid_open_close — lift pot lid (steam vents), then replace lid
  spray_chamber_tray   — insert tray into mist-filled spray chamber
  steam_valve_close    — rotate valve knob to stop a steam leak

Each task shares the dual-backend pattern:
  MockSteamTaskBackend  — pure Python, deterministic, no Isaac required
  IsaacSteamTaskBackend — Isaac Sim physics + RTX rendering + particle cloud steam

Usage
-----
  python franka_steam_tasks.py --task steam_pot_place --backend mock
  python franka_steam_tasks.py --task steam_valve_close --backend mock --seed 1
  python franka_steam_tasks.py --task steam_lid_open_close --backend isaac --headless
"""
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
import sys
import tempfile
import traceback
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_TYPES = [
    "steam_pot_place",
    "steam_lid_open_close",
    "spray_chamber_tray",
    "steam_valve_close",
]

PHASE_ORDERS: Dict[str, List[str]] = {
    "steam_pot_place": [
        "approach", "descend", "grasp", "lift",
        "move_to_pot", "lower_into_pot", "release", "retract", "hold_done",
    ],
    "steam_lid_open_close": [
        "approach", "descend", "grasp_lid", "lift_lid",
        "hold_open", "return_to_pot", "lower_lid", "release", "retract", "hold_done",
    ],
    "spray_chamber_tray": [
        "approach", "descend", "grasp_tray", "lift",
        "align_to_chamber", "insert", "release", "retract", "hold_done",
    ],
    "steam_valve_close": [
        "approach", "contact", "rotate", "hold", "retract", "hold_done",
    ],
}

STATUS_VALUES = {"completed", "aborted", "error"}


# ---------------------------------------------------------------------------
# Utility functions (geometry, canvas, ffmpeg)
# ---------------------------------------------------------------------------

def _now_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _round(v: float, n: int = 6) -> float:
    return round(float(v), n)


def _vec3(x: float, y: float, z: float) -> List[float]:
    return [_round(x), _round(y), _round(z)]


def _vec_add(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[i] + b[i]) for i in range(3)]


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> List[float]:
    return [_round(a[i] - b[i]) for i in range(3)]


def _vec_mul(a: Sequence[float], s: float) -> List[float]:
    return [_round(v * s) for v in a]


def _vec_lerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    return [_round(a[i] + (b[i] - a[i]) * t) for i in range(3)]


def _vec_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _deep_update(base: Dict, patch: Optional[Dict]) -> Dict:
    result = deepcopy(base)
    for k, v in (patch or {}).items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_update(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def _quat_tool_down() -> List[float]:
    return [0.0, 1.0, 0.0, 0.0]


def _clamp_byte(v: float) -> int:
    return max(0, min(255, int(v)))


def _rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (_clamp_byte(r), _clamp_byte(g), _clamp_byte(b))


def _new_canvas(w: int, h: int, color: Tuple) -> bytearray:
    return bytearray(color * (w * h))


def _put_pixel(canvas: bytearray, w: int, h: int, x: int, y: int, color: Tuple) -> None:
    if 0 <= x < w and 0 <= y < h:
        i = (y * w + x) * 3
        canvas[i:i + 3] = bytes(color)


def _blend_pixel(canvas: bytearray, w: int, h: int, x: int, y: int, color: Tuple, alpha: float) -> None:
    if 0 <= x < w and 0 <= y < h:
        i = (y * w + x) * 3
        a = 1.0 - alpha
        canvas[i] = _clamp_byte(canvas[i] * a + color[0] * alpha)
        canvas[i + 1] = _clamp_byte(canvas[i + 1] * a + color[1] * alpha)
        canvas[i + 2] = _clamp_byte(canvas[i + 2] * a + color[2] * alpha)


def _fill_rect(canvas: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int,
               color: Tuple, alpha: float = 1.0) -> None:
    for py in range(max(0, min(y0, y1)), min(h, max(y0, y1))):
        for px in range(max(0, min(x0, x1)), min(w, max(x0, x1))):
            if alpha >= 1.0:
                _put_pixel(canvas, w, h, px, py, color)
            else:
                _blend_pixel(canvas, w, h, px, py, color, alpha)


def _draw_line(canvas: bytearray, w: int, h: int, x0: int, y0: int, x1: int, y1: int,
               color: Tuple, thickness: int = 1) -> None:
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    r = max(0, thickness // 2)
    while True:
        for oy in range(-r, r + 1):
            for ox in range(-r, r + 1):
                _put_pixel(canvas, w, h, x0 + ox, y0 + oy, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy; x0 += sx
        if e2 < dx:
            err += dx; y0 += sy


def _draw_circle(canvas: bytearray, w: int, h: int, cx: int, cy: int, radius: int,
                 color: Tuple, alpha: float = 1.0) -> None:
    r2 = max(1, radius) ** 2
    for py in range(cy - radius, cy + radius + 1):
        for px in range(cx - radius, cx + radius + 1):
            if (px - cx) ** 2 + (py - cy) ** 2 <= r2:
                if alpha >= 1.0:
                    _put_pixel(canvas, w, h, px, py, color)
                else:
                    _blend_pixel(canvas, w, h, px, py, color, alpha)


def _write_ppm(path: Path, w: int, h: int, canvas: bytearray) -> None:
    path.write_bytes(f"P6\n{w} {h}\n255\n".encode() + bytes(canvas))


def _resolve_ffmpeg() -> Optional[str]:
    candidates = [shutil.which("ffmpeg")]
    conda = os.environ.get("CONDA_EXE")
    if conda:
        candidates.append(str(Path(conda).with_name("ffmpeg")))
    # imageio-ffmpeg binary
    try:
        import imageio.plugins.ffmpeg as _ff
        candidates.append(_ff.get_exe())
    except Exception:
        pass
    return next((c for c in candidates if c and Path(c).exists()), None)


def _encode_mp4(frames_dir: Path, out: Path, fps: int = 30) -> bool:
    ffmpeg = _resolve_ffmpeg()
    if ffmpeg:
        r = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", str(frames_dir / "frame_%05d.ppm"), "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
            capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k not in {"PYTHONHOME", "PYTHONPATH", "LD_PRELOAD"}},
        )
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return True
    try:
        import cv2
        files = sorted(frames_dir.glob("frame_*.ppm"))
        if not files:
            return False
        first = cv2.imread(str(files[0]))
        if first is None:
            return False
        wrt = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
        for f in files:
            fr = cv2.imread(str(f))
            if fr is not None:
                wrt.write(fr)
        wrt.release()
        return out.exists() and out.stat().st_size > 0
    except Exception:
        return False


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_TABLE = {
    "position": [0.58, 0.0, 0.025],
    "size": {"x": 0.92, "y": 0.72, "z": 0.05},
    "color": [0.74, 0.69, 0.61],
}
_TABLE_TOP_Z = _TABLE["position"][2] + _TABLE["size"]["z"] * 0.5  # 0.05

_SIM = {
    "physics_dt": 1.0 / 60.0,
    "tool_follow_alpha": 0.22,
    "gripper_width_open": 0.040,
    "gripper_width_closed": 0.002,
    "robot_reset_settle_steps": 140,
    "robot_settle_substeps": 10,
}

_POT = {
    "position": [0.70, 0.10, 0.11],   # pot center; pot floor_z = 0.05
    "inner_radius": 0.072,
    "outer_radius": 0.082,
    "height": 0.12,
    "wall_thickness": 0.010,
    "color": [0.28, 0.28, 0.30],
    "rim_color": [0.45, 0.45, 0.48],
}
_POT_TOP_Z = _POT["position"][2] + _POT["height"] * 0.5   # 0.17
_POT_FLOOR_Z = _POT["position"][2] - _POT["height"] * 0.5  # 0.05


def _base_config(task_type: str) -> Dict[str, Any]:
    if task_type == "steam_pot_place":
        obj_half_z = 0.009
        obj_z = _TABLE_TOP_Z + obj_half_z  # 0.059
        return {
            "name": "franka_steam_tasks",
            "task_type": task_type,
            "preset": "default",
            "backend": "auto",
            "seed": 0,
            "headless": True,
            "video": {"enabled": True, "fps": 30, "min_duration_seconds": 3.0},
            "output": {"root_dir": "outputs/franka_steam_tasks/steam_pot_place"},
            "scene": {
                "table": deepcopy(_TABLE),
                "pot": deepcopy(_POT),
                "object": {
                    "position": [0.42, -0.22, obj_z],
                    "size": [0.040, 0.040, 0.018],
                    "color": [0.72, 0.52, 0.34],
                },
                "steam": {
                    "origin": [_POT["position"][0], _POT["position"][1], _POT_TOP_Z + 0.030],
                    "bounds": [0.20, 0.20, 0.50],
                    "density": 0.55,
                    "rise_speed": 0.014,
                    "lateral_jitter": 0.008,
                    "swirl_strength": 0.006,
                    "blob_count": 10,
                    "blob_radius": 0.042,
                    "always_active": True,
                },
            },
            "task": {
                "grasp_depth_offset": 0.008,
                "lift_z": 0.28,
                "pot_approach_z": 0.28,
                "insert_z": 0.093,  # EE z while object is in pot
                "release_progress": 0.50,
                "phase_steps": {
                    "approach": 50, "descend": 45, "grasp": 25, "lift": 45,
                    "move_to_pot": 65, "lower_into_pot": 50,
                    "release": 20, "retract": 40, "hold_done": 120,
                },
            },
            "simulation": deepcopy(_SIM),
        }

    elif task_type == "steam_lid_open_close":
        lid_z = _POT_TOP_Z + 0.005       # 0.175  (lid center on pot rim)
        handle_tip_z = lid_z + 0.010 + 0.040 + 0.006  # lid_top + handle_h + grasp clearance = 0.231
        aside = [0.52, -0.18, 0.42]
        return {
            "name": "franka_steam_tasks",
            "task_type": task_type,
            "preset": "default",
            "backend": "auto",
            "seed": 0,
            "headless": True,
            "video": {"enabled": True, "fps": 30, "min_duration_seconds": 4.0},
            "output": {"root_dir": "outputs/franka_steam_tasks/steam_lid_open_close"},
            "scene": {
                "table": deepcopy(_TABLE),
                "pot": deepcopy(_POT),
                "lid": {
                    "position": [_POT["position"][0], _POT["position"][1], lid_z],
                    "radius": 0.079,
                    "height": 0.010,
                    "handle_radius": 0.018,
                    "handle_height": 0.040,
                    "color": [0.52, 0.52, 0.54],
                    "handle_color": [0.35, 0.35, 0.38],
                },
                "steam": {
                    "origin": [_POT["position"][0], _POT["position"][1], _POT_TOP_Z + 0.025],
                    "bounds": [0.22, 0.22, 0.55],
                    "density": 0.70,
                    "rise_speed": 0.016,
                    "lateral_jitter": 0.010,
                    "swirl_strength": 0.008,
                    "blob_count": 12,
                    "blob_radius": 0.050,
                    "always_active": False,  # triggered when lid is above threshold
                },
                "steam_trigger_z": _POT_TOP_Z + 0.050,  # lid must be above this to activate
            },
            "task": {
                "grasp_depth_offset": 0.005,
                "lift_z": 0.42,
                "aside_position": aside,
                "replace_tolerance": 0.030,
                "release_progress": 0.50,
                "phase_steps": {
                    "approach": 50, "descend": 45, "grasp_lid": 25, "lift_lid": 50,
                    "hold_open": 80, "return_to_pot": 55, "lower_lid": 50,
                    "release": 20, "retract": 40, "hold_done": 120,
                },
                "_handle_grip_z": handle_tip_z,  # internal precomputed value
            },
            "simulation": deepcopy(_SIM),
        }

    elif task_type == "spray_chamber_tray":
        tray_half_z = 0.019
        tray_z = _TABLE_TOP_Z + tray_half_z  # 0.069
        tray_top_z = tray_z + tray_half_z    # 0.088
        chamber_pos = [0.76, 0.0, 0.145]
        chamber_size = {"x": 0.20, "y": 0.28, "z": 0.12}
        chamber_floor_z = chamber_pos[2] - chamber_size["z"] * 0.5  # 0.085
        tray_in_chamber_z = chamber_floor_z + tray_half_z           # 0.104
        ee_in_chamber_z = tray_in_chamber_z + tray_half_z + 0.006   # 0.123 approx

        return {
            "name": "franka_steam_tasks",
            "task_type": task_type,
            "preset": "default",
            "backend": "auto",
            "seed": 0,
            "headless": True,
            "video": {"enabled": True, "fps": 30, "min_duration_seconds": 3.0},
            "output": {"root_dir": "outputs/franka_steam_tasks/spray_chamber_tray"},
            "scene": {
                "table": deepcopy(_TABLE),
                "tray": {
                    "position": [0.40, 0.0, tray_z],
                    "size": [0.160, 0.220, 0.038],
                    "color": [0.68, 0.72, 0.78],
                },
                "chamber": {
                    "position": chamber_pos,
                    "size": chamber_size,
                    "opening_x": chamber_pos[0] - chamber_size["x"] * 0.5,  # 0.66
                    "tray_rest_z": tray_in_chamber_z,
                    "ee_rest_z": ee_in_chamber_z,
                    "color": [0.35, 0.38, 0.40],
                    "interior_color": [0.22, 0.24, 0.26],
                },
                "steam": {
                    "origin": chamber_pos,
                    "bounds": [0.12, 0.16, 0.08],
                    "density": 0.45,
                    "rise_speed": 0.008,
                    "lateral_jitter": 0.012,
                    "swirl_strength": 0.010,
                    "blob_count": 8,
                    "blob_radius": 0.035,
                    "always_active": True,
                },
            },
            "task": {
                "grasp_height_offset": 0.018,
                "lift_z": 0.26,
                "align_z": 0.145,
                "insert_z": ee_in_chamber_z,
                "grasp_threshold": 0.055,
                "release_progress": 0.50,
                "phase_steps": {
                    "approach": 50, "descend": 45, "grasp_tray": 25, "lift": 45,
                    "align_to_chamber": 65, "insert": 55,
                    "release": 20, "retract": 40, "hold_done": 120,
                },
            },
            "simulation": deepcopy(_SIM),
        }

    elif task_type == "steam_valve_close":
        ped_pos = [0.62, 0.0, 0.10]
        ped_top_z = ped_pos[2] + 0.05      # 0.15
        valve_z = ped_top_z + 0.013        # 0.163
        contact_offset = 0.048             # y offset: EE contacts valve at [x, y+offset, z]
        return {
            "name": "franka_steam_tasks",
            "task_type": task_type,
            "preset": "default",
            "backend": "auto",
            "seed": 0,
            "headless": True,
            "video": {"enabled": True, "fps": 30, "min_duration_seconds": 3.0},
            "output": {"root_dir": "outputs/franka_steam_tasks/steam_valve_close"},
            "scene": {
                "table": deepcopy(_TABLE),
                "pedestal": {
                    "position": ped_pos,
                    "radius": 0.040,
                    "height": 0.10,
                    "color": [0.42, 0.38, 0.34],
                },
                "valve": {
                    "center": [ped_pos[0], ped_pos[1], valve_z],
                    "radius": 0.036,
                    "height": 0.026,
                    "initial_angle_deg": 0.0,
                    "target_angle_deg": 270.0,
                    "close_threshold_deg": 260.0,
                    "color": [0.62, 0.20, 0.12],
                    "indicator_color": [0.9, 0.9, 0.2],
                },
                "nozzle": {
                    "position": [ped_pos[0], ped_pos[1] - 0.10, ped_top_z - 0.010],
                    "direction": [0.0, 1.0, 0.0],
                    "color": [0.55, 0.55, 0.58],
                },
                "steam": {
                    "origin": [ped_pos[0], ped_pos[1] - 0.050, ped_top_z + 0.010],
                    "bounds": [0.10, 0.14, 0.20],
                    "density": 0.65,
                    "rise_speed": 0.012,
                    "lateral_jitter": 0.012,
                    "swirl_strength": 0.008,
                    "blob_count": 10,
                    "blob_radius": 0.040,
                    "always_active": False,  # active until valve closed
                },
                "steam_stop_angle": 260.0,  # valve angle at which steam stops
            },
            "task": {
                "contact_y_offset": contact_offset,
                "rotate_end_x_offset": -0.050,   # EE x shift during rotate phase
                "rotate_end_y_offset": -contact_offset,
                "phase_steps": {
                    "approach": 55, "contact": 35, "rotate": 80,
                    "hold": 30, "retract": 40, "hold_done": 120,
                },
            },
            "simulation": deepcopy(_SIM),
        }

    raise ValueError(f"Unknown task_type: {task_type!r}")


def get_preset_config(task_type: str, name: str = "default",
                      overrides: Optional[Dict] = None) -> Dict[str, Any]:
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unknown task_type: {task_type!r}  choices: {TASK_TYPES}")
    config = _base_config(task_type)
    if name == "dense_steam":
        config["preset"] = "dense_steam"
        config["scene"]["steam"]["density"] = min(0.92, config["scene"]["steam"]["density"] * 1.5)
        config["scene"]["steam"]["blob_count"] = min(20, config["scene"]["steam"]["blob_count"] + 6)
    elif name != "default":
        raise ValueError(f"Unknown preset: {name!r}")
    config = _deep_update(config, overrides)
    config["preset"] = name
    return config


# ---------------------------------------------------------------------------
# 2-D mock video renderer (top-down schematic)
# ---------------------------------------------------------------------------

def _write_episode_video(path: Path, task_type: str, config: Dict,
                         rows: Sequence[Dict], fps: int = 30) -> bool:
    if not rows:
        path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41")
        return False

    W, H = 640, 360
    margin = 22

    # Scene bounds per task (world-space x/y)
    bounds = {
        "steam_pot_place":      (0.22, 0.88, -0.34, 0.34),
        "steam_lid_open_close": (0.22, 0.88, -0.34, 0.34),
        "spray_chamber_tray":   (0.22, 0.92, -0.36, 0.36),
        "steam_valve_close":    (0.36, 0.80, -0.28, 0.28),
    }
    xmin, xmax, ymin, ymax = bounds.get(task_type, (0.20, 0.90, -0.40, 0.40))
    panel_w, panel_h = 390, 270
    side_left = 440

    def proj(wx: float, wy: float) -> Tuple[int, int]:
        nx = (wx - xmin) / (xmax - xmin) if xmax != xmin else 0
        ny = (wy - ymin) / (ymax - ymin) if ymax != ymin else 0
        return (int(margin + max(0.0, min(1.0, nx)) * panel_w),
                int(margin + panel_h - max(0.0, min(1.0, ny)) * panel_h))

    phase_colors = {
        "approach": _rgb(108, 117, 125), "descend": _rgb(70, 130, 180),
        "grasp": _rgb(230, 159, 0), "grasp_lid": _rgb(230, 159, 0),
        "grasp_tray": _rgb(230, 159, 0), "contact": _rgb(230, 159, 0),
        "lift": _rgb(46, 139, 87), "lift_lid": _rgb(46, 139, 87),
        "move_to_pot": _rgb(196, 90, 48), "lower_into_pot": _rgb(178, 34, 34),
        "hold_open": _rgb(70, 130, 180), "return_to_pot": _rgb(196, 90, 48),
        "lower_lid": _rgb(178, 34, 34), "align_to_chamber": _rgb(196, 90, 48),
        "insert": _rgb(178, 34, 34), "rotate": _rgb(178, 34, 34),
        "release": _rgb(120, 120, 200), "retract": _rgb(112, 128, 144),
        "hold": _rgb(78, 154, 6), "hold_done": _rgb(78, 154, 6),
    }

    step = max(1, len(rows) // 120)
    sampled = list(rows[::step])
    if sampled[-1] is not rows[-1]:
        sampled.append(rows[-1])
    min_frames = max(1, int(fps * float(config.get("video", {}).get("min_duration_seconds", 3.0))))
    while len(sampled) < min_frames:
        sampled.append(sampled[-1])

    table = config["scene"]["table"]
    tool_trail: List[Tuple[int, int]] = []

    with tempfile.TemporaryDirectory(prefix="steam_task_vid_") as tmp:
        frames_dir = Path(tmp)
        for fi, row in enumerate(sampled):
            canvas = _new_canvas(W, H, _rgb(246, 241, 232))
            phase = str(row.get("phase", "approach"))
            ph_col = phase_colors.get(phase, _rgb(120, 120, 120))
            _fill_rect(canvas, W, H, 0, 0, W, 18, ph_col)

            # Panel background
            _fill_rect(canvas, W, H, margin - 8, margin - 8, margin + panel_w + 8, margin + panel_h + 8, _rgb(224, 219, 210))
            _fill_rect(canvas, W, H, margin, margin, margin + panel_w, margin + panel_h, _rgb(255, 252, 247))

            # Table
            tp = table["position"]
            ts = table["size"]
            lo = proj(tp[0] - ts["x"] * 0.5, tp[1] - ts["y"] * 0.5)
            hi = proj(tp[0] + ts["x"] * 0.5, tp[1] + ts["y"] * 0.5)
            _fill_rect(canvas, W, H, lo[0], hi[1], hi[0], lo[1], _rgb(198, 180, 148))

            # Task-specific scene elements
            if task_type in ("steam_pot_place", "steam_lid_open_close"):
                pot = config["scene"]["pot"]
                pc = proj(pot["position"][0], pot["position"][1])
                r = max(5, int(pot["outer_radius"] / (xmax - xmin) * panel_w))
                _draw_circle(canvas, W, H, pc[0], pc[1], r, _rgb(52, 52, 58))
                _draw_circle(canvas, W, H, pc[0], pc[1], max(2, r - 4), _rgb(22, 22, 26))
                if task_type == "steam_pot_place":
                    ox = float(row.get("obj_x", 0.42))
                    oy = float(row.get("obj_y", -0.22))
                    op = proj(ox, oy)
                    _fill_rect(canvas, W, H, op[0] - 7, op[1] - 4, op[0] + 7, op[1] + 4, _rgb(185, 130, 80))
                else:
                    lid_z = float(row.get("lid_z", 0.175))
                    lid_active = lid_z > float(config["scene"].get("steam_trigger_z", 0.22))
                    lp = proj(float(row.get("lid_x", 0.70)), float(row.get("lid_y", 0.10)))
                    lr = max(3, int(0.079 / (xmax - xmin) * panel_w))
                    lid_col = _rgb(180, 180, 185) if not lid_active else _rgb(130, 130, 135)
                    _draw_circle(canvas, W, H, lp[0], lp[1], lr, lid_col)

            elif task_type == "spray_chamber_tray":
                ch = config["scene"]["chamber"]
                cp = ch["position"]
                cs = ch["size"]
                clo = proj(cp[0] - cs["x"] * 0.5, cp[1] - cs["y"] * 0.5)
                chi = proj(cp[0] + cs["x"] * 0.5, cp[1] + cs["y"] * 0.5)
                _fill_rect(canvas, W, H, clo[0], chi[1], chi[0], clo[1], _rgb(58, 64, 68))
                # Opening (lighter face)
                open_x = int((ch["opening_x"] - xmin) / (xmax - xmin) * panel_w + margin)
                _fill_rect(canvas, W, H, open_x - 2, chi[1], open_x + 2, clo[1], _rgb(120, 130, 140))
                tray = config["scene"]["tray"]
                tx = float(row.get("obj_x", tray["position"][0]))
                ty = float(row.get("obj_y", tray["position"][1]))
                tp2 = proj(tx, ty)
                tw = max(6, int(tray["size"][0] * 0.5 / (xmax - xmin) * panel_w))
                th = max(4, int(tray["size"][1] * 0.5 / (ymax - ymin) * panel_h))
                _fill_rect(canvas, W, H, tp2[0] - tw, tp2[1] - th, tp2[0] + tw, tp2[1] + th, _rgb(104, 124, 148))

            elif task_type == "steam_valve_close":
                ped = config["scene"]["pedestal"]
                pp = proj(ped["position"][0], ped["position"][1])
                pr = max(4, int(ped["radius"] / (xmax - xmin) * panel_w))
                _draw_circle(canvas, W, H, pp[0], pp[1], pr, _rgb(102, 90, 80))
                vangle = float(row.get("valve_angle_deg", 0.0))
                va = proj(config["scene"]["valve"]["center"][0], config["scene"]["valve"]["center"][1])
                vr = max(3, int(config["scene"]["valve"]["radius"] / (xmax - xmin) * panel_w))
                _draw_circle(canvas, W, H, va[0], va[1], vr, _rgb(160, 50, 30))
                # Indicator line showing valve angle
                angle_rad = math.radians(vangle)
                ix = int(va[0] + vr * math.cos(angle_rad))
                iy = int(va[1] - vr * math.sin(angle_rad))
                _draw_line(canvas, W, H, va[0], va[1], ix, iy, _rgb(230, 230, 50), 2)

            # Steam cloud
            smoke_active = bool(row.get("steam_active", True))
            strength = max(0.0, min(1.0, float(row.get("steam_strength", 0.5))))
            sx_w = float(row.get("steam_x", config["scene"]["steam"]["origin"][0]))
            sy_w = float(row.get("steam_y", config["scene"]["steam"]["origin"][1]))
            sp = proj(sx_w, sy_w)
            if smoke_active and strength > 0.05:
                for ci in range(7):
                    ox = int(math.cos(fi * 0.18 + ci * 0.9) * (6 + ci * 2))
                    oy = int(-ci * 9 - strength * 10)
                    cr = int(7 + ci * 3 + strength * 8)
                    a = min(0.10 + strength * 0.12, 0.30)
                    _draw_circle(canvas, W, H, sp[0] + ox, sp[1] + oy, cr, _rgb(180, 185, 195), a)

            # EE tool position
            tool_x = float(row.get("tool_x", 0.5))
            tool_y = float(row.get("tool_y", 0.0))
            tp3 = proj(tool_x, tool_y)
            tool_trail.append(tp3)
            tool_trail = tool_trail[-22:]
            for ti in range(1, len(tool_trail)):
                _draw_line(canvas, W, H, tool_trail[ti - 1][0], tool_trail[ti - 1][1],
                           tool_trail[ti][0], tool_trail[ti][1], _rgb(169, 61, 38), 2)
            _draw_circle(canvas, W, H, tp3[0], tp3[1], 6, _rgb(182, 52, 34))

            # Side panel
            success = bool(row.get("success", False))
            gripper_open = bool(row.get("gripper_open", True))
            _fill_rect(canvas, W, H, side_left, 30, side_left + 160, 195, _rgb(255, 250, 242))
            bars = [
                (40, 1.0 if success else 0.0, _rgb(67, 160, 71)),
                (74, 1.0 if not gripper_open else 0.0, _rgb(33, 150, 243)),
                (108, strength if smoke_active else 0.0, _rgb(158, 158, 158)),
            ]
            for top, val, col in bars:
                _fill_rect(canvas, W, H, side_left + 14, top, side_left + 130, top + 16, _rgb(230, 224, 212))
                _fill_rect(canvas, W, H, side_left + 14, top,
                           side_left + 14 + int(116 * max(0.0, min(1.0, val))), top + 16, col)

            _write_ppm(frames_dir / f"frame_{fi:05d}.ppm", W, H, canvas)

        if _encode_mp4(frames_dir, path, fps=fps):
            return True
    path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41")
    return False


# ---------------------------------------------------------------------------
# EpisodeArtifacts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# MockSteamTaskBackend — shared base for all 4 tasks
# ---------------------------------------------------------------------------

class MockSteamTaskBackend:
    """Pure-Python base backend shared across all 4 steam tasks."""

    def __init__(self, config: Dict[str, Any]):
        self.config = deepcopy(config)
        self.task_type: str = config["task_type"]
        self.PHASE_ORDER: List[str] = PHASE_ORDERS[self.task_type]
        self.seed = int(config.get("seed", 0))
        self._rng = random.Random(self.seed)
        self._created = False
        self._step_index = 0
        self._phase = self.PHASE_ORDER[0]
        self._phase_events: List[Dict] = []
        self._stdout_lines: List[str] = []
        self._latest_metrics: Dict = {}
        self._phase_boundaries: List[Tuple[str, int]] = []
        self._tool_position = [0.0, 0.0, 0.0]
        self._target_position = [0.0, 0.0, 0.0]
        self._tool_orientation = _quat_tool_down()
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._ee_position = [0.0, 0.0, 0.0]
        self._ee_orientation = _quat_tool_down()
        self._gripper_open = True
        self._gripper_width = float(config["simulation"]["gripper_width_open"])
        self._success = False
        self._done = False
        self._smoke_mode = "synthetic_plume"
        self._smoke_blobs: List[List[float]] = []
        self._steam_active = bool(config["scene"]["steam"].get("always_active", True))
        self._phase_boundaries = self._compute_phase_boundaries()

    def _log(self, msg: str) -> None:
        self._stdout_lines.append(msg)

    # ---- Phase management ------------------------------------------------

    def _compute_phase_boundaries(self) -> List[Tuple[str, int]]:
        total = 0
        out = []
        for p in self.PHASE_ORDER:
            total += int(self.config["task"]["phase_steps"][p])
            out.append((p, total))
        return out

    def _determine_phase(self, step: int) -> str:
        for name, boundary in self._phase_boundaries:
            if step < boundary:
                return name
        return self.PHASE_ORDER[-1]

    def _phase_start(self, phase: str) -> int:
        total = 0
        for name in self.PHASE_ORDER:
            if name == phase:
                return total
            total += int(self.config["task"]["phase_steps"][name])
        return total

    def _phase_progress(self, phase: str, step: int) -> float:
        start = self._phase_start(phase)
        dur = max(int(self.config["task"]["phase_steps"].get(phase, 1)), 1)
        return min(max((step - start) / dur, 0.0), 1.0)

    def _advance_phase(self) -> None:
        new = self._determine_phase(self._step_index)
        if new != self._phase:
            self._phase = new
            self._phase_events.append({"step": self._step_index, "event": "phase_enter", "phase": new})

    # ---- Smoke / steam ---------------------------------------------------

    def _reset_smoke(self) -> None:
        sc = self.config["scene"]["steam"]
        origin = sc["origin"]
        self._steam_active = bool(sc.get("always_active", True))
        self._smoke_blobs = []
        for i in range(int(sc["blob_count"])):
            self._smoke_blobs.append([
                _round(origin[0] + self._rng.uniform(-0.015, 0.015)),
                _round(origin[1] + self._rng.uniform(-0.015, 0.015)),
                _round(origin[2] + 0.018 * i),
            ])

    def _update_steam_active(self) -> None:
        """Override in subclasses to make steam conditional."""
        sc = self.config["scene"]["steam"]
        if sc.get("always_active", True):
            self._steam_active = True

    def _update_smoke(self) -> None:
        self._update_steam_active()
        if not self._steam_active:
            return
        sc = self.config["scene"]["steam"]
        origin = sc["origin"]
        bounds = sc["bounds"]
        rise = float(sc["rise_speed"])
        jitter = float(sc["lateral_jitter"])
        swirl = float(sc["swirl_strength"])
        phase_shift = self._step_index * 0.10
        updated = []
        for i, blob in enumerate(self._smoke_blobs):
            x = origin[0] + math.sin(phase_shift + i * 0.7) * swirl + self._rng.uniform(-jitter, jitter) * 0.2
            y = origin[1] + math.cos(phase_shift * 0.7 + i * 0.5) * swirl + self._rng.uniform(-jitter, jitter) * 0.2
            z = blob[2] + rise + i * 0.0005
            if z > origin[2] + bounds[2]:
                z = origin[2] + self._rng.uniform(0.0, 0.03)
            updated.append([
                _round(max(origin[0] - bounds[0] * 0.5, min(origin[0] + bounds[0] * 0.5, x))),
                _round(max(origin[1] - bounds[1] * 0.5, min(origin[1] + bounds[1] * 0.5, y))),
                _round(z),
            ])
        self._smoke_blobs = updated

    def _smoke_bounds(self) -> Tuple[List[float], List[float], List[float]]:
        if not self._smoke_blobs or not self._steam_active:
            o = list(self.config["scene"]["steam"]["origin"])
            return o, o, o
        xs = [b[0] for b in self._smoke_blobs]
        ys = [b[1] for b in self._smoke_blobs]
        zs = [b[2] for b in self._smoke_blobs]
        return (_vec3(sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)),
                _vec3(min(xs), min(ys), min(zs)),
                _vec3(max(xs), max(ys), max(zs)))

    def _smoke_strength(self) -> float:
        if not self._steam_active:
            return 0.0
        _, bbox_min, bbox_max = self._smoke_bounds()
        return max(0.0, min(1.0, float(self.config["scene"]["steam"]["density"]) * (
            0.7 + (bbox_max[2] - bbox_min[2]) * 0.5)))

    # ---- Robot state -----------------------------------------------------

    def _update_robot_state(self, prev_tool: Sequence[float]) -> None:
        alpha = float(self.config["simulation"]["tool_follow_alpha"])
        self._target_position = self._compute_planned_target(self._step_index)
        self._tool_position = _vec_lerp(self._tool_position, self._target_position, alpha)
        dt = float(self.config["simulation"]["physics_dt"])
        delta = _vec_sub(self._tool_position, prev_tool)
        self._ee_position = list(self._tool_position)
        self._ee_orientation = list(self._tool_orientation)
        rel = _vec_sub(self._tool_position, self.config["scene"]["table"]["position"])
        prev_j = list(self._robot_joint_positions)
        f = self._gripper_width
        joints = [
            rel[0] * 4.2, rel[1] * 4.4, (self._tool_position[2] - 0.25) * 3.0,
            -1.0 + rel[0] * 1.4, 0.9 + rel[1] * 1.8, 1.45 - rel[0] * 1.1,
            0.70, f, f,
        ]
        self._robot_joint_positions = [_round(v) for v in joints]
        self._robot_joint_velocities = [_round((a - b) / dt) for a, b in zip(self._robot_joint_positions, prev_j)]

    # ---- Abstract interface ----------------------------------------------

    def _compute_planned_target(self, step: int) -> List[float]:
        raise NotImplementedError

    def _reset_objects(self) -> None:
        raise NotImplementedError

    def _apply_gripper_state(self) -> None:
        raise NotImplementedError

    def _update_object_state(self) -> None:
        raise NotImplementedError

    def _compute_metrics(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _object_obs(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _trajectory_row(self) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- Shared public interface -----------------------------------------

    def create_scene(self, config: Optional[Dict] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self._phase_boundaries = self._compute_phase_boundaries()
        self._created = True
        self._log(f"mock {self.task_type} scene created")
        self.reset_scene(self.seed)

    def reset_scene(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.seed = int(self.seed if seed is None else seed)
        self._rng = random.Random(self.seed)
        self._step_index = 0
        self._phase = self.PHASE_ORDER[0]
        self._phase_events = [
            {"step": 0, "event": "reset", "phase": self.PHASE_ORDER[0], "seed": self.seed},
            {"step": 0, "event": "phase_enter", "phase": self.PHASE_ORDER[0]},
        ]
        self._gripper_open = True
        self._gripper_width = float(self.config["simulation"]["gripper_width_open"])
        self._success = False
        self._done = False
        self._reset_smoke()
        self._reset_objects()
        self._target_position = self._compute_planned_target(0)
        self._tool_position = list(self._target_position)
        self._ee_position = list(self._tool_position)
        self._robot_joint_positions = [0.0] * 9
        self._robot_joint_velocities = [0.0] * 9
        self._latest_metrics = self._compute_metrics()
        self._log(f"mock scene reset seed={self.seed}")
        return self.get_obs()

    def step_sim(self, num_steps: int = 1) -> Dict[str, Any]:
        for _ in range(num_steps):
            self._advance_phase()
            prev = list(self._tool_position)
            self._apply_gripper_state()
            self._update_robot_state(prev)
            self._update_object_state()
            self._update_smoke()
            self._latest_metrics = self._compute_metrics()
            self._success = bool(self._latest_metrics["success"])
            self._done = bool(self._latest_metrics["done"])
            self._step_index += 1
            self._advance_phase()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        self._latest_metrics = self._compute_metrics()
        self._success = bool(self._latest_metrics["success"])
        self._done = bool(self._latest_metrics["done"])
        centroid, _, _ = self._smoke_bounds()
        return {
            "robot": {
                "joint_positions": list(self._robot_joint_positions),
                "joint_velocities": list(self._robot_joint_velocities),
                "ee_position": list(self._ee_position),
                "ee_orientation": list(self._ee_orientation),
                "gripper_open": bool(self._gripper_open),
                "gripper_width": _round(self._gripper_width),
            },
            "steam": {
                "mode": self._smoke_mode,
                "active": bool(self._steam_active),
                "origin": list(self.config["scene"]["steam"]["origin"]),
                "centroid": list(centroid),
                "strength": _round(self._smoke_strength()),
            },
            "task": {
                "type": self.task_type,
                "phase": self._phase,
                "done": bool(self._done),
                "grasped": bool(self._latest_metrics.get("grasped", False)),
                "success": bool(self._success),
            },
            **self._object_obs(),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return deepcopy(self._latest_metrics)

    def get_events(self) -> List[Dict]:
        return deepcopy(self._phase_events)

    def get_stdout_log(self) -> str:
        return "\n".join(self._stdout_lines) + ("\n" if self._stdout_lines else "")

    def close(self) -> None:
        self._log(f"mock {self.task_type} scene closed")


# ---------------------------------------------------------------------------
# Task 1: steam_pot_place
# ---------------------------------------------------------------------------

class MockSteamPotBackend(MockSteamTaskBackend):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._object_position = [0.0, 0.0, 0.0]
        self._object_grasped = False

    def _reset_objects(self) -> None:
        self._object_position = [float(v) for v in self.config["scene"]["object"]["position"]]
        self._object_grasped = False

    def _planned_positions(self) -> Dict[str, List[float]]:
        obj = self.config["scene"]["object"]["position"]
        pot = self.config["scene"]["pot"]["position"]
        task = self.config["task"]
        lift_z = float(task["lift_z"])
        pot_approach_z = float(task["pot_approach_z"])
        insert_z = float(task["insert_z"])
        pregrasp_z = float(obj[2]) + 0.13
        grasp_z = float(obj[2]) + float(task["grasp_depth_offset"])
        retract_z = insert_z + 0.22
        start = [float(obj[0]) - 0.18, float(obj[1]) - 0.10, lift_z + 0.10]
        return {
            "start":          start,
            "pregrasp":       [float(obj[0]), float(obj[1]), pregrasp_z],
            "grasp":          [float(obj[0]), float(obj[1]), grasp_z],
            "lift":           [float(obj[0]), float(obj[1]), lift_z],
            "over_pot":       [float(pot[0]), float(pot[1]), pot_approach_z],
            "into_pot":       [float(pot[0]), float(pot[1]), insert_z],
            "retract":        [float(pot[0]), float(pot[1]), retract_z],
        }

    def _compute_planned_target(self, step: int) -> List[float]:
        p = self._planned_positions()
        ph = self._determine_phase(step)
        t = self._phase_progress(ph, step)
        mapping = {
            "approach":       ("start",   "pregrasp"),
            "descend":        ("pregrasp","grasp"),
            "grasp":          ("grasp",   "grasp"),
            "lift":           ("grasp",   "lift"),
            "move_to_pot":    ("lift",    "over_pot"),
            "lower_into_pot": ("over_pot","into_pot"),
            "release":        ("into_pot","into_pot"),
            "retract":        ("into_pot","retract"),
            "hold_done":      ("retract", "retract"),
        }
        a, b = mapping.get(ph, ("start", "start"))
        return _vec_lerp(p[a], p[b], t)

    def _apply_gripper_state(self) -> None:
        closed_phases = {"grasp", "lift", "move_to_pot", "lower_into_pot"}
        if self._phase == "release":
            closed = self._phase_progress("release", self._step_index) < float(self.config["task"]["release_progress"])
        else:
            closed = self._phase in closed_phases
        self._gripper_open = not closed
        self._gripper_width = float(
            self.config["simulation"]["gripper_width_open" if self._gripper_open else "gripper_width_closed"])

    def _update_object_state(self) -> None:
        if not self._object_grasped and not self._gripper_open and self._phase in {"grasp", "lift"}:
            if _vec_distance(self._tool_position, self._object_position) <= 0.060:
                self._object_grasped = True
                self._phase_events.append({"step": self._step_index, "event": "object_grasped"})
                self._log("object grasped")
        if self._object_grasped:
            self._object_position = _vec_add(self._tool_position, [0.0, 0.0, -0.010])
        if self._phase in {"release", "retract", "hold_done"} and self._object_grasped and self._gripper_open:
            self._object_grasped = False
            pot = self.config["scene"]["pot"]["position"]
            obj_half_z = self.config["scene"]["object"]["size"][2] * 0.5
            self._object_position = [float(pot[0]), float(pot[1]),
                                      _POT_FLOOR_Z + obj_half_z + 0.002]
            self._phase_events.append({"step": self._step_index, "event": "object_released_in_pot"})
            self._log("object placed in pot")

    def _in_pot(self) -> bool:
        pot = self.config["scene"]["pot"]["position"]
        dx = abs(self._object_position[0] - pot[0])
        dy = abs(self._object_position[1] - pot[1])
        dz = self._object_position[2] - _POT_FLOOR_Z
        return (math.sqrt(dx * dx + dy * dy) <= float(pot[2]) if False else
                dx <= _POT["inner_radius"] and dy <= _POT["inner_radius"] and 0 <= dz <= _POT["height"])

    def _compute_metrics(self) -> Dict[str, Any]:
        in_pot = self._in_pot()
        success = bool(in_pot and self._gripper_open and not self._object_grasped)
        done = bool(self._phase == "hold_done" and
                    self._phase_progress("hold_done", self._step_index) >= 0.98 and success)
        return {
            "success": success, "done": done,
            "grasped": bool(self._object_grasped),
            "object_in_pot": in_pot,
            "object_height": _round(self._object_position[2]),
            "steam_strength": _round(self._smoke_strength()),
        }

    def _object_obs(self) -> Dict:
        return {
            "object": {
                "type": "block",
                "position": list(self._object_position),
                "size": list(self.config["scene"]["object"]["size"]),
                "grasped": bool(self._object_grasped),
            },
            "target": {
                "type": "pot_interior",
                "position": list(self.config["scene"]["pot"]["position"]),
                "radius": float(self.config["scene"]["pot"]["inner_radius"]),
            },
        }

    def _trajectory_row(self) -> Dict:
        m = self._latest_metrics
        centroid, _, _ = self._smoke_bounds()
        return {
            "obj_x": self._object_position[0], "obj_y": self._object_position[1],
            "obj_z": self._object_position[2],
            "target_x": self.config["scene"]["pot"]["position"][0],
            "target_y": self.config["scene"]["pot"]["position"][1],
            "steam_x": centroid[0], "steam_y": centroid[1], "steam_z": centroid[2],
            "steam_strength": m["steam_strength"], "steam_active": self._steam_active,
        }


# ---------------------------------------------------------------------------
# Task 2: steam_lid_open_close
# ---------------------------------------------------------------------------

class MockSteamLidBackend(MockSteamTaskBackend):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._lid_position = [0.0, 0.0, 0.0]
        self._lid_grasped = False
        self._lid_replaced = False

    def _reset_objects(self) -> None:
        self._lid_position = [float(v) for v in self.config["scene"]["lid"]["position"]]
        self._lid_grasped = False
        self._lid_replaced = False
        self._steam_active = False  # starts inactive (lid on pot)

    def _update_steam_active(self) -> None:
        trigger_z = float(self.config["scene"].get("steam_trigger_z", _POT_TOP_Z + 0.05))
        active_phases = {"lift_lid", "hold_open", "return_to_pot"}
        self._steam_active = (self._phase in active_phases or
                               float(self._lid_position[2]) > trigger_z)

    def _grip_z(self) -> float:
        return float(self.config["task"].get("_handle_grip_z", 0.231))

    def _planned_positions(self) -> Dict[str, List[float]]:
        lid = self.config["scene"]["lid"]["position"]
        task = self.config["task"]
        lift_z = float(task["lift_z"])
        grip_z = self._grip_z()
        pregrasp_z = grip_z + 0.10
        start = [float(lid[0]) - 0.18, float(lid[1]) - 0.08, lift_z + 0.04]
        aside = [float(v) for v in task["aside_position"]]
        return {
            "start":        start,
            "over_lid":     [float(lid[0]), float(lid[1]), pregrasp_z],
            "grasp":        [float(lid[0]), float(lid[1]), grip_z],
            "lifted":       [float(lid[0]), float(lid[1]), lift_z],
            "aside":        aside,
            "over_pot":     [float(lid[0]), float(lid[1]), lift_z],
            "lower_back":   [float(lid[0]), float(lid[1]), grip_z],
            "retract":      start,
        }

    def _compute_planned_target(self, step: int) -> List[float]:
        p = self._planned_positions()
        ph = self._determine_phase(step)
        t = self._phase_progress(ph, step)
        mapping = {
            "approach":     ("start",    "over_lid"),
            "descend":      ("over_lid", "grasp"),
            "grasp_lid":    ("grasp",    "grasp"),
            "lift_lid":     ("grasp",    "lifted"),
            "hold_open":    ("lifted",   "aside"),
            "return_to_pot":("aside",    "over_pot"),
            "lower_lid":    ("over_pot", "lower_back"),
            "release":      ("lower_back","lower_back"),
            "retract":      ("lower_back","retract"),
            "hold_done":    ("retract",  "retract"),
        }
        a, b = mapping.get(ph, ("start", "start"))
        return _vec_lerp(p[a], p[b], t)

    def _apply_gripper_state(self) -> None:
        closed_phases = {"grasp_lid", "lift_lid", "hold_open", "return_to_pot", "lower_lid"}
        if self._phase == "release":
            closed = self._phase_progress("release", self._step_index) < float(self.config["task"]["release_progress"])
        else:
            closed = self._phase in closed_phases
        self._gripper_open = not closed
        self._gripper_width = float(
            self.config["simulation"]["gripper_width_open" if self._gripper_open else "gripper_width_closed"])

    def _update_object_state(self) -> None:
        grip_z = self._grip_z()
        if not self._lid_grasped and not self._gripper_open and self._phase == "grasp_lid":
            if _vec_distance(self._tool_position, [self._lid_position[0], self._lid_position[1], grip_z]) <= 0.055:
                self._lid_grasped = True
                self._phase_events.append({"step": self._step_index, "event": "lid_grasped"})
                self._log("lid grasped")
        if self._lid_grasped:
            # Lid follows tool but stays upright; z offset = handle midpoint below EE
            self._lid_position = [
                self._tool_position[0],
                self._tool_position[1],
                self._tool_position[2] - (grip_z - float(self.config["scene"]["lid"]["position"][2])),
            ]
        if self._phase in {"release", "retract", "hold_done"} and self._lid_grasped and self._gripper_open:
            self._lid_grasped = False
            orig = self.config["scene"]["lid"]["position"]
            self._lid_position = [float(orig[0]), float(orig[1]), float(orig[2])]
            self._lid_replaced = True
            self._phase_events.append({"step": self._step_index, "event": "lid_replaced"})
            self._log("lid replaced on pot")

    def _lid_in_place(self) -> bool:
        orig = self.config["scene"]["lid"]["position"]
        tol = float(self.config["task"].get("replace_tolerance", 0.030))
        return (_vec_distance(self._lid_position, orig) <= tol)

    def _compute_metrics(self) -> Dict[str, Any]:
        in_place = self._lid_in_place()
        success = bool(in_place and self._gripper_open and not self._lid_grasped and self._lid_replaced)
        done = bool(self._phase == "hold_done" and
                    self._phase_progress("hold_done", self._step_index) >= 0.98 and success)
        return {
            "success": success, "done": done,
            "grasped": bool(self._lid_grasped),
            "lid_replaced": bool(self._lid_replaced),
            "lid_position_error": _round(_vec_distance(self._lid_position,
                                                        self.config["scene"]["lid"]["position"])),
            "steam_strength": _round(self._smoke_strength()),
        }

    def _object_obs(self) -> Dict:
        return {
            "object": {
                "type": "lid",
                "position": list(self._lid_position),
                "radius": float(self.config["scene"]["lid"]["radius"]),
                "grasped": bool(self._lid_grasped),
                "replaced": bool(self._lid_replaced),
            },
            "target": {
                "type": "pot_rim",
                "position": list(self.config["scene"]["lid"]["position"]),
                "tolerance": float(self.config["task"].get("replace_tolerance", 0.030)),
            },
        }

    def _trajectory_row(self) -> Dict:
        m = self._latest_metrics
        centroid, _, _ = self._smoke_bounds()
        return {
            "lid_x": self._lid_position[0], "lid_y": self._lid_position[1],
            "lid_z": self._lid_position[2],
            "target_x": self.config["scene"]["lid"]["position"][0],
            "target_y": self.config["scene"]["lid"]["position"][1],
            "steam_x": centroid[0], "steam_y": centroid[1], "steam_z": centroid[2],
            "steam_strength": m["steam_strength"], "steam_active": self._steam_active,
        }


# ---------------------------------------------------------------------------
# Task 3: spray_chamber_tray
# ---------------------------------------------------------------------------

class MockSprayChamberBackend(MockSteamTaskBackend):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._tray_position = [0.0, 0.0, 0.0]
        self._tray_grasped = False

    def _reset_objects(self) -> None:
        self._tray_position = [float(v) for v in self.config["scene"]["tray"]["position"]]
        self._tray_grasped = False

    def _planned_positions(self) -> Dict[str, List[float]]:
        tray = self.config["scene"]["tray"]["position"]
        tray_top_z = float(tray[2]) + self.config["scene"]["tray"]["size"][2] * 0.5
        grasp_z = tray_top_z + float(self.config["task"]["grasp_height_offset"])
        lift_z = float(self.config["task"]["lift_z"])
        ch = self.config["scene"]["chamber"]
        align_z = float(self.config["task"]["align_z"])
        insert_z = float(self.config["task"]["insert_z"])
        chamber_x = float(ch["position"][0])
        chamber_y = float(ch["position"][1])
        opening_x = float(ch["opening_x"])
        retract_z = lift_z
        start = [float(tray[0]) - 0.18, float(tray[1]) - 0.12, lift_z + 0.06]
        return {
            "start":            start,
            "over_tray":        [float(tray[0]), float(tray[1]), grasp_z + 0.12],
            "grasp_pos":        [float(tray[0]), float(tray[1]), grasp_z],
            "lift_pos":         [float(tray[0]), float(tray[1]), lift_z],
            "at_opening":       [opening_x, chamber_y, align_z],
            "in_chamber":       [chamber_x, chamber_y, insert_z],
            "retract_pos":      [opening_x - 0.10, chamber_y, retract_z],
        }

    def _compute_planned_target(self, step: int) -> List[float]:
        p = self._planned_positions()
        ph = self._determine_phase(step)
        t = self._phase_progress(ph, step)
        mapping = {
            "approach":          ("start",      "over_tray"),
            "descend":           ("over_tray",  "grasp_pos"),
            "grasp_tray":        ("grasp_pos",  "grasp_pos"),
            "lift":              ("grasp_pos",  "lift_pos"),
            "align_to_chamber":  ("lift_pos",   "at_opening"),
            "insert":            ("at_opening", "in_chamber"),
            "release":           ("in_chamber", "in_chamber"),
            "retract":           ("in_chamber", "retract_pos"),
            "hold_done":         ("retract_pos","retract_pos"),
        }
        a, b = mapping.get(ph, ("start", "start"))
        return _vec_lerp(p[a], p[b], t)

    def _apply_gripper_state(self) -> None:
        closed_phases = {"grasp_tray", "lift", "align_to_chamber", "insert"}
        if self._phase == "release":
            closed = self._phase_progress("release", self._step_index) < float(self.config["task"]["release_progress"])
        else:
            closed = self._phase in closed_phases
        self._gripper_open = not closed
        self._gripper_width = float(
            self.config["simulation"]["gripper_width_open" if self._gripper_open else "gripper_width_closed"])

    def _update_object_state(self) -> None:
        tray_top_z = float(self.config["scene"]["tray"]["position"][2]) + \
                     self.config["scene"]["tray"]["size"][2] * 0.5
        grasp_z = tray_top_z + float(self.config["task"]["grasp_height_offset"])
        if not self._tray_grasped and not self._gripper_open and self._phase == "grasp_tray":
            dist_xy = math.sqrt((self._tool_position[0] - self._tray_position[0]) ** 2 +
                                (self._tool_position[1] - self._tray_position[1]) ** 2)
            if dist_xy <= float(self.config["task"].get("grasp_threshold", 0.055)):
                self._tray_grasped = True
                self._phase_events.append({"step": self._step_index, "event": "tray_grasped"})
                self._log("tray grasped")
        if self._tray_grasped:
            tray_half_z = self.config["scene"]["tray"]["size"][2] * 0.5
            self._tray_position = [
                self._tool_position[0],
                self._tool_position[1],
                self._tool_position[2] - tray_half_z - float(self.config["task"]["grasp_height_offset"]),
            ]
        if self._phase in {"release", "retract", "hold_done"} and self._tray_grasped and self._gripper_open:
            self._tray_grasped = False
            ch = self.config["scene"]["chamber"]
            self._tray_position = [
                float(ch["position"][0]),
                float(ch["position"][1]),
                float(ch["tray_rest_z"]),
            ]
            self._phase_events.append({"step": self._step_index, "event": "tray_placed_in_chamber"})
            self._log("tray placed in spray chamber")

    def _in_chamber(self) -> bool:
        ch = self.config["scene"]["chamber"]
        cp = ch["position"]
        cs = ch["size"]
        return (abs(self._tray_position[0] - float(cp[0])) <= cs["x"] * 0.5 and
                abs(self._tray_position[1] - float(cp[1])) <= cs["y"] * 0.5 and
                abs(self._tray_position[2] - float(ch["tray_rest_z"])) <= 0.025)

    def _compute_metrics(self) -> Dict[str, Any]:
        in_ch = self._in_chamber()
        success = bool(in_ch and self._gripper_open and not self._tray_grasped)
        done = bool(self._phase == "hold_done" and
                    self._phase_progress("hold_done", self._step_index) >= 0.98 and success)
        return {
            "success": success, "done": done,
            "grasped": bool(self._tray_grasped),
            "tray_in_chamber": in_ch,
            "steam_strength": _round(self._smoke_strength()),
        }

    def _object_obs(self) -> Dict:
        ch = self.config["scene"]["chamber"]
        return {
            "object": {
                "type": "tray",
                "position": list(self._tray_position),
                "size": list(self.config["scene"]["tray"]["size"]),
                "grasped": bool(self._tray_grasped),
            },
            "target": {
                "type": "chamber_interior",
                "position": list(ch["position"]),
                "opening_x": float(ch["opening_x"]),
            },
        }

    def _trajectory_row(self) -> Dict:
        m = self._latest_metrics
        centroid, _, _ = self._smoke_bounds()
        return {
            "obj_x": self._tray_position[0], "obj_y": self._tray_position[1],
            "obj_z": self._tray_position[2],
            "target_x": self.config["scene"]["chamber"]["position"][0],
            "target_y": self.config["scene"]["chamber"]["position"][1],
            "steam_x": centroid[0], "steam_y": centroid[1], "steam_z": centroid[2],
            "steam_strength": m["steam_strength"], "steam_active": self._steam_active,
        }


# ---------------------------------------------------------------------------
# Task 4: steam_valve_close
# ---------------------------------------------------------------------------

class MockSteamValveBackend(MockSteamTaskBackend):
    def __init__(self, config: Dict):
        super().__init__(config)
        self._valve_angle_deg = 0.0

    def _reset_objects(self) -> None:
        self._valve_angle_deg = float(self.config["scene"]["valve"].get("initial_angle_deg", 0.0))
        stop_angle = float(self.config["scene"].get("steam_stop_angle", 260.0))
        self._steam_active = self._valve_angle_deg < stop_angle

    def _update_steam_active(self) -> None:
        stop = float(self.config["scene"].get("steam_stop_angle", 260.0))
        self._steam_active = self._valve_angle_deg < stop

    def _planned_positions(self) -> Dict[str, List[float]]:
        valve = self.config["scene"]["valve"]
        vc = valve["center"]
        task = self.config["task"]
        cy_off = float(task["contact_y_offset"])
        rot_dx = float(task["rotate_end_x_offset"])
        rot_dy = float(task["rotate_end_y_offset"])
        start = [float(vc[0]) - 0.18, float(vc[1]) - 0.12, float(vc[2]) + 0.14]
        contact_approach = [float(vc[0]), float(vc[1]) + cy_off, float(vc[2]) + 0.06]
        contact_pos = [float(vc[0]), float(vc[1]) + cy_off, float(vc[2])]
        end_rotate = [float(vc[0]) + rot_dx, float(vc[1]) + rot_dy, float(vc[2])]
        retract = start
        return {
            "start":           start,
            "contact_approach":contact_approach,
            "contact_pos":     contact_pos,
            "end_rotate":      end_rotate,
            "hold_pos":        end_rotate,
            "retract":         retract,
        }

    def _compute_planned_target(self, step: int) -> List[float]:
        p = self._planned_positions()
        ph = self._determine_phase(step)
        t = self._phase_progress(ph, step)
        mapping = {
            "approach":  ("start",           "contact_approach"),
            "contact":   ("contact_approach","contact_pos"),
            "rotate":    ("contact_pos",     "end_rotate"),
            "hold":      ("end_rotate",      "hold_pos"),
            "retract":   ("hold_pos",        "retract"),
            "hold_done": ("retract",         "retract"),
        }
        a, b = mapping.get(ph, ("start", "start"))
        return _vec_lerp(p[a], p[b], t)

    def _apply_gripper_state(self) -> None:
        # Gripper stays slightly open (simulates push contact, not wrap grasp)
        # Closed during contact and rotate to simulate pressing
        closed_phases = {"contact", "rotate", "hold"}
        self._gripper_open = self._phase not in closed_phases
        self._gripper_width = float(
            self.config["simulation"]["gripper_width_open" if self._gripper_open else "gripper_width_closed"])

    def _update_object_state(self) -> None:
        if self._phase == "rotate":
            target_deg = float(self.config["scene"]["valve"]["target_angle_deg"])
            progress = self._phase_progress("rotate", self._step_index)
            self._valve_angle_deg = min(target_deg, progress * target_deg)

    def _valve_closed(self) -> bool:
        return self._valve_angle_deg >= float(self.config["scene"]["valve"]["close_threshold_deg"])

    def _compute_metrics(self) -> Dict[str, Any]:
        closed = self._valve_closed()
        success = bool(closed)
        done = bool(self._phase == "hold_done" and
                    self._phase_progress("hold_done", self._step_index) >= 0.98 and success)
        return {
            "success": success, "done": done,
            "grasped": False,  # valve task: no grasp
            "valve_angle_deg": _round(self._valve_angle_deg),
            "target_angle_deg": float(self.config["scene"]["valve"]["target_angle_deg"]),
            "valve_closed": closed,
            "steam_strength": _round(self._smoke_strength()),
        }

    def _object_obs(self) -> Dict:
        return {
            "object": {
                "type": "valve",
                "center": list(self.config["scene"]["valve"]["center"]),
                "angle_deg": _round(self._valve_angle_deg),
                "closed": bool(self._valve_closed()),
            },
            "target": {
                "type": "valve_closed_position",
                "angle_deg": float(self.config["scene"]["valve"]["target_angle_deg"]),
            },
        }

    def _trajectory_row(self) -> Dict:
        m = self._latest_metrics
        centroid, _, _ = self._smoke_bounds()
        return {
            "valve_angle_deg": m["valve_angle_deg"],
            "target_angle_deg": m["target_angle_deg"],
            "steam_x": centroid[0], "steam_y": centroid[1], "steam_z": centroid[2],
            "steam_strength": m["steam_strength"], "steam_active": self._steam_active,
        }


# ---------------------------------------------------------------------------
# Isaac backend — shared infrastructure + task-specific scene builders
# ---------------------------------------------------------------------------

class IsaacSteamTaskBackend:
    """Isaac Sim runtime mixin for all steam tasks."""

    def _ensure_runtime(self) -> None:
        if getattr(self, "_runtime_initialized", False):
            return
        from isaacsim import SimulationApp
        self._simulation_app = SimulationApp({"headless": bool(self.config.get("headless", True))})
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver as FrankaKS
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
        self._World = World
        self._Franka = Franka
        self._FrankaKS = FrankaKS
        self._get_assets_root_path = get_assets_root_path
        self._np = np
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._UsdShade = UsdShade
        self._Sdf = Sdf
        self._Gf = Gf
        self._Usd = Usd
        self._smoke_prims: List[str] = []
        self._smoke_particle_state: List = []
        self._franka = None
        self._ik_solver = None
        self._world = None
        self._stage = None
        self._runtime_initialized = True
        self._log("Isaac runtime initialized")

    def _set_translate(self, path: str, pos: Sequence[float]) -> None:
        p = self._stage.GetPrimAtPath(path)
        if p.IsValid():
            self._UsdGeom.XformCommonAPI(p).SetTranslate(tuple(float(v) for v in pos))

    def _define_cube(self, path: str, pos: Sequence[float], scale: Sequence[float], color: Sequence[float]) -> None:
        c = self._UsdGeom.Cube.Define(self._stage, path)
        c.CreateSizeAttr(1.0)
        c.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._UsdGeom.XformCommonAPI(self._stage.GetPrimAtPath(path)).SetTranslate(
            tuple(float(v) for v in pos))
        self._UsdGeom.XformCommonAPI(self._stage.GetPrimAtPath(path)).SetScale(
            tuple(float(v) for v in scale))

    def _define_cylinder(self, path: str, pos: Sequence[float], radius: float, height: float,
                          color: Sequence[float]) -> None:
        cyl = self._UsdGeom.Cylinder.Define(self._stage, path)
        cyl.CreateRadiusAttr(float(radius))
        cyl.CreateHeightAttr(float(height))
        cyl.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(path, pos)

    def _define_sphere(self, path: str, pos: Sequence[float], radius: float, color: Sequence[float]) -> None:
        s = self._UsdGeom.Sphere.Define(self._stage, path)
        s.CreateRadiusAttr(float(radius))
        s.CreateDisplayColorAttr().Set([self._Gf.Vec3f(*[float(v) for v in color])])
        self._set_translate(path, pos)

    def _make_mat(self, mat_path: str, color: Sequence[float], opacity: float = 1.0,
                  roughness: float = 0.5, metallic: float = 0.0,
                  emissive: Optional[Sequence[float]] = None):
        mat = self._UsdShade.Material.Define(self._stage, mat_path)
        sh = self._UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(
            self._Gf.Vec3f(*[float(v) for v in color]))
        sh.CreateInput("opacity", self._Sdf.ValueTypeNames.Float).Set(float(opacity))
        sh.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(float(roughness))
        sh.CreateInput("metallic", self._Sdf.ValueTypeNames.Float).Set(float(metallic))
        if emissive:
            sh.CreateInput("emissiveColor", self._Sdf.ValueTypeNames.Color3f).Set(
                self._Gf.Vec3f(*[float(v) for v in emissive]))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def _bind_mat(self, path: str, mat) -> None:
        p = self._stage.GetPrimAtPath(path)
        if p.IsValid():
            self._UsdShade.MaterialBindingAPI.Apply(p).Bind(mat)

    def _create_particle_cloud_smoke(self) -> None:
        try:
            sc = self.config["scene"]["steam"]
            origin = sc["origin"]
            N = int(sc["blob_count"]) * 15
            rng = random.Random(getattr(self, "seed", 42))
            spread_r = float(sc["bounds"][0]) * 0.45
            max_h = float(sc["bounds"][2]) * 0.9
            smoke_z0 = float(origin[2])
            mat = self._make_mat("/World/Looks/SteamParticle", [0.82, 0.84, 0.88],
                                 opacity=0.45, roughness=1.0)
            self._smoke_particle_state = []
            self._smoke_prims = []
            for i in range(N):
                angle = rng.uniform(0, 2 * math.pi)
                r = math.sqrt(rng.random()) * spread_r
                px = float(origin[0]) + r * math.cos(angle)
                py = float(origin[1]) + r * math.sin(angle)
                pz = smoke_z0 + rng.random() * max_h
                vx = rng.gauss(0, 0.003)
                vy = rng.gauss(0, 0.003)
                vz = rng.uniform(0.002, 0.008)
                self._smoke_particle_state.append([px, py, pz, vx, vy, vz])
                prim_path = f"/World/SteamCloud/P_{i:03d}"
                self._define_sphere(prim_path, [px, py, pz], float(sc["blob_radius"]), [0.80, 0.82, 0.86])
                self._bind_mat(prim_path, mat)
                self._smoke_prims.append(prim_path)
            self._smoke_mode = "particle_cloud"
            self._log(f"Particle cloud steam: {N} prims")
        except Exception as exc:
            self._smoke_mode = "synthetic_plume"
            self._log(f"Particle cloud steam failed: {exc}")

    def _update_smoke(self) -> None:  # type: ignore[override]
        self._update_steam_active()
        if not self._smoke_particle_state:
            # Fallback to parent (synthetic blobs)
            super()._update_smoke()  # type: ignore[misc]
            return
        if not self._steam_active:
            return
        try:
            sc = self.config["scene"]["steam"]
            origin = sc["origin"]
            max_z = float(origin[2]) + float(sc["bounds"][2])
            spawn_r = float(sc["bounds"][0]) * 0.35
            for i, p in enumerate(self._smoke_particle_state):
                p[0] += p[3] + self._rng.gauss(0, 0.002)  # type: ignore[attr-defined]
                p[1] += p[4] + self._rng.gauss(0, 0.002)
                p[2] += p[5]
                p[5] = max(0.001, p[5] * 0.999)
                if p[2] > max_z:
                    angle = self._rng.uniform(0, 2 * math.pi)  # type: ignore[attr-defined]
                    r = math.sqrt(self._rng.random()) * spawn_r  # type: ignore[attr-defined]
                    p[0] = float(origin[0]) + r * math.cos(angle)
                    p[1] = float(origin[1]) + r * math.sin(angle)
                    p[2] = float(origin[2]) + self._rng.uniform(0, 0.03)  # type: ignore[attr-defined]
                    p[3] = self._rng.gauss(0, 0.003)  # type: ignore[attr-defined]
                    p[4] = self._rng.gauss(0, 0.003)
                    p[5] = self._rng.uniform(0.002, 0.008)  # type: ignore[attr-defined]
                if i < len(self._smoke_prims):
                    self._set_translate(self._smoke_prims[i], [p[0], p[1], p[2]])
        except Exception as exc:
            self._log(f"Smoke update warning: {exc}")

    def _setup_lighting(self) -> None:
        dome = self._UsdLux.DomeLight.Define(self._stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(700.0)
        dome.CreateColorAttr(self._Gf.Vec3f(0.88, 0.92, 1.0))
        key = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Key")
        key.CreateIntensityAttr(48000.0)
        key.CreateRadiusAttr(0.12)
        key.CreateColorAttr(self._Gf.Vec3f(1.0, 0.97, 0.90))
        self._set_translate("/World/Lights/Key", [0.95, -0.80, 1.10])
        fill = self._UsdLux.SphereLight.Define(self._stage, "/World/Lights/Fill")
        fill.CreateIntensityAttr(12000.0)
        fill.CreateRadiusAttr(0.10)
        fill.CreateColorAttr(self._Gf.Vec3f(0.82, 0.90, 1.0))
        self._set_translate("/World/Lights/Fill", [0.26, 0.60, 0.82])

    def _build_common_scene(self) -> None:
        """Table + lighting + tool marker + Franka (shared by all tasks)."""
        self._world = self._World(stage_units_in_meters=1.0)
        self._stage = self._simulation_app.context.get_stage()
        self._world.scene.add_default_ground_plane()
        self._setup_lighting()

        table = self.config["scene"]["table"]
        t_mat = self._make_mat("/World/Looks/TableWood", table["color"], roughness=0.72)
        self._define_cube("/World/Table", table["position"],
                          [table["size"]["x"], table["size"]["y"], table["size"]["z"]], table["color"])
        self._bind_mat("/World/Table", t_mat)

        self._define_sphere("/World/ToolMarker", self._tool_position, 0.012, [0.85, 0.18, 0.12])

        # Franka robot
        try:
            self._assets_root_path = self._get_assets_root_path()
            robot_cfg = self.config["scene"]["robot"]
            robot_usd = self._assets_root_path + robot_cfg["asset_path"]
            self._franka = self._Franka(
                prim_path=robot_cfg["prim_path"],
                name=f"franka_{self.task_type}",
                usd_path=robot_usd,
                end_effector_prim_name=robot_cfg["ee_name"],
            )
            self._world.scene.add(self._franka)
            self._log(f"Franka loaded from {robot_usd}")
        except Exception as exc:
            self._franka = None
            self._log(f"Robot load skipped: {exc}")

        self._world.reset()
        if self._franka is not None:
            try:
                robot_cfg = self.config["scene"]["robot"]
                self._ik_solver = self._FrankaKS(
                    self._franka,
                    end_effector_frame_name=robot_cfg["ik_frame_name"],
                )
            except Exception as exc:
                self._ik_solver = None
                self._log(f"IK setup skipped: {exc}")

    def _build_steam_pot_scene(self) -> None:
        pot = self.config["scene"]["pot"]
        pp = pot["position"]
        # Outer cylinder
        pot_mat = self._make_mat("/World/Looks/Pot", pot["color"], roughness=0.35, metallic=0.7)
        self._define_cylinder("/World/Pot/Body", pp, float(pot["outer_radius"]), float(pot["height"]), pot["color"])
        self._bind_mat("/World/Pot/Body", pot_mat)
        # Rim highlight
        rim_mat = self._make_mat("/World/Looks/PotRim", pot["rim_color"], roughness=0.2, metallic=0.9)
        rim_z = float(pp[2]) + float(pot["height"]) * 0.5
        self._define_cylinder("/World/Pot/Rim", [pp[0], pp[1], rim_z + 0.004],
                               float(pot["outer_radius"]) + 0.003, 0.008, pot["rim_color"])
        self._bind_mat("/World/Pot/Rim", rim_mat)
        # Object on tray
        obj = self.config["scene"]["object"]
        obj_mat = self._make_mat("/World/Looks/Object", obj["color"], roughness=0.6)
        self._define_cube("/World/Object", obj["position"], obj["size"], obj["color"])
        self._bind_mat("/World/Object", obj_mat)
        # Small tray (just a thin flat surface)
        tray_pos = [float(obj["position"][0]), float(obj["position"][1]), _TABLE_TOP_Z + 0.003]
        self._define_cube("/World/Tray", tray_pos, [0.14, 0.14, 0.006], [0.55, 0.55, 0.58])

    def _build_steam_lid_scene(self) -> None:
        # Pot
        pot = self.config["scene"]["pot"]
        pp = pot["position"]
        pot_mat = self._make_mat("/World/Looks/Pot", pot["color"], roughness=0.35, metallic=0.7)
        self._define_cylinder("/World/Pot/Body", pp, float(pot["outer_radius"]), float(pot["height"]), pot["color"])
        self._bind_mat("/World/Pot/Body", pot_mat)
        # Lid disc
        lid = self.config["scene"]["lid"]
        lid_mat = self._make_mat("/World/Looks/Lid", lid["color"], roughness=0.4, metallic=0.6)
        self._define_cylinder("/World/Lid/Disc", lid["position"],
                               float(lid["radius"]), float(lid["height"]), lid["color"])
        self._bind_mat("/World/Lid/Disc", lid_mat)
        # Handle
        handle_pos = [float(lid["position"][0]), float(lid["position"][1]),
                      float(lid["position"][2]) + float(lid["height"]) * 0.5 + float(lid["handle_height"]) * 0.5]
        handle_mat = self._make_mat("/World/Looks/LidHandle", lid["handle_color"], roughness=0.5, metallic=0.5)
        self._define_cylinder("/World/Lid/Handle", handle_pos,
                               float(lid["handle_radius"]), float(lid["handle_height"]), lid["handle_color"])
        self._bind_mat("/World/Lid/Handle", handle_mat)
        # Small steam vent emissive
        boil_mat = self._make_mat("/World/Looks/Boiling", [0.8, 0.85, 0.9], opacity=0.08, roughness=1.0)

    def _build_spray_chamber_scene(self) -> None:
        ch = self.config["scene"]["chamber"]
        cp = ch["position"]
        cs = ch["size"]
        ch_mat = self._make_mat("/World/Looks/Chamber", ch["color"], roughness=0.6, metallic=0.3)
        # Walls (left, right, back, top, floor) — leave front face open
        wall_thickness = 0.010
        # Left wall
        self._define_cube("/World/Chamber/WallLeft",
                          [cp[0], cp[1] - cs["y"] * 0.5 + wall_thickness * 0.5, cp[2]],
                          [cs["x"], wall_thickness, cs["z"]], ch["color"])
        # Right wall
        self._define_cube("/World/Chamber/WallRight",
                          [cp[0], cp[1] + cs["y"] * 0.5 - wall_thickness * 0.5, cp[2]],
                          [cs["x"], wall_thickness, cs["z"]], ch["color"])
        # Back wall
        self._define_cube("/World/Chamber/WallBack",
                          [cp[0] + cs["x"] * 0.5 - wall_thickness * 0.5, cp[1], cp[2]],
                          [wall_thickness, cs["y"], cs["z"]], ch["color"])
        # Top
        self._define_cube("/World/Chamber/Top",
                          [cp[0], cp[1], cp[2] + cs["z"] * 0.5 - wall_thickness * 0.5],
                          [cs["x"], cs["y"], wall_thickness], ch["color"])
        # Floor
        self._define_cube("/World/Chamber/Floor",
                          [cp[0], cp[1], cp[2] - cs["z"] * 0.5 + wall_thickness * 0.5],
                          [cs["x"], cs["y"], wall_thickness], ch["interior_color"])
        for w in ["WallLeft", "WallRight", "WallBack", "Top", "Floor"]:
            self._bind_mat(f"/World/Chamber/{w}", ch_mat)
        # Tray on table
        tray = self.config["scene"]["tray"]
        tray_mat = self._make_mat("/World/Looks/Tray", tray["color"], roughness=0.5, metallic=0.2)
        self._define_cube("/World/Tray", tray["position"], tray["size"], tray["color"])
        self._bind_mat("/World/Tray", tray_mat)
        # Spray nozzles inside chamber (visual only — 3 sphere prims)
        nozzle_mat = self._make_mat("/World/Looks/Nozzle", [0.7, 0.7, 0.75], roughness=0.3, metallic=0.7)
        for ni, ny in enumerate([-0.08, 0.0, 0.08]):
            npos = [cp[0], cp[1] + ny, cp[2] + cs["z"] * 0.5 - 0.015]
            self._define_sphere(f"/World/Chamber/Nozzle_{ni}", npos, 0.008, [0.7, 0.7, 0.75])
            self._bind_mat(f"/World/Chamber/Nozzle_{ni}", nozzle_mat)

    def _build_steam_valve_scene(self) -> None:
        ped = self.config["scene"]["pedestal"]
        pp = ped["position"]
        ped_mat = self._make_mat("/World/Looks/Pedestal", ped["color"], roughness=0.7, metallic=0.1)
        self._define_cylinder("/World/Pedestal", pp, float(ped["radius"]), float(ped["height"]), ped["color"])
        self._bind_mat("/World/Pedestal", ped_mat)
        # Valve knob
        valve = self.config["scene"]["valve"]
        vc = valve["center"]
        valve_mat = self._make_mat("/World/Looks/Valve", valve["color"], roughness=0.4, metallic=0.6)
        self._define_cylinder("/World/Valve", vc, float(valve["radius"]), float(valve["height"]), valve["color"])
        self._bind_mat("/World/Valve", valve_mat)
        # Indicator notch on knob (emissive yellow)
        ind_mat = self._make_mat("/World/Looks/Indicator", valve["indicator_color"], roughness=0.5,
                                 emissive=[2.0, 2.0, 0.2])
        angle_rad = math.radians(float(valve.get("initial_angle_deg", 0.0)))
        ind_pos = [float(vc[0]) + float(valve["radius"]) * 0.7 * math.cos(angle_rad),
                   float(vc[1]) + float(valve["radius"]) * 0.7 * math.sin(angle_rad),
                   float(vc[2]) + float(valve["height"]) * 0.5 + 0.002]
        self._define_sphere("/World/ValveIndicator", ind_pos, 0.006, valve["indicator_color"])
        self._bind_mat("/World/ValveIndicator", ind_mat)
        # Nozzle (steam leak source)
        nozzle = self.config["scene"]["nozzle"]
        nozzle_mat = self._make_mat("/World/Looks/Nozzle", nozzle["color"], roughness=0.4, metallic=0.8)
        self._define_cylinder("/World/Nozzle", nozzle["position"], 0.012, 0.040, nozzle["color"])
        self._bind_mat("/World/Nozzle", nozzle_mat)

    def _build_runtime_scene(self) -> None:
        self._build_common_scene()
        builders = {
            "steam_pot_place":      self._build_steam_pot_scene,
            "steam_lid_open_close": self._build_steam_lid_scene,
            "spray_chamber_tray":   self._build_spray_chamber_scene,
            "steam_valve_close":    self._build_steam_valve_scene,
        }
        builders[self.task_type]()
        self._create_particle_cloud_smoke()

    def _command_robot_to_target(self) -> None:
        if self._franka is None or self._ik_solver is None:
            return
        try:
            pos = self._np.asarray(self._target_position, dtype=float)
            ori = self._np.asarray(self._tool_orientation, dtype=float)
            action, ok = self._ik_solver.compute_inverse_kinematics(
                pos, target_orientation=ori, position_tolerance=0.005, orientation_tolerance=0.25)
            if ok:
                self._franka.apply_action(action)
        except Exception as exc:
            self._log(f"IK command warning: {exc}")

    def _sync_runtime_scene(self) -> None:
        self._set_translate("/World/ToolMarker", self._tool_position)
        # Sync object position
        if self.task_type == "steam_pot_place":
            self._set_translate("/World/Object", self._object_position)  # type: ignore[attr-defined]
        elif self.task_type == "steam_lid_open_close":
            lp = self._lid_position  # type: ignore[attr-defined]
            self._set_translate("/World/Lid/Disc", lp)
            lid_h = float(self.config["scene"]["lid"]["height"])
            handle_h = float(self.config["scene"]["lid"]["handle_height"])
            handle_z = float(lp[2]) + lid_h * 0.5 + handle_h * 0.5
            self._set_translate("/World/Lid/Handle", [lp[0], lp[1], handle_z])
        elif self.task_type == "spray_chamber_tray":
            self._set_translate("/World/Tray", self._tray_position)  # type: ignore[attr-defined]
        elif self.task_type == "steam_valve_close":
            # Rotate indicator with valve
            angle_rad = math.radians(float(self._valve_angle_deg))  # type: ignore[attr-defined]
            vc = self.config["scene"]["valve"]["center"]
            r = float(self.config["scene"]["valve"]["radius"])
            ind_pos = [float(vc[0]) + r * 0.7 * math.cos(angle_rad),
                       float(vc[1]) + r * 0.7 * math.sin(angle_rad),
                       float(vc[2]) + float(self.config["scene"]["valve"]["height"]) * 0.5 + 0.002]
            self._set_translate("/World/ValveIndicator", ind_pos)

    def create_scene(self, config: Optional[Dict] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
            self._phase_boundaries = self._compute_phase_boundaries()  # type: ignore[attr-defined]
        if not importlib.util.find_spec("isaacsim"):
            raise RuntimeError("Isaac backend requires an Isaac Sim Python runtime.")
        # Inject robot config if not present
        if "robot" not in self.config["scene"]:
            self.config["scene"]["robot"] = {
                "asset_path": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
                "prim_path": "/World/Franka",
                "ee_name": "panda_hand",
                "ik_frame_name": "panda_hand",
            }
        self._ensure_runtime()
        self._reset_smoke()  # type: ignore[attr-defined]
        self._reset_objects()  # type: ignore[attr-defined]
        self._build_runtime_scene()
        self._created = True  # type: ignore[attr-defined]
        self.reset_scene(getattr(self, "seed", 0))  # type: ignore[attr-defined]

    def reset_scene(self, seed: Optional[int] = None) -> Dict:
        obs = super().reset_scene(seed)  # type: ignore[misc]
        if getattr(self, "_world", None) is not None:
            try:
                self._world.reset()
            except Exception as exc:
                self._log(f"World reset warning: {exc}")
        self._sync_runtime_scene()
        self._phase_events.append(  # type: ignore[attr-defined]
            {"step": 0, "event": "runtime_mode", "backend": "isaac", "steam_mode": self._smoke_mode})
        return obs

    def step_sim(self, num_steps: int = 1) -> Dict:
        for _ in range(num_steps):
            prev = list(self._tool_position)  # type: ignore[attr-defined]
            self._advance_phase()  # type: ignore[attr-defined]
            self._apply_gripper_state()  # type: ignore[attr-defined]
            self._target_position = self._compute_planned_target(self._step_index)  # type: ignore[attr-defined]
            if getattr(self, "_world", None) is not None:
                substeps = int(self.config["simulation"]["robot_settle_substeps"])
                for _ in range(substeps):
                    self._command_robot_to_target()
                    try:
                        self._world.step(render=False)
                    except Exception as exc:
                        self._log(f"World step warning: {exc}")
                        break
            self._update_robot_state(prev)  # type: ignore[attr-defined]
            self._update_object_state()  # type: ignore[attr-defined]
            self._update_smoke()
            self._latest_metrics = self._compute_metrics()  # type: ignore[attr-defined]
            self._success = bool(self._latest_metrics["success"])  # type: ignore[attr-defined]
            self._done = bool(self._latest_metrics["done"])  # type: ignore[attr-defined]
            self._sync_runtime_scene()
            self._step_index += 1  # type: ignore[attr-defined]
            self._advance_phase()  # type: ignore[attr-defined]
        return self.get_obs()  # type: ignore[attr-defined]

    def close(self) -> None:
        if getattr(self, "_simulation_app", None) is not None:
            try:
                self._simulation_app.close()
            except Exception:
                pass
            self._simulation_app = None
        super().close()  # type: ignore[misc]


# Concrete Isaac backends (dispatch handled by IsaacSteamTaskBackend._build_runtime_scene)
class IsaacSteamPotBackend(IsaacSteamTaskBackend, MockSteamPotBackend):
    pass

class IsaacSteamLidBackend(IsaacSteamTaskBackend, MockSteamLidBackend):
    pass

class IsaacSprayChamberBackend(IsaacSteamTaskBackend, MockSprayChamberBackend):
    pass

class IsaacSteamValveBackend(IsaacSteamTaskBackend, MockSteamValveBackend):
    pass


# ---------------------------------------------------------------------------
# FrankaSteamTaskEnv — public API
# ---------------------------------------------------------------------------

_BACKEND_CLASSES = {
    "steam_pot_place":      (MockSteamPotBackend,      IsaacSteamPotBackend),
    "steam_lid_open_close": (MockSteamLidBackend,      IsaacSteamLidBackend),
    "spray_chamber_tray":   (MockSprayChamberBackend,  IsaacSprayChamberBackend),
    "steam_valve_close":    (MockSteamValveBackend,    IsaacSteamValveBackend),
}


class FrankaSteamTaskEnv:
    def __init__(self, task_type: str, config: Optional[Dict] = None,
                 preset: str = "default", backend: str = "auto"):
        if task_type not in TASK_TYPES:
            raise ValueError(f"Unknown task_type: {task_type!r}")
        self.task_type = task_type
        self.config = get_preset_config(task_type, preset, config)
        self.config["backend"] = backend
        self.backend_name = "isaac" if (backend == "isaac" or
                                        (backend == "auto" and importlib.util.find_spec("isaacsim"))) else "mock"
        mock_cls, isaac_cls = _BACKEND_CLASSES[task_type]
        self._backend: MockSteamTaskBackend = (
            isaac_cls(self.config) if self.backend_name == "isaac" else mock_cls(self.config))

    def create_scene(self, config: Optional[Dict] = None) -> None:
        if config is not None:
            self.config = deepcopy(config)
        self._backend.create_scene(config or self.config)

    def reset_scene(self, seed: Optional[int] = None) -> Dict:
        return self._backend.reset_scene(seed)

    def get_obs(self) -> Dict:
        return self._backend.get_obs()

    def step_sim(self, n: int = 1) -> Dict:
        return self._backend.step_sim(n)

    def close(self) -> None:
        self._backend.close()

    def get_metrics(self) -> Dict:
        return self._backend.get_metrics()

    def get_events(self) -> List[Dict]:
        return self._backend.get_events()

    def run_episode(self, seed: Optional[int] = None, output_dir: Optional[Path] = None,
                    max_steps: Optional[int] = None) -> Tuple[Dict, EpisodeArtifacts]:
        run_seed = int(self.config.get("seed", 0) if seed is None else seed)
        self.config["seed"] = run_seed
        if not getattr(self._backend, "_created", False):
            self.create_scene(self.config)
        self.reset_scene(run_seed)

        out = Path(output_dir) if output_dir else (
            Path(self.config["output"]["root_dir"]) /
            f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_{run_seed}")
        out.mkdir(parents=True, exist_ok=True)

        artifacts = EpisodeArtifacts(
            output_dir=out,
            config_path=out / "config.json",
            obs_schema_path=out / "obs_schema.json",
            metrics_path=out / "metrics.json",
            episode_summary_path=out / "episode_summary.json",
            trajectory_path=out / "trajectory.csv",
            events_path=out / "events.jsonl",
            video_path=out / "video.mp4",
            stdout_path=out / "stdout.log",
        )
        _write_json(artifacts.config_path, self.config)
        _write_json(artifacts.obs_schema_path, {"task_type": self.task_type,
                                                  "phase_order": PHASE_ORDERS[self.task_type]})

        budget = max_steps or (sum(self.config["task"]["phase_steps"].values()) + 5)
        rows: List[Dict] = []
        status = "completed"
        error_text = None

        try:
            for _ in range(budget):
                obs = self.get_obs()
                m = self.get_metrics()
                base_row = {
                    "step": getattr(self._backend, "_step_index", 0),
                    "phase": obs["task"]["phase"],
                    "task_done": obs["task"]["done"],
                    "success": obs["task"]["success"],
                    "grasped": obs["task"]["grasped"],
                    "tool_x": obs["robot"]["ee_position"][0],
                    "tool_y": obs["robot"]["ee_position"][1],
                    "tool_z": obs["robot"]["ee_position"][2],
                    "gripper_open": obs["robot"]["gripper_open"],
                    "steam_strength": obs["steam"]["strength"],
                    "steam_active": obs["steam"]["active"],
                }
                task_row = self._backend._trajectory_row()
                rows.append({**base_row, **task_row})
                if obs["task"]["done"]:
                    break
                self.step_sim(1)
            else:
                status = "aborted"
        except Exception as exc:
            status = "error"
            error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()

        final_metrics = self.get_metrics()
        phase_seq = [e["phase"] for e in self.get_events() if e["event"] == "phase_enter"]
        summary = {
            "status": status,
            "backend": self.backend_name,
            "task_type": self.task_type,
            "preset": self.config["preset"],
            "seed": run_seed,
            "step_count": getattr(self._backend, "_step_index", 0),
            "phase_sequence": phase_seq,
            "steam_mode": getattr(self._backend, "_smoke_mode", "synthetic_plume"),
            "error": error_text,
            "metrics": final_metrics,
        }

        if rows:
            with artifacts.trajectory_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        _append_jsonl(artifacts.events_path, self.get_events())
        _write_json(artifacts.metrics_path, {"backend": self.backend_name,
                                              "task_type": self.task_type, "seed": run_seed,
                                              "final": final_metrics, "phase_sequence": phase_seq})
        _write_json(artifacts.episode_summary_path, summary)
        if self.config.get("video", {}).get("enabled", True):
            _write_episode_video(artifacts.video_path, self.task_type, self.config, rows,
                                 fps=int(self.config["video"]["fps"]))
        else:
            artifacts.video_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2mp41")
        artifacts.stdout_path.write_text(self._backend.get_stdout_log(), encoding="utf-8")
        return summary, artifacts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="FluidBench steam/spray manipulation tasks")
    parser.add_argument("--task", choices=TASK_TYPES, required=True)
    parser.add_argument("--preset", choices=["default", "dense_steam"], default="default")
    parser.add_argument("--backend", choices=["auto", "isaac", "mock"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--disable-video", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    config: Dict[str, Any] = {"seed": args.seed, "headless": args.headless}
    if args.output_root:
        config["output"] = {"root_dir": args.output_root}
    if args.disable_video:
        config["video"] = {"enabled": False}

    env = FrankaSteamTaskEnv(args.task, config=config, preset=args.preset, backend=args.backend)
    try:
        summary, artifacts = env.run_episode(seed=args.seed, max_steps=args.max_steps)
        print(json.dumps({"summary": summary, "output_dir": str(artifacts.output_dir)}, indent=2, sort_keys=True))
        return 0 if summary["status"] == "completed" else 1
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
