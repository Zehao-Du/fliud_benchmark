#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


CAMERA_PRESETS = ["wide_front", "wide_diag", "wide_side", "top_diag_far"]
LABELS = {
    "wide_front": "wide_front",
    "wide_diag": "wide_diag",
    "wide_side": "wide_side",
    "top_diag_far": "top_diag_far",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and compose a four-view Franka stirring video.")
    parser.add_argument("--preset", choices=["default", "stress"], default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/franka_stir_liquid_multiview"))
    return parser.parse_args()


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _render_view(repo_root: Path, args: argparse.Namespace, camera_preset: str, output_dir: Path) -> Path:
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "render_franka_stir_real_video.py"),
        "--preset",
        args.preset,
        "--seed",
        str(args.seed),
        "--frames",
        str(args.frames),
        "--fps",
        str(args.fps),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--camera-preset",
        camera_preset,
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)
    return output_dir / "demo.mp4"


def _annotate(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    frame[:40, :, :] = (frame[:40, :, :] * 0.35).astype(np.uint8)
    x0 = 16
    y0 = 10
    for idx, ch in enumerate(text[:24]):
        value = ord(ch)
        for bit in range(8):
            if value & (1 << bit):
                x = x0 + idx * 10 + bit
                frame[y0 : y0 + 14, x : x + 2] = np.array([255, 255, 255], dtype=np.uint8)
    return frame


def _compose_grid(video_paths: list[Path], labels: list[str], output_path: Path, fps: int) -> None:
    readers = [imageio.get_reader(path) for path in video_paths]
    try:
        frame_count = min(reader.count_frames() for reader in readers)
        with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
            for frame_idx in range(frame_count):
                frames = []
                for reader, label in zip(readers, labels):
                    frames.append(_annotate(reader.get_data(frame_idx), label))
                top = np.concatenate([frames[0], frames[1]], axis=1)
                bottom = np.concatenate([frames[2], frames[3]], axis=1)
                writer.append_data(np.concatenate([top, bottom], axis=0))
    finally:
        for reader in readers:
            reader.close()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    _reset_dir(output_dir)

    rendered_videos = []
    for camera_preset in CAMERA_PRESETS:
        view_dir = output_dir / camera_preset
        rendered_videos.append(_render_view(repo_root, args, camera_preset, view_dir))

    grid_video = output_dir / "demo.mp4"
    _compose_grid(rendered_videos, [LABELS[name] for name in CAMERA_PRESETS], grid_video, args.fps)
    lines = [f"{name}: {path}" for name, path in zip(CAMERA_PRESETS, rendered_videos)]
    lines.append(f"grid: {grid_video}")
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(grid_video)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
