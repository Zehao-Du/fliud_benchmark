#!/usr/bin/env python3

import argparse
import shutil
from pathlib import Path

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a simple Isaac Sim demo video in headless mode.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo_video"), help="Directory for frames/video.")
    parser.add_argument("--frames", type=int, default=120, help="Number of frames to capture.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the output video.")
    parser.add_argument("--width", type=int, default=1280, help="Video width.")
    parser.add_argument("--height", type=int, default=720, help="Video height.")
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp(launch_config={"headless": True})

import carb.settings
import imageio.v2 as imageio
import omni.replicator.core as rep
import omni.timeline
import omni.usd


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _find_rgb_frames(output_dir: Path) -> list[Path]:
    frames = sorted(output_dir.rglob("*.png"))
    if not frames:
        raise RuntimeError(f"No rendered PNG frames found under {output_dir}")
    return frames


def _write_video(frames: list[Path], output_path: Path, fps: int) -> None:
    with imageio.get_writer(output_path, fps=fps, codec="libx264", format="FFMPEG") as writer:
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))


def generate_video() -> Path:
    output_dir = args.output_dir.resolve()
    frame_dir = output_dir / "frames"
    video_path = output_dir / "demo.mp4"
    _reset_output_dir(output_dir)

    omni.usd.get_context().new_stage()
    carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)
    rep.orchestrator.set_capture_on_play(False)
    rep.settings.set_stage_up_axis("Z")
    rep.settings.set_stage_meters_per_unit(1.0)

    rep.functional.create.dome_light(intensity=900)
    plane = rep.functional.create.plane(position=(0, 0, 0), scale=(12, 12, 1), semantics={"class": "ground"})
    rep.functional.physics.apply_collider(plane)

    sphere = rep.functional.create.sphere(position=(-0.8, 0.0, 3.8), scale=0.8, semantics={"class": "sphere"})
    rep.functional.physics.apply_collider(sphere)
    rep.functional.physics.apply_rigid_body(sphere)

    cube = rep.functional.create.cube(position=(1.0, 0.7, 2.7), scale=0.7, semantics={"class": "cube"})
    rep.functional.physics.apply_collider(cube)
    rep.functional.physics.apply_rigid_body(cube)

    camera = rep.functional.create.camera(position=(6.0, 6.0, 3.5), look_at=(0.0, 0.0, 1.0))
    render_product = rep.create.render_product(camera, (args.width, args.height))

    writer = rep.writers.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach(render_product)

    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for frame_idx in range(args.frames):
        print(f"Capturing frame {frame_idx + 1}/{args.frames}")
        simulation_app.update()
        rep.orchestrator.step(delta_time=1.0 / args.fps, pause_timeline=False)
    timeline.pause()

    rep.orchestrator.wait_until_complete()
    writer.detach()
    render_product.destroy()

    frames = _find_rgb_frames(frame_dir)
    _write_video(frames, video_path, args.fps)
    return video_path


if __name__ == "__main__":
    try:
        video_path = generate_video()
        print(f"Video written to: {video_path}")
    finally:
        simulation_app.close()
