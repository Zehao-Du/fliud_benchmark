#!/usr/bin/env python3
"""Cinematic render of franka_steam_tasks — NvFlow volumetric steam.

Key differences vs the broken sphere-prim version:
  - blob_count=0 in config → no sphere prims created at all
  - bbq_smoke.usda loaded into the scene stage at the steam origin (same as grill)
  - /rtx/flow/enabled=True, RaytracedLighting (same as grill cinematic)
  - hero_front_diag camera showing full robot + scene (same composition as grill)
  - Lighting kept brighter (no harsh dimming) so robot is visible

Usage (run under xvfb-run — NvFlow requires display):
    xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \\
      conda run -n env_isaaclab python tools/render_franka_steam_tasks_cinematic.py \\
        --task spray_chamber_tray --frames 360 --spp 16

    xvfb-run -a -s "-screen 0 1920x1080x24 +extension GLX" \\
      conda run -n env_isaaclab python tools/render_franka_steam_tasks_cinematic.py \\
        --task steam_pot_place --frames 360 --spp 16
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Hero camera — same composition style as grill cinematic "hero_front_diag":
# positioned far-left-front, showing full robot + workspace + rising steam
CAMERA_PRESETS: Dict[str, Dict] = {
    "spray_chamber_tray": {
        "position": (1.38, -0.88, 0.90),
        "look_at":  (0.60, 0.02, 0.18),
        "focal_length": 26.0,
        "name": "SteamCam_spray",
    },
    "steam_pot_place": {
        "position": (1.36, -0.84, 0.88),
        "look_at":  (0.62, 0.06, 0.16),
        "focal_length": 26.0,
        "name": "SteamCam_pot",
    },
    "steam_lid_open_close": {
        "position": (1.36, -0.84, 0.90),
        "look_at":  (0.66, 0.10, 0.18),
        "focal_length": 26.0,
        "name": "SteamCam_lid",
    },
    "steam_valve_close": {
        "position": (1.30, -0.80, 0.86),
        "look_at":  (0.60, 0.00, 0.18),
        "focal_length": 28.0,
        "name": "SteamCam_valve",
    },
}


def _load_steam_module():
    path = REPO_ROOT / "source/standalone_examples/api/isaacsim.robot.manipulators/franka/franka_steam_tasks.py"
    spec = importlib.util.spec_from_file_location("franka_steam_tasks_cin", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["franka_steam_tasks_cin"] = mod
    spec.loader.exec_module(mod)
    return mod


def _configure_rtx_for_nvflow() -> None:
    """Same RTX settings as render_franka_meat_on_grill_cinematic.py."""
    import carb.settings
    cfg = carb.settings.get_settings()
    cfg.set("/rtx/rendermode", "RaytracedLighting")
    cfg.set("/rtx/flow/enabled", True)
    cfg.set("/rtx/flow/rayTracedTranslucencyEnabled", True)
    cfg.set("/rtx/flow/rayTracedReflectionsEnabled", False)
    cfg.set("/rtx/flow/pathTracingEnabled", False)
    cfg.set("/omni/replicator/captureOnPlay", False)
    print("[render] RTX RaytracedLighting + NvFlow enabled")


def _soften_lights(stage) -> None:
    """Gentle light adjustment: dim dome just enough to make smoke visible,
    but keep scene bright enough that the robot is clearly visible."""
    try:
        from pxr import UsdGeom
        # Hide ground plane (same as grill)
        for gp_path in ["/World/defaultGroundPlane", "/World/groundPlane"]:
            gp = stage.GetPrimAtPath(gp_path)
            if gp.IsValid():
                UsdGeom.Imageable(gp).MakeInvisible()
                print(f"[render] Hidden ground plane {gp_path}")
                break
        for prim in stage.Traverse():
            attr = prim.GetAttribute("inputs:intensity")
            if not attr.IsValid():
                continue
            v = attr.Get()
            if v is None:
                continue
            tp = prim.GetTypeName()
            if tp == "DomeLight":
                # Keep dome visible (300 max) so robot materials look good
                new_v = min(v, 300.0)
                attr.Set(new_v)
                print(f"[render] DomeLight: {v:.0f} → {new_v:.0f}")
            elif tp == "SphereLight" and "defaultGroundPlane" in str(prim.GetPath()):
                attr.Set(0.0)
    except Exception as exc:
        print(f"[render] Light adjust warning: {exc}")


def _add_nvflow_smoke(stage, steam_origin, gf) -> bool:
    """Load bbq_smoke.usda at steam_origin — identical to franka_meat_on_grill._try_create_flow_smoke."""
    try:
        import omni.kit.app
        import carb.settings

        carb.settings.get_settings().set("/rtx/flow/enabled", True)
        carb.settings.get_settings().set("/rtx/flow/rayTracedTranslucencyEnabled", True)

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        if not ext_mgr.is_extension_enabled("omni.flowusd"):
            ext_mgr.set_extension_enabled_immediate("omni.flowusd", True)
            for _ in range(3):
                omni.kit.app.get_app().update()
            print("[render] omni.flowusd enabled")
        else:
            print("[render] omni.flowusd already active")

        usda = REPO_ROOT / "tools" / "bbq_smoke.usda"
        if not usda.exists():
            print(f"[render] bbq_smoke.usda not found at {usda}")
            return False

        root_path = "/World/FlowSteam"
        root_prim = stage.DefinePrim(root_path, "Xform")
        root_prim.GetReferences().AddReference(str(usda))

        from pxr import UsdGeom as _G
        _G.Xformable(root_prim).AddTranslateOp().Set(
            gf.Vec3d(steam_origin[0], steam_origin[1], steam_origin[2])
        )
        print(f"[render] NvFlow smoke placed at {steam_origin}")

        omni.kit.app.get_app().update()

        emitter = stage.GetPrimAtPath(root_path + "/flowEmitterSphere")
        if emitter.IsValid():
            print(f"[render] NvFlow emitter type: {emitter.GetTypeName()}")
        else:
            print(f"[render] WARNING: NvFlow emitter not found (omni.flowusd may be inactive)")
            return False

        # Boost smoke visibility (same as grill cinematic)
        cloud = stage.GetPrimAtPath(root_path + "/flowRender/rayMarch/cloud")
        if cloud.IsValid():
            cloud.GetAttribute("densityMultiplier").Set(3.5)
        cmap = stage.GetPrimAtPath(root_path + "/flowOffscreen/colormap")
        if cmap.IsValid():
            cmap.GetAttribute("colorScale").Set(6.0)
        print("[render] NvFlow density boosted")
        return True
    except Exception as exc:
        print(f"[render] NvFlow setup error: {exc}")
        return False


def _encode_video(frame_paths, out_path: Path, fps: int = 24) -> bool:
    try:
        import imageio.v2 as iio
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with iio.get_writer(str(out_path), fps=fps, codec="libx264",
                            quality=9, format="FFMPEG") as w:
            for p in frame_paths:
                w.append_data(iio.imread(str(p)))
        sz = out_path.stat().st_size
        print(f"\n[encode] {len(frame_paths)} frames → {out_path}  ({sz//1024} KB)")
        return True
    except Exception as exc:
        print(f"\n[encode] ERROR: {exc}")
        return False


def render_task(args: argparse.Namespace) -> int:
    mod = _load_steam_module()
    cam_cfg = CAMERA_PRESETS.get(args.task, CAMERA_PRESETS["spray_chamber_tray"])

    print(f"[render] Task:       {args.task}")
    print(f"[render] Resolution: {args.width}×{args.height}  spp={args.spp}  frames={args.frames}")
    print(f"[render] Camera:     {cam_cfg['position']} → {cam_cfg['look_at']}")

    config = mod.get_preset_config(args.task, args.preset)
    config["seed"] = args.seed
    config["headless"] = False          # NvFlow requires display
    config["video"]["enabled"] = False

    # ── KEY FIX: zero sphere prims → no ugly balls in the render ──
    config["scene"]["steam"]["blob_count"] = 0

    out_dir = Path(args.output_dir) / args.task
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    env = mod.FrankaSteamTaskEnv(
        task_type=args.task, config=config, preset=args.preset, backend="isaac"
    )
    env.create_scene()
    env.reset_scene(args.seed)

    # ── omni/pxr imports only valid after SimulationApp is running ──
    import omni.replicator.core as rep
    import omni.kit.app
    import omni.usd
    import omni.timeline
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()

    # ── Configure render + add NvFlow smoke + adjust lights ──
    _configure_rtx_for_nvflow()
    steam_origin = config["scene"]["steam"]["origin"]
    _add_nvflow_smoke(stage, steam_origin, Gf)
    _soften_lights(stage)

    # ── Pin replicator output root so frame_dir is absolute ──
    try:
        import carb
        carb.settings.get_settings().set("/omni/replicator/backends/disk/root_dir", "/")
    except Exception:
        pass

    # ── Create camera (same pattern as grill cinematic) ──
    cam_name = cam_cfg["name"]
    rep.create.camera(
        position=cam_cfg["position"],
        look_at=cam_cfg["look_at"],
        focal_length=cam_cfg["focal_length"],
        name=cam_name,
    )
    omni.kit.app.get_app().update()

    cam_path = ""
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Camera" and cam_name in str(prim.GetPath()):
            cam_path = str(prim.GetPath())
            break
    if not cam_path:
        cam_path = f"/Replicator/{cam_name}_Xform/{cam_name}"
    print(f"[render] Camera prim: {cam_path}")

    render_product = rep.create.render_product(cam_path, (args.width, args.height))
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=str(frame_dir), rgb=True)
    writer.attach(render_product)

    omni.timeline.get_timeline_interface().play()

    # ── Render loop (identical to grill cinematic) ──
    captured = 0
    hold_count = 0
    try:
        for frame_idx in range(args.frames):
            obs = env.get_obs()
            phase = obs["task"]["phase"]
            steam_str = obs.get("steam", {}).get("strength", 0.0)
            success = obs["task"].get("success", False)

            if phase == "hold_done":
                hold_count += 1
                if hold_count >= 30:
                    break

            print(
                f"\r[render] frame {frame_idx+1:4d}/{args.frames}"
                f"  phase={phase:22s}  steam={steam_str:.3f}  success={int(success)}",
                end="", flush=True,
            )

            env.step_sim(1)
            omni.kit.app.get_app().update()
            rep.orchestrator.step(delta_time=1.0 / 60.0, pause_timeline=False)
            captured += 1

        print()
        rep.orchestrator.wait_until_complete()
    finally:
        writer.detach()
        render_product.destroy()

    env.close()

    # ── Find frames (handle replicator root_dir prefix) ──
    frame_paths = sorted(frame_dir.glob("rgb_*.png"))
    if not frame_paths:
        for candidate in [
            Path("/outputs") / str(frame_dir).lstrip("/"),
            Path("/root/omni.replicator_out") / str(frame_dir).lstrip("/"),
        ]:
            frame_paths = sorted(candidate.glob("rgb_*.png"))
            if frame_paths:
                print(f"[render] Frames at: {candidate}")
                break

    print(f"[render] {len(frame_paths)} frames captured")
    if not frame_paths:
        print("[render] ERROR: No frames produced.")
        return 1

    video_path = out_dir / "cinematic.mp4"
    ok = _encode_video(frame_paths, video_path, fps=args.fps)
    if ok:
        print(f"[render] ✓ {video_path}")
    return 0 if ok else 1


def _parse_args() -> argparse.Namespace:
    from benchmark.tasks import STEAM_TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[t.task_id for t in STEAM_TASKS],
                        default="spray_chamber_tray")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frames", type=int, default=360)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/benchmark_videos_cinematic/steam"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(render_task(_parse_args()))
