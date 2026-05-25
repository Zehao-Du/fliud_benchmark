#!/usr/bin/env python3
"""Test: does NvFlow fire/smoke render via omni.replicator when run under Xvfb?

Usage:
    xvfb-run -a -s "-screen 0 1920x1080x24" conda run -n env_isaaclab \
        python tools/test_nvflow_xvfb.py

Outputs frames to /tmp/nvflow_xvfb_test/
"""
from __future__ import annotations
import sys, pathlib

FIRE_USDA = pathlib.Path(
    "/root/miniconda3/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim"
    "/extscache/omni.flowusd-107.1.8+107.1.0.lx64.r.cp311.u353/data/presets/Fire/Fire.usda"
)
OUTPUT_DIR = pathlib.Path("/tmp/nvflow_xvfb_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_FRAMES = 20

print(f"[test] Fire.usda exists: {FIRE_USDA.exists()}", flush=True)

# --- Isaac Sim bootstrap ---
from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "width": 1280, "height": 720})
print("[test] SimulationApp started", flush=True)

import carb.settings
import omni.usd
import omni.kit.app
import omni.timeline

# Enable Flow extension
ext_mgr = omni.kit.app.get_app().get_extension_manager()
ext_mgr.set_extension_enabled_immediate("omni.flowusd", True)
print(f"[test] omni.flowusd enabled: {ext_mgr.is_extension_enabled('omni.flowusd')}", flush=True)

# RTX settings
cfg = carb.settings.get_settings()
cfg.set("/rtx/rendermode", "RaytracedLighting")  # Real-Time first
cfg.set("/rtx/flow/enabled", True)
cfg.set("/rtx/flow/pathTracingEnabled", False)
cfg.set("/rtx/flow/rayTracedTranslucencyEnabled", True)

# Open stage and reference Fire.usda
stage = omni.usd.get_context().get_stage()
from pxr import UsdGeom, Gf, UsdLux, Sdf

# Simple camera
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
cam_prim = stage.DefinePrim("/World/Camera", "Camera")
from pxr import UsdGeom as UG
xf = UG.Xformable(cam_prim)
xf.AddTranslateOp().Set(Gf.Vec3d(0, -300, 150))
xf.AddRotateXYZOp().Set(Gf.Vec3f(70, 0, 0))

# Dome light
dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.CreateIntensityAttr(1000.0)

# Reference Fire.usda
fire_xf = stage.DefinePrim("/World/Fire", "Xform")
fire_xf.GetReferences().AddReference(str(FIRE_USDA))
UG.Xformable(fire_xf).AddTranslateOp().Set(Gf.Vec3d(0, 0, 0))

app.update()

# Check prims
for path in ["/World/Fire/flowEmitterSphere", "/World/Fire/flowSimulate", "/World/Fire/flowRender"]:
    p = stage.GetPrimAtPath(path)
    print(f"[test] {path}: type={p.GetTypeName() if p.IsValid() else 'MISSING'}", flush=True)

# Play timeline
timeline = omni.timeline.get_timeline_interface()
timeline.play()

# Set up replicator capture
import omni.replicator.core as rep

render_product = rep.create.render_product("/World/Camera", (1280, 720))
writer = rep.WriterRegistry.get("BasicWriter")
writer.initialize(output_dir=str(OUTPUT_DIR), rgb=True)
writer.attach(render_product)

print(f"[test] Capturing {NUM_FRAMES} frames...", flush=True)
for i in range(NUM_FRAMES):
    app.update()
    rep.orchestrator.step(delta_time=1.0 / 60.0, pause_timeline=False)
    print(f"[test] frame {i+1}/{NUM_FRAMES}", flush=True)

rep.orchestrator.wait_until_complete()
writer.detach()
render_product.destroy()

frames = sorted(OUTPUT_DIR.glob("rgb_*.png"))
print(f"[test] Frames written: {len(frames)}", flush=True)

if frames:
    import subprocess
    result = subprocess.run(
        ["python3", "-c",
         f"import struct; data=open('{frames[0]}','rb').read();"
         "w=struct.unpack('>I',data[16:20])[0]; h=struct.unpack('>I',data[20:24])[0];"
         "print(f'size={w}x{h}, bytes={len(data)}')"],
        capture_output=True, text=True
    )
    print(f"[test] First frame: {result.stdout.strip()}", flush=True)

    # Check if frame has non-trivial content (not all one color)
    try:
        import struct
        data = frames[0].read_bytes()
        # Sample a few pixels from the center region (PNG decode via imageio if available)
        try:
            import imageio.v2 as imageio
            import numpy as np
            img = imageio.imread(str(frames[-1]))  # last frame should have most fire
            mean = img.mean()
            std = img.std()
            print(f"[test] Last frame pixel stats: mean={mean:.1f} std={std:.1f} (non-trivial if std>5)", flush=True)
            non_bg = (img > 20).sum()
            print(f"[test] Pixels > 20 brightness: {non_bg} / {img.size}", flush=True)
        except ImportError:
            print("[test] imageio not available for pixel analysis", flush=True)
    except Exception as e:
        print(f"[test] Analysis error: {e}", flush=True)

app.close()
print("[test] Done.", flush=True)
