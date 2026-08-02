"""
Interactive viewer for pre-rendered black-hole orbit frames.

Generate frames first, e.g.:
  python kerr_raytracer.py --prerender 180 --res 2560 1440 --ssaa 2 --spin 0.95 --jet

Then:
  python orbit_viewer.py                # newest folder under ./prerender/
  python orbit_viewer.py prerender/orbit_20260610_140000

Controls:
  mouse drag (LMB) : rotate around the black hole (scrubs the orbit)
  Space            : auto-rotate on/off       A / D : step one frame
  P                : save current frame copy to ./renders/
  ESC              : quit

All frames are preloaded into RAM as uint8 (a 180-frame 1440p set is ~2 GB).
"""

import argparse
import glob
import json
import math
import os
import shutil
import time

import numpy as np
import taichi as ti


def find_folder(arg):
    if arg:
        return arg
    cands = sorted(glob.glob(os.path.join("prerender", "orbit_*")), key=os.path.getmtime)
    if not cands:
        raise SystemExit("no prerender/orbit_* folder found -- run kerr_raytracer.py --prerender first")
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=None)
    args = ap.parse_args()
    folder = find_folder(args.folder)

    with open(os.path.join(folder, "meta.json")) as fh:
        meta = json.load(fh)
    files = sorted(glob.glob(os.path.join(folder, "frame_*.jpg")))
    if not files:
        raise SystemExit(f"no frames in {folder}")

    from PIL import Image
    print(f"loading {len(files)} frames from {folder} ...")
    stack = []
    for i, f in enumerate(files):
        a = np.asarray(Image.open(f).convert("RGB"))      # (H, W, 3) top-down
        stack.append(a[::-1].transpose(1, 0, 2).copy())   # -> (W, H, 3) bottom-up
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(files)}")
    W, H = stack[0].shape[0], stack[0].shape[1]
    n = len(stack)
    print(f"{n} frames, {W}x{H}, ~{sum(s.nbytes for s in stack) / 1e9:.1f} GB RAM")

    ti.init(arch=ti.gpu)
    field = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))
    window = ti.ui.Window(
        f"轨道查看器 | a={meta.get('spin')} 倾角={meta.get('inc')}° | 拖动=旋转 空格=自动 P=截图",
        (W, H), vsync=True)
    canvas = window.get_canvas()

    idx, shown = 0.0, -1
    playing = False
    last_cursor = None
    t_prev = time.time()

    while window.running:
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                playing = not playing
            elif e.key == "a":
                idx -= 1.0
            elif e.key == "d":
                idx += 1.0
            elif e.key == "p":
                os.makedirs("renders", exist_ok=True)
                dst = os.path.join("renders", f"orbit_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                shutil.copy(files[int(idx) % n], dst)
                print("saved", dst)

        if window.is_pressed(ti.ui.LMB):
            cur = window.get_cursor_pos()
            if last_cursor is not None:
                idx -= (cur[0] - last_cursor[0]) * n * 0.6
            last_cursor = cur
        else:
            last_cursor = None

        now = time.time()
        if playing:
            idx += (now - t_prev) * n / 12.0      # full orbit in ~12 s
        t_prev = now

        i = int(round(idx)) % n
        if i != shown:
            field.from_numpy(stack[i].astype(np.float32) / 255.0)
            shown = i
        canvas.set_image(field)
        window.show()


if __name__ == "__main__":
    main()
