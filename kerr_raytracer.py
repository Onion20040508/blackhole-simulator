"""
Real-time Kerr / Schwarzschild black hole sandbox (GPU, Taichi).
渲染窗口 (Taichi/CUDA) + 中文控制面板 (tkinter)。

Physics
-------
* Geometric units G = c = M = 1.  Spin parameter a in [0, 1); a = 0 is Schwarzschild.
* Metric in Kerr-Schild Cartesian form (horizon-penetrating, no coordinate
  singularity at the horizon, smooth a -> 0 limit):

      g_{mu nu} = eta_{mu nu} + 2 H l_mu l_nu,
      H  = r^3 / (r^4 + a^2 z^2),
      l_mu = (1, (r x + a y)/(r^2+a^2), (r y - a x)/(r^2+a^2), z/r),

  where r(x,y,z) is the Boyer-Lindquist radius, the positive root of
      r^4 - (x^2+y^2+z^2 - a^2) r^2 - a^2 z^2 = 0.
  (See e.g. Visser, "The Kerr spacetime: a brief introduction", arXiv:0706.0622.)

* Null geodesics integrated backwards from the camera with the super-Hamiltonian
      W = (1/2) g^{mu nu} p_mu p_nu,   g^{mu nu} = eta^{mu nu} - 2 H l^mu l^nu,
  Hamilton's equations, RK4, analytic gradient (p_t conserved).
  The camera may fly through the horizon (the coordinates are regular there);
  rays then terminate near the singularity instead of at the horizon.

* Stars: swarms of particles on full 3D timelike Kerr geodesics (KS Cartesian,
  midpoint method, coordinate-time stepping).  Launch them with the right mouse
  button (aim-preview = one exactly integrated test geodesic).  Tidal stretching
  into debris streams is automatic; an optional dissipation term relaxes
  particles toward the local mean flow (deposited each frame into density +
  velocity grids), standing in for the stream-stream collisions that circularize
  TDE debris into an accretion disk.  Emission: blackbody, exact redshift factor
  g = (p.u_obs)/(p.u_em) from the deposited velocity field, beaming g^{4 beta}.
  Particles are deleted when they cross the horizon or escape unbound.

* Reflective metal balls: analytic spheres on geodesic orbits; rays mirror on
  the surface (static-mirror approximation) and keep integrating, so the ball
  images the lensed scene around it.

Rendering: jittered HDR -> temporal accumulation while the view is static ->
bloom + starburst -> ACES + gamma -> upscale.  Two pre-warmed kernel variants
(lean / extras) -- toggling features never triggers a JIT pause.

Controls (render window)
------------------------
  Mouse drag (LMB) : look around      W/A/S/D : move   Q/E : down/up
  Shift : faster                      RMB hold : aim, release : launch
  P : save PNG                        ESC : quit

CLI examples
------------
  python kerr_raytracer.py
  python kerr_raytracer.py --still --res 3840 2160 --ssaa 2 --spin 0.95
  python kerr_raytracer.py --still --launch --ball --t-anim 10
  python kerr_raytracer.py --anim 240 --orbit 90
"""

import argparse
import ctypes
import glob
import math
import os
import sys
import time

import numpy as np
import taichi as ti

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

if getattr(sys, "frozen", False):
    # Taichi's JIT needs the kernel SOURCE at runtime, but PyInstaller ships
    # only bytecode. The .py is bundled as a data file; feed it to linecache
    # under the exact co_filename key so inspect.getsourcelines works.
    import linecache
    for _src in (os.path.join(getattr(sys, "_MEIPASS", ""), "kerr_raytracer.py"),
                 os.path.join(os.path.dirname(sys.executable), "_internal", "kerr_raytracer.py"),
                 os.path.join(os.path.dirname(sys.executable), "kerr_raytracer.py")):
        if _src and os.path.exists(_src):
            with open(_src, encoding="utf-8") as _fh:
                _lines = _fh.readlines()
            linecache.cache["kerr_raytracer.py"] = (len(_lines), None, _lines, "kerr_raytracer.py")
            break

if "--_probe-cuda" in sys.argv:
    # Subprocess self-test (see _cuda_usable below): compile + run a tiny CUDA
    # kernel. If the driver's JIT state is broken this crashes natively --
    # in THIS throwaway process, not in the app.
    ti.init(arch=ti.cuda, log_level="error")
    _pf = ti.field(ti.f32, shape=64)

    @ti.kernel
    def _pk():
        for i in _pf:
            _pf[i] = ti.sin(i * 0.5)

    _pk()
    ti.sync()
    print("CUDA_PROBE_OK")
    sys.exit(0)


def _cuda_usable() -> bool:
    """Probe CUDA in a sacrificial subprocess. A corrupted driver JIT state
    (e.g. after many hard-crashed CUDA processes; fixed by a reboot) kills the
    probe with 0xC0000005 instead of killing the app."""
    if sys.platform == "darwin":
        return False
    try:
        import subprocess
        cmd = [sys.executable]
        if not getattr(sys, "frozen", False):
            cmd.append(os.path.abspath(__file__))
        cmd.append("--_probe-cuda")
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        return r.returncode == 0 and b"CUDA_PROBE_OK" in r.stdout
    except Exception:
        return False


# ----------------------------------------------------------------------------
# CLI / init
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Kerr black hole GPU sandbox")
parser.add_argument("--res", type=int, nargs=2, default=None, metavar=("W", "H"))
parser.add_argument("--spin", type=float, default=0.85, help="spin a in [0, 0.999]")
parser.add_argument("--inc", type=float, default=80.0, help="camera inclination from pole, degrees")
parser.add_argument("--dist", type=float, default=28.0, help="camera distance in M")
parser.add_argument("--steps", type=int, default=420, help="max integration steps per ray")
parser.add_argument("--still", action="store_true", help="render one high-quality frame and exit")
parser.add_argument("--ssaa", type=int, default=1, help="supersampling factor for --still (1..3)")
parser.add_argument("--anim", type=int, default=0, help="render N animation frames and exit")
parser.add_argument("--prerender", type=int, default=0,
                    help="pre-render N orbit frames at still quality for orbit_viewer.py")
parser.add_argument("--orbit", type=float, default=0.0, help="camera azimuth sweep (deg) for --anim")
parser.add_argument("--rotrate", type=float, default=4.0, help="sim time advance per frame for --anim")
parser.add_argument("--mb", type=int, default=1, help="motion-blur sub-frames per frame")
parser.add_argument("--star", action="store_true", help="spawn a star (TDE) before offline rendering")
parser.add_argument("--srp", type=float, default=4.5, help="star orbit pericenter radius (M)")
parser.add_argument("--launch", action="store_true", help="launch a star along the view axis (offline)")
parser.add_argument("--ball", action="store_true", help="launched object is a reflective metal ball")
parser.add_argument("--disk-seed", action="store_true",
                    help="auto-launch one star on a disk-forming orbit (offline)")
parser.add_argument("--np-star-max", type=int, default=8_000_000,
                    help="star particle pool size (VRAM ~28 B/particle)")
parser.add_argument("--star-extent", type=float, default=56.0,
                    help="star grid half-extent in M (particles render within this box)")
parser.add_argument("--mode", type=int, default=0, help="background 0=sky 1=grid 2=lattice (offline)")
parser.add_argument("--t-anim", type=float, default=0.0, help="sim time advance for --still")
parser.add_argument("--beam", type=float, default=0.5,
                    help="Doppler beaming strength: 1 = physical g^4, 0 = off")
parser.add_argument("--sky", type=str, default=None, help="equirectangular sky image (overrides assets/)")
parser.add_argument("--ui-scale", type=float, default=1.0, help="control-panel font scale")
parser.add_argument("--out", type=str, default="renders")
parser.add_argument("--cpu", action="store_true", help="force CPU backend (debug)")
args = parser.parse_args()

if args.cpu:
    ti.init(arch=ti.cpu, default_fp=ti.f32)
elif _cuda_usable():
    ti.init(arch=ti.cuda, default_fp=ti.f32)
else:
    print("CUDA 自检未通过（驱动状态异常时重启电脑通常可恢复）-> 回退 Vulkan 后端")
    try:
        ti.init(arch=ti.vulkan, default_fp=ti.f32)
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)

OFFLINE = args.still or args.anim > 0 or args.prerender > 0

if args.res is None:
    try:
        sw = ctypes.windll.user32.GetSystemMetrics(0)
        sh = ctypes.windll.user32.GetSystemMetrics(1)
        W, H = int(sw * 0.86) // 2 * 2, int(sh * 0.86) // 2 * 2
    except Exception:
        W, H = 1920, 1080
else:
    W, H = args.res

SS = max(args.ssaa, 1) if (args.still or args.prerender > 0) else 1
RW, RH = W * SS, H * SS

hdr = ti.Vector.field(3, dtype=ti.f32, shape=(RW, RH))     # current frame, linear
acc = ti.Vector.field(3, dtype=ti.f32, shape=(RW, RH))     # temporal accumulation
img = ti.Vector.field(3, dtype=ti.f32, shape=(W, H))       # display, tonemapped
BW, BH = max(W // 2, 1), max(H // 2, 1)
bloom0 = ti.Vector.field(3, dtype=ti.f32, shape=(BW, BH))
bloom1 = ti.Vector.field(3, dtype=ti.f32, shape=(BW, BH))

vec3 = ti.math.vec3

# star (TDE) particles: full 3D Kerr geodesics in KS Cartesian coordinates,
# state (pos, covariant p_i, conserved p_t); p_t = 0 marks a dead particle
NP_STAR_MAX = max(args.np_star_max, 100_000)
spx = ti.Vector.field(3, ti.f32, shape=NP_STAR_MAX)
spp = ti.Vector.field(3, ti.f32, shape=NP_STAR_MAX)
spt = ti.field(ti.f32, shape=NP_STAR_MAX)
sT = ti.field(ti.f32, shape=NP_STAR_MAX)      # per-particle temperature [K]
sbat = ti.field(ti.i32, shape=NP_STAR_MAX)    # star id for self-gravity, -1 = free
SB_MAX = 16                                   # simultaneously bound stars
sb_sum = ti.Vector.field(3, ti.f32, shape=SB_MAX)
sb_cnt = ti.field(ti.i32, shape=SB_MAX)
sb_com = ti.Vector.field(3, ti.f32, shape=SB_MAX)
sb_mu = ti.field(ti.f32, shape=SB_MAX)        # star mass / M_BH
sb_rad = ti.field(ti.f32, shape=SB_MAX)       # Plummer softening ~ star radius
SGN = 320
SG_MAX = max(args.star_extent, 20.0)          # hard cap on the grid half-extent
sgf = ti.field(ti.f32, shape=())              # CURRENT half-extent: auto-fits the
bmax = ti.field(ti.f32, shape=())             # particle bounding radius each frame,
#                                               so a compact scene gets ~3x finer
#                                               voxels (kills the staircase look)
sgrid = ti.field(ti.f32, shape=(SGN, SGN, SGN))
svgrid = ti.Vector.field(3, ti.f32, shape=(SGN, SGN, SGN))
stgrid = ti.field(ti.f32, shape=(SGN, SGN, SGN))   # mass-weighted temperature
# occupancy mip for empty-space skipping: rays take big steps through empty
# cells and only refine where debris actually is (dilated -> safe to skip)
OCN = 64
OC_BLK = SGN // OCN
occ0 = ti.field(ti.i32, shape=(OCN, OCN, OCN))
occ = ti.field(ti.i32, shape=(OCN, OCN, OCN))

TRAJ_N = 256                                  # aim-preview trajectory samples
AIM_DOTS = 64
traj = ti.Vector.field(3, ti.f32, shape=TRAJ_N)
traj_len = ti.field(ti.i32, shape=())
traj_fate = ti.field(ti.i32, shape=())        # 0 plunge / 1 escape / 2 bound
traj_rmin = ti.field(ti.f32, shape=())
aim_line = ti.Vector.field(2, ti.f32, shape=2 * TRAJ_N)
aim_lcol = ti.Vector.field(3, ti.f32, shape=2 * TRAJ_N)
aim_dot = ti.Vector.field(2, ti.f32, shape=AIM_DOTS)
aim_dcol = ti.Vector.field(3, ti.f32, shape=AIM_DOTS)

BALL_MAX = 4                                  # reflective metal balls
ball_dat = ti.Vector.field(4, ti.f32, shape=BALL_MAX)   # (x, y, z, radius)
ball_vel = ti.Vector.field(3, ti.f32, shape=BALL_MAX)   # coordinate velocity

# ----------------------------------------------------------------------------
# sky texture (equirectangular), optional
# ----------------------------------------------------------------------------
def load_sky():
    # the procedural starfield is the default look; an equirectangular photo
    # sky is loaded ONLY when explicitly requested with --sky <file>
    cands = [args.sky] if args.sky else []
    for path in cands:
        if path is None or not os.path.isfile(path):
            continue
        if not path.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")):
            continue
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(path).convert("RGB")
            if im.width > 4096:
                im = im.resize((4096, im.height * 4096 // im.width), Image.LANCZOS)
            arr = (np.asarray(im, dtype=np.float32) / 255.0) ** 2.2   # sRGB -> linear
            print(f"sky texture: {os.path.basename(path)} ({im.width}x{im.height})")
            return arr.transpose(1, 0, 2).copy()                       # -> (w, h, 3)
        except Exception as e:
            print(f"sky load failed for {path}: {e}")
    return None

_sky = load_sky()
HAS_SKY = _sky is not None
if HAS_SKY:
    sky_tex = ti.Vector.field(3, dtype=ti.f32, shape=_sky.shape[:2])
    sky_tex.from_numpy(_sky)
    SKY_W, SKY_H = _sky.shape[0], _sky.shape[1]
else:
    sky_tex = ti.Vector.field(3, dtype=ti.f32, shape=(1, 1))
    SKY_W, SKY_H = 1, 1
del _sky

# ----------------------------------------------------------------------------
# Kerr-Schild geometry
# ----------------------------------------------------------------------------
@ti.func
def metric_Hl(pos: vec3, a: ti.f32):
    """H, l_i (spatial covariant Kerr-Schild quantities) and BL radius r."""
    rho2 = pos.dot(pos)
    b = rho2 - a * a
    disc = ti.sqrt(b * b + 4.0 * a * a * pos.z * pos.z)
    r2 = 0.5 * (b + disc)
    r = ti.sqrt(ti.max(r2, 1e-12))
    f = r2 * r2 + a * a * pos.z * pos.z
    Hh = r2 * r / ti.max(f, 1e-12)
    c = 1.0 / (r2 + a * a)
    l = vec3((r * pos.x + a * pos.y) * c, (r * pos.y - a * pos.x) * c, pos.z / r)
    return Hh, l, r


@ti.func
def geodesic_rhs(pos: vec3, pp: vec3, pt: ti.f32, a: ti.f32):
    """Hamilton's equations dx/dlam, dp/dlam for W = (1/2) g^{mu nu} p_mu p_nu,
    with the gradient of W computed analytically."""
    x, y, z = pos.x, pos.y, pos.z
    rho2 = pos.dot(pos)
    b = rho2 - a * a
    disc = ti.sqrt(b * b + 4.0 * a * a * z * z)
    r2 = 0.5 * (b + disc)
    r = ti.sqrt(ti.max(r2, 1e-12))

    # grad r:  r_,i = (r^2 x_i + a^2 z delta_iz) / (r * disc)
    drv = (r2 * pos + vec3(0.0, 0.0, a * a * z)) / ti.max(r * disc, 1e-9)

    # H = r^3 / f,  f = r^4 + a^2 z^2
    f = r2 * r2 + a * a * z * z
    Hh = r2 * r / ti.max(f, 1e-12)
    dfv = 4.0 * r2 * r * drv + vec3(0.0, 0.0, 2.0 * a * a * z)
    dHv = (3.0 * r2 * f * drv - r2 * r * dfv) / ti.max(f * f, 1e-12)

    # l and its gradient
    c = 1.0 / (r2 + a * a)
    A = r * x + a * y
    B = r * y - a * x
    l = vec3(A * c, B * c, z / r)
    common = 2.0 * r * c * c
    dlx = c * (x * drv + vec3(r, a, 0.0)) - A * common * drv
    dly = c * (y * drv + vec3(-a, r, 0.0)) - B * common * drv
    dlz = vec3(0.0, 0.0, 1.0 / r) - (z / ti.max(r2, 1e-12)) * drv

    lp = -pt + l.dot(pp)                      # l^mu p_mu
    dpos = pp - 2.0 * Hh * lp * l             # dx^i/dlam
    grad_lp = pp.x * dlx + pp.y * dly + pp.z * dlz
    dpp = dHv * (lp * lp) + 2.0 * Hh * lp * grad_lp   # dp_i/dlam = -dW/dx^i
    return dpos, dpp, r


@ti.func
def solve_pt(pos: vec3, pp: vec3, a: ti.f32) -> ti.f32:
    """Past-directed root of the null condition g^{mu nu} p_mu p_nu = 0
    (backward ray tracing: dt/dlam < 0)."""
    Hh, l, r = metric_Hl(pos, a)
    bb = l.dot(pp)
    A2 = 1.0 + 2.0 * Hh
    disc = ti.sqrt(ti.max(4.0 * Hh * Hh * bb * bb + A2 * (pp.dot(pp) - 2.0 * Hh * bb * bb), 0.0))
    pt = (2.0 * Hh * bb + disc) / A2
    if -pt + 2.0 * Hh * (-pt + bb) > 0.0:      # require dt/dlam < 0
        pt = (2.0 * Hh * bb - disc) / A2
    return pt


@ti.func
def doppler_g_v(ps: vec3, cp: vec3, pt: ti.f32, e_obs: ti.f32, vv: vec3, a: ti.f32) -> ti.f32:
    """Redshift factor for an emitter with coordinate 3-velocity v^i,
    u = u^t (d_t + v^i d_i) normalized with the exact local KS metric."""
    Hh, l, _ = metric_Hl(ps, a)
    lv = l.dot(vv)
    nrm = -((-1.0 + 2.0 * Hh) + 4.0 * Hh * lv + vv.dot(vv) + 2.0 * Hh * lv * lv)
    g = 1.0
    if nrm > 1e-5:
        ut = 1.0 / ti.sqrt(nrm)
        e_em = ti.abs(ut * (pt + vv.dot(cp)))
        g = e_obs / ti.max(e_em, 1e-9)
    return g


# ----------------------------------------------------------------------------
# color / background helpers
# ----------------------------------------------------------------------------
@ti.func
def hash3(p: vec3) -> vec3:
    q = vec3(p.dot(vec3(127.1, 311.7, 74.7)),
             p.dot(vec3(269.5, 183.3, 246.1)),
             p.dot(vec3(113.5, 271.9, 124.6)))
    return ti.math.fract(ti.sin(q) * 43758.5453)


@ti.func
def hash1(p: vec3) -> ti.f32:
    return ti.math.fract(ti.sin(p.dot(vec3(127.1, 311.7, 74.7))) * 43758.5453)


@ti.func
def vnoise(p: vec3) -> ti.f32:
    i = ti.floor(p)
    f = p - i
    u = f * f * (3.0 - 2.0 * f)
    nx00 = ti.math.mix(hash1(i), hash1(i + vec3(1, 0, 0)), u.x)
    nx10 = ti.math.mix(hash1(i + vec3(0, 1, 0)), hash1(i + vec3(1, 1, 0)), u.x)
    nx01 = ti.math.mix(hash1(i + vec3(0, 0, 1)), hash1(i + vec3(1, 0, 1)), u.x)
    nx11 = ti.math.mix(hash1(i + vec3(0, 1, 1)), hash1(i + vec3(1, 1, 1)), u.x)
    return ti.math.mix(ti.math.mix(nx00, nx10, u.y), ti.math.mix(nx01, nx11, u.y), u.z)


@ti.func
def blackbody_rgb(T: ti.f32) -> vec3:
    """Approximate linear-RGB chromaticity of a blackbody at temperature T [K]."""
    t = ti.math.clamp(T, 1000.0, 40000.0) / 100.0
    r, g, b = 1.0, 1.0, 1.0
    if t > 66.0:
        r = 1.292936 * ti.pow(t - 60.0, -0.1332047)
        g = 1.129891 * ti.pow(t - 60.0, -0.0755148)
    else:
        g = (99.4708 * ti.log(t) - 161.1196) / 255.0
        if t < 19.0:
            b = 0.0
        else:
            b = (138.5177 * ti.log(t - 10.0) - 305.0448) / 255.0
    col = ti.math.clamp(vec3(r, g, b), 0.0, 1.0)
    return vec3(ti.pow(col.x, 2.2), ti.pow(col.y, 2.2), ti.pow(col.z, 2.2))  # sRGB -> linear


@ti.func
def sky_sample(d: vec3, gain: ti.f32) -> vec3:
    # fixed tilt so the galactic band crosses the frame diagonally
    cb, sb = 0.515, 0.857    # ~59 deg about x
    dd = vec3(d.x, cb * d.y - sb * d.z, sb * d.y + cb * d.z)
    u = (ti.atan2(dd.y, dd.x) / (2.0 * math.pi) + 0.5) * (SKY_W - 1)
    v = ti.acos(ti.math.clamp(dd.z, -1.0, 1.0)) / math.pi * (SKY_H - 1)
    i0 = int(ti.math.clamp(u, 0.0, SKY_W - 1.001))
    j0 = int(ti.math.clamp(v, 0.0, SKY_H - 1.001))
    fx, fy = u - i0, v - j0
    i1, j1 = ti.min(i0 + 1, SKY_W - 1), ti.min(j0 + 1, SKY_H - 1)
    c = ti.math.mix(ti.math.mix(sky_tex[i0, j0], sky_tex[i1, j0], fx),
                    ti.math.mix(sky_tex[i0, j1], sky_tex[i1, j1], fx), fy)
    return c * gain


@ti.func
def background(d: vec3, mode: ti.i32, sky_on: ti.i32, gain: ti.f32) -> vec3:
    col = vec3(0.0)
    if mode == 2:
        col = vec3(0.002, 0.002, 0.003)        # 3D lattice mode: dark far field
    elif mode == 1:
        # latitude/longitude celestial grid -- the lensing map
        ph = ti.atan2(d.y, d.x)
        th = ti.acos(ti.math.clamp(d.z, -1.0, 1.0))
        gu = ti.abs(ti.math.fract(ph * 18.0 / (2.0 * math.pi)) - 0.5)
        gv = ti.abs(ti.math.fract(th * 18.0 / math.pi) - 0.5)
        line = 1.0 - ti.math.smoothstep(0.0, 0.04, ti.min(gu, gv))
        base = vec3(0.012, 0.014, 0.022)
        tint = vec3(0.20, 0.55, 0.90)
        if d.z < 0.0:
            tint = vec3(0.95, 0.55, 0.18)
        col = base + line * tint * 0.8
    elif sky_on == 1:
        col = sky_sample(d, gain)
    else:
        # procedural starfield + faint galactic band
        S = 220.0
        cell = ti.floor(d * S)
        h = hash3(cell)
        if h.x < 0.006:
            sp = (cell + 0.15 + 0.7 * hash3(cell + 17.31)).normalized()
            ang = (sp - d).norm()
            mag = ti.pow(h.y, 6.0) * 14.0 + 0.25
            star = mag * ti.exp(-ang * ang * 4.0e5)
            col += star * blackbody_rgb(2600.0 + 12000.0 * h.z)
        band = ti.exp(-d.dot(vec3(0.21, 0.41, 0.89)) ** 2 * 28.0)
        col += band * vec3(0.045, 0.043, 0.060)
        col += vec3(0.004, 0.005, 0.008)
    return col


@ti.func
def aces(x: vec3) -> vec3:
    return ti.math.clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0)


# ----------------------------------------------------------------------------
# particle deposition / sampling
# ----------------------------------------------------------------------------
@ti.func
def splat(grid: ti.template(), gx: ti.f32, gy: ti.f32, gz: ti.f32, w: ti.f32,
          nx: ti.i32, ny: ti.i32, nz: ti.i32):
    if gx >= 0.0 and gy >= 0.0 and gz >= 0.0 and gx < nx - 1 and gy < ny - 1 and gz < nz - 1:
        i0, j0, k0 = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - i0, gy - j0, gz - k0
        ti.atomic_add(grid[i0, j0, k0], w * (1 - fx) * (1 - fy) * (1 - fz))
        ti.atomic_add(grid[i0 + 1, j0, k0], w * fx * (1 - fy) * (1 - fz))
        ti.atomic_add(grid[i0, j0 + 1, k0], w * (1 - fx) * fy * (1 - fz))
        ti.atomic_add(grid[i0 + 1, j0 + 1, k0], w * fx * fy * (1 - fz))
        ti.atomic_add(grid[i0, j0, k0 + 1], w * (1 - fx) * (1 - fy) * fz)
        ti.atomic_add(grid[i0 + 1, j0, k0 + 1], w * fx * (1 - fy) * fz)
        ti.atomic_add(grid[i0, j0 + 1, k0 + 1], w * (1 - fx) * fy * fz)
        ti.atomic_add(grid[i0 + 1, j0 + 1, k0 + 1], w * fx * fy * fz)


@ti.func
def splat_vec(grid: ti.template(), gx: ti.f32, gy: ti.f32, gz: ti.f32, w: vec3,
              nx: ti.i32, ny: ti.i32, nz: ti.i32):
    if gx >= 0.0 and gy >= 0.0 and gz >= 0.0 and gx < nx - 1 and gy < ny - 1 and gz < nz - 1:
        i0, j0, k0 = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - i0, gy - j0, gz - k0
        ti.atomic_add(grid[i0, j0, k0], w * ((1 - fx) * (1 - fy) * (1 - fz)))
        ti.atomic_add(grid[i0 + 1, j0, k0], w * (fx * (1 - fy) * (1 - fz)))
        ti.atomic_add(grid[i0, j0 + 1, k0], w * ((1 - fx) * fy * (1 - fz)))
        ti.atomic_add(grid[i0 + 1, j0 + 1, k0], w * (fx * fy * (1 - fz)))
        ti.atomic_add(grid[i0, j0, k0 + 1], w * ((1 - fx) * (1 - fy) * fz))
        ti.atomic_add(grid[i0 + 1, j0, k0 + 1], w * (fx * (1 - fy) * fz))
        ti.atomic_add(grid[i0, j0 + 1, k0 + 1], w * ((1 - fx) * fy * fz))
        ti.atomic_add(grid[i0 + 1, j0 + 1, k0 + 1], w * (fx * fy * fz))


@ti.func
def trilerp(grid: ti.template(), gx: ti.f32, gy: ti.f32, gz: ti.f32,
            nx: ti.i32, ny: ti.i32, nz: ti.i32) -> ti.f32:
    v = 0.0
    if gx >= 0.0 and gy >= 0.0 and gz >= 0.0 and gx < nx - 1 and gy < ny - 1 and gz < nz - 1:
        i0, j0, k0 = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - i0, gy - j0, gz - k0
        c00 = ti.math.mix(grid[i0, j0, k0], grid[i0 + 1, j0, k0], fx)
        c10 = ti.math.mix(grid[i0, j0 + 1, k0], grid[i0 + 1, j0 + 1, k0], fx)
        c01 = ti.math.mix(grid[i0, j0, k0 + 1], grid[i0 + 1, j0, k0 + 1], fx)
        c11 = ti.math.mix(grid[i0, j0 + 1, k0 + 1], grid[i0 + 1, j0 + 1, k0 + 1], fx)
        v = ti.math.mix(ti.math.mix(c00, c10, fy), ti.math.mix(c01, c11, fy), fz)
    return v


@ti.func
def trilerp_vec(grid: ti.template(), gx: ti.f32, gy: ti.f32, gz: ti.f32,
                nx: ti.i32, ny: ti.i32, nz: ti.i32) -> vec3:
    v = vec3(0.0)
    if gx >= 0.0 and gy >= 0.0 and gz >= 0.0 and gx < nx - 1 and gy < ny - 1 and gz < nz - 1:
        i0, j0, k0 = int(gx), int(gy), int(gz)
        fx, fy, fz = gx - i0, gy - j0, gz - k0
        c00 = ti.math.mix(grid[i0, j0, k0], grid[i0 + 1, j0, k0], fx)
        c10 = ti.math.mix(grid[i0, j0 + 1, k0], grid[i0 + 1, j0 + 1, k0], fx)
        c01 = ti.math.mix(grid[i0, j0, k0 + 1], grid[i0 + 1, j0, k0 + 1], fx)
        c11 = ti.math.mix(grid[i0, j0 + 1, k0 + 1], grid[i0 + 1, j0 + 1, k0 + 1], fx)
        v = ti.math.mix(ti.math.mix(c00, c10, fy), ti.math.mix(c01, c11, fy), fz)
    return v


@ti.func
def star_dens_sample(p: vec3) -> ti.f32:
    s = sgf[None]
    return trilerp(sgrid,
                   (p.x / (2 * s) + 0.5) * (SGN - 1),
                   (p.y / (2 * s) + 0.5) * (SGN - 1),
                   (p.z / (2 * s) + 0.5) * (SGN - 1), SGN, SGN, SGN)


@ti.func
def star_vel_sample(p: vec3) -> vec3:
    s = sgf[None]
    return trilerp_vec(svgrid,
                       (p.x / (2 * s) + 0.5) * (SGN - 1),
                       (p.y / (2 * s) + 0.5) * (SGN - 1),
                       (p.z / (2 * s) + 0.5) * (SGN - 1), SGN, SGN, SGN)


@ti.func
def star_temp_sample(p: vec3) -> ti.f32:
    s = sgf[None]
    return trilerp(stgrid,
                   (p.x / (2 * s) + 0.5) * (SGN - 1),
                   (p.y / (2 * s) + 0.5) * (SGN - 1),
                   (p.z / (2 * s) + 0.5) * (SGN - 1), SGN, SGN, SGN)


# ----------------------------------------------------------------------------
# star particle dynamics (full 3D timelike geodesics)
# ----------------------------------------------------------------------------
@ti.func
def vel_to_p(pos: vec3, v: vec3, a: ti.f32):
    """Future-directed unit-mass momentum (p_t, p_i) for coordinate velocity v^i."""
    Hh, l, r = metric_Hl(pos, a)
    lv = l.dot(v)
    nrm = -((-1.0 + 2.0 * Hh) + 4.0 * Hh * lv + v.dot(v) + 2.0 * Hh * lv * lv)
    ut = 1.0 / ti.sqrt(ti.max(nrm, 1e-5))
    lpv = 1.0 + lv                              # l_mu u^mu / u^t
    pt = ut * (-1.0 + 2.0 * Hh * lpv)
    pi = ut * (v + 2.0 * Hh * lpv * l)
    return pt, pi


@ti.func
def coord_vel(pos: vec3, pp: vec3, pt: ti.f32, a: ti.f32):
    """Coordinate 3-velocity v^i = (dx^i/dlam)/(dt/dlam) and BL radius."""
    Hh, l, r = metric_Hl(pos, a)
    lp = -pt + l.dot(pp)
    dtdl = ti.max(-pt + 2.0 * Hh * lp, 1e-5)
    dx = pp - 2.0 * Hh * lp * l
    return dx / dtdl, dtdl, r


@ti.kernel
def spawn_star(off: ti.i32, n: ti.i32, c: vec3, vb: vec3, rad: ti.f32,
               sigma: ti.f32, a: ti.f32, T0: ti.f32, batch: ti.i32):
    """Append n particles starting at slot off (multiple stars accumulate).
    batch >= 0 binds the particles by their own self-gravity."""
    for k in range(n):
        i = off + k
        dx = vec3(ti.random() + ti.random() + ti.random() - 1.5,
                  ti.random() + ti.random() + ti.random() - 1.5,
                  ti.random() + ti.random() + ti.random() - 1.5) * (rad * 0.62)
        v = vb + sigma * vec3(ti.random() - 0.5, ti.random() - 0.5, ti.random() - 0.5) * 2.0
        pt, pi = vel_to_p(c + dx, v, a)
        spx[i] = c + dx
        spp[i] = pi
        spt[i] = pt
        sT[i] = T0 * (0.9 + 0.2 * ti.random(ti.f32))
        sbat[i] = batch


@ti.kernel
def star_com(n: ti.i32):
    """Center of mass of each bound star (for the self-gravity force)."""
    for b in range(SB_MAX):
        sb_sum[b] = vec3(0.0)
        sb_cnt[b] = 0
    for i in range(n):
        b = sbat[i]
        if b >= 0 and spt[i] != 0.0:
            ti.atomic_add(sb_sum[b], spx[i])
            ti.atomic_add(sb_cnt[b], 1)
    for b in range(SB_MAX):
        if sb_cnt[b] > 0:
            sb_com[b] = sb_sum[b] / sb_cnt[b]


@ti.kernel
def update_star_particles(n: ti.i32, dt: ti.f32, a: ti.f32, gamma: ti.f32,
                          r_kill: ti.f32, nb: ti.i32):
    """Full 3D timelike geodesics (midpoint method, coordinate-time stepping).
    Optional dissipation relaxes each particle toward the local mean flow --
    a stand-in for the stream-stream collisions that circularize TDE debris.
    Deleted when crossing the horizon or escaping unbound."""
    for i in range(n):
        if spt[i] != 0.0:
            pos, pp, pt = spx[i], spp[i], spt[i]
            nsub = 1 + int(dt / 0.2)
            hh = dt / nsub
            for _s in range(nsub):
                if pt != 0.0:
                    # micro-stepping in the strong-field zone: the plunge gets
                    # stiff near the hole and coarse f32 steps break the mass
                    # shell, ejecting particles unphysically ("center jets")
                    _, _, rr = coord_vel(pos, pp, pt, a)
                    m = ti.min(1 + int(hh / ti.max(0.035 * rr, 0.004)), 64)
                    hm = hh / m
                    for _m in range(m):
                        if pt != 0.0:
                            k1x, k1p, r1 = geodesic_rhs(pos, pp, pt, a)
                            _, dtdl, _ = coord_vel(pos, pp, pt, a)
                            w = hm / dtdl          # affine step per coordinate time
                            k2x, k2p, r2 = geodesic_rhs(pos + 0.5 * w * k1x,
                                                        pp + 0.5 * w * k1p, pt, a)
                            pos += w * k2x
                            pp += w * k2p
                            if r2 < r_kill or pos.norm() > 2.5 * SG_MAX:
                                pt = 0.0           # near singularity / far gone
                            elif -pt >= 1.0 and pos.norm() > SG_MAX * 1.15:
                                pt = 0.0           # unbound + outside box: escaped
            if pt != 0.0:
                v, _, rv = coord_vel(pos, pp, pt, a)
                changed = 0
                # self-gravity: Newtonian pull toward the star's own center of
                # mass (Plummer-softened) -- holds the star together until the
                # BH tide wins at r_t ~ R (M_BH/M_star)^(1/3)
                b = sbat[i]
                if b >= 0 and sb_mu[b] > 0.0 and sb_cnt[b] > 50:
                    dvec = sb_com[b] - pos
                    r2s = dvec.dot(dvec) + sb_rad[b] * sb_rad[b]
                    v += sb_mu[b] * dvec / (r2s * ti.sqrt(r2s)) * dt
                    changed = 1
                if gamma > 0.0:
                    dloc = star_dens_sample(pos)
                    if dloc > 0.05:
                        vm = star_vel_sample(pos) / dloc
                        f = ti.min(gamma * dt * dloc, 0.5)
                        dv = f * (vm - v)
                        v += dv
                        changed = 1
                        # friction: dissipated relative kinetic energy heats
                        # the gas (this makes inner disks glow blue-white)
                        sT[i] += 4e5 * dv.dot(dv)
                # collision with mirror balls: elastic bounce off the surface
                # (in the ball's rest frame, restitution 0.55, impact heating)
                for b_ in range(nb):
                    cb = vec3(ball_dat[b_].x, ball_dat[b_].y, ball_dat[b_].z)
                    dvec2 = pos - cb
                    dist = dvec2.norm()
                    if dist < ball_dat[b_].w and dist > 1e-4:
                        nrm_ = dvec2 / dist
                        pos = cb + nrm_ * ball_dat[b_].w * 1.02
                        vrel = v - ball_vel[b_]
                        vn_ = vrel.dot(nrm_)
                        if vn_ < 0.0:
                            vrel -= 1.55 * vn_ * nrm_
                            v = ball_vel[b_] + vrel
                            sT[i] += 5e4 * vn_ * vn_   # impact sparks
                            changed = 1
                if rv < 6.0:
                    changed = 1                    # always re-project near hole
                vn = v.norm()
                if vn > 1.5:
                    pt = 0.0                       # superluminal garbage: delete
                elif changed == 1:
                    if vn > 0.985:
                        v = v * (0.985 / vn)
                    # back onto the timelike mass shell (numerical drift in
                    # g^{mu nu} p_mu p_nu = -1 caused fake ejection jets)
                    pt, pp = vel_to_p(pos, v, a)
            if pt != 0.0:
                # slow radiative cooling toward a 1500 K floor
                sT[i] = 1500.0 + (sT[i] - 1500.0) * ti.exp(-0.004 * dt)
            spx[i], spp[i], spt[i] = pos, pp, pt


@ti.kernel
def scatter_star(n: ti.i32, a: ti.f32, w: ti.f32):
    for i in range(n):
        if spt[i] != 0.0:
            pos = spx[i]
            ti.atomic_max(bmax[None], ti.max(ti.abs(pos.x),
                                             ti.max(ti.abs(pos.y), ti.abs(pos.z))))
            s = sgf[None]
            v, _, _ = coord_vel(pos, spp[i], spt[i], a)
            gx = (pos.x / (2 * s) + 0.5) * (SGN - 1)
            gy = (pos.y / (2 * s) + 0.5) * (SGN - 1)
            gz = (pos.z / (2 * s) + 0.5) * (SGN - 1)
            splat(sgrid, gx, gy, gz, w, SGN, SGN, SGN)
            splat_vec(svgrid, gx, gy, gz, w * v, SGN, SGN, SGN)
            splat(stgrid, gx, gy, gz, w * sT[i], SGN, SGN, SGN)


@ti.kernel
def build_occupancy():
    """Two-pass occupancy mip of sgrid: block max-pool, then 3^3 dilation, so
    an empty `occ` cell guarantees >= 1 occ-cell of truly empty space around."""
    for i, j, k in occ0:
        m = 0
        for ii in range(i * OC_BLK, (i + 1) * OC_BLK):
            for jj in range(j * OC_BLK, (j + 1) * OC_BLK):
                for kk in range(k * OC_BLK, (k + 1) * OC_BLK):
                    if sgrid[ii, jj, kk] > 1e-4:
                        m = 1
        occ0[i, j, k] = m
    for i, j, k in occ:
        m = 0
        for di in range(ti.max(i - 1, 0), ti.min(i + 2, OCN)):
            for dj in range(ti.max(j - 1, 0), ti.min(j + 2, OCN)):
                for dk in range(ti.max(k - 1, 0), ti.min(k + 2, OCN)):
                    if occ0[di, dj, dk] == 1:
                        m = 1
        occ[i, j, k] = m


@ti.kernel
def predict_traj(c: vec3, v: vec3, a: ti.f32, r_kill: ti.f32):
    """Single test-particle geodesic from the launch state -> aim-preview line.
    Also classifies the fate: 0 = plunges into the hole, 1 = unbound/escapes,
    2 = bound orbit (the disk-forming kind)."""
    for _one in range(1):
        pt, pp = vel_to_p(c, v, a)
        E = -pt
        pos = c
        idx = 0
        acc_ = 1e9                               # force a sample at the start
        rmin = 1e9
        fate = 2
        if E >= 1.0:
            fate = 1
        for s in range(6000):
            k1x, k1p, r1 = geodesic_rhs(pos, pp, pt, a)
            rmin = ti.min(rmin, r1)
            if r1 < r_kill:
                fate = 0
                break
            if pos.norm() > 90.0 or idx >= TRAJ_N:
                break
            h = ti.min(0.06 / (k1p.norm() + 1e-4), 0.3 * (r1 - r_kill) + 0.02, 1.2)
            k2x, k2p, _ = geodesic_rhs(pos + 0.5 * h * k1x, pp + 0.5 * h * k1p, pt, a)
            pos += h * k2x
            pp += h * k2p
            acc_ += h * k2x.norm()
            if acc_ >= 0.6:                      # sample by arc length
                traj[idx] = pos
                idx += 1
                acc_ = 0.0
        traj_len[None] = idx
        traj_fate[None] = fate
        traj_rmin[None] = rmin


# ----------------------------------------------------------------------------
# render kernel (writes linear HDR)
# ----------------------------------------------------------------------------
@ti.func
def static_ray(cam_pos: vec3, cam_r: vec3, cam_u: vec3, cam_f: vec3,
               u: ti.f32, v: ti.f32, a: ti.f32):
    """Pixel ray: past-directed null momentum + observed-energy normalization.
    (A coordinate-stationary camera; inside the ergosphere/horizon the blueshift
    normalization is clamped -- the view there is a coordinate-frame image.)"""
    rd = (cam_f + u * cam_r + v * cam_u).normalized()
    pt = solve_pt(cam_pos, rd, a)
    H_cam, _, _ = metric_Hl(cam_pos, a)
    e_obs = ti.min(ti.abs(pt) / ti.sqrt(ti.max(1.0 - 2.0 * H_cam, 1e-4)), 6.0)
    return rd, pt, e_obs


@ti.kernel
def render(cam_pos: vec3, cam_r: vec3, cam_u: vec3, cam_f: vec3, tanfov: ti.f32,
           extras: ti.template(),
           a: ti.f32, r_hor: ti.f32, cap_r: ti.f32,
           exposure: ti.f32, max_steps: ti.i32, h0: ti.f32, r_esc: ti.f32,
           redshift_on: ti.i32, beam: ti.f32, mode: ti.i32,
           star_on: ti.i32, s_bright: ti.f32,
           lat_on: ti.i32, ball_on: ti.i32, nballs: ti.i32,
           sky_on: ti.i32, sky_gain: ti.f32, jx: ti.f32, jy: ti.f32,
           rw: ti.i32, rh: ti.i32):
    # `extras` is the ONLY compile-time flag: 0 = lean kernel (stars/balls/
    # lattice compiled out entirely), 1 = full kernel with cheap runtime flags.
    # Both variants are pre-warmed at startup -> no JIT pauses when toggling.
    for i, j in hdr:
        if i >= rw or j >= rh:
            continue
        u = (2.0 * (i + jx) / rw - 1.0) * tanfov * (W / H)
        v = (2.0 * (j + jy) / rh - 1.0) * tanfov

        pos = cam_pos
        pp, pt, e_obs = static_ray(cam_pos, cam_r, cam_u, cam_f, u, v, a)
        col = vec3(0.0)
        trans = 1.0
        tint = vec3(1.0)
        nbounce = 0
        done = 0
        escaped = 0
        rd_out = pp
        last_dir = pp
        r_last = 1e9

        for _s in range(max_steps):
            k1x, k1p, r1 = geodesic_rhs(pos, pp, pt, a)
            last_dir = k1x
            r_last = r1
            # adaptive step: bound the bending angle per step (~|dp|/|p| < 0.06)
            # and shrink near the capture radius so termination is resolved
            h = h0 * ti.min(0.06 / (k1p.norm() + 1e-4),
                            0.22 * ti.max(r1 - cap_r, 0.0) + 0.015, 5.0)
            in_star = 0
            if ti.static(extras == 1):
                sgl = sgf[None]
                if star_on == 1 and ti.abs(pos.x) < sgl and ti.abs(pos.y) < sgl and ti.abs(pos.z) < sgl:
                    # occupancy mip: one cheap read decides between fine steps
                    # in debris and large empty-space skips
                    oi = ti.math.clamp(int((pos.x / (2 * sgl) + 0.5) * OCN), 0, OCN - 1)
                    oj = ti.math.clamp(int((pos.y / (2 * sgl) + 0.5) * OCN), 0, OCN - 1)
                    ok = ti.math.clamp(int((pos.z / (2 * sgl) + 0.5) * OCN), 0, OCN - 1)
                    if occ[oi, oj, ok] == 1:
                        # resolve at the (adaptive) voxel scale; stepping much
                        # below one voxel just re-reads the same trilinear data
                        h = ti.min(h, ti.max(1.0 * (2.0 * sgl / SGN), 0.12))
                        in_star = 1
                    else:
                        h = ti.min(h, 0.9 * (2.0 * sgl / OCN))   # dilated -> safe skip
                if lat_on == 1 and r1 < 40.0:
                    h = ti.min(h, 0.3)             # resolve the 3D lattice lines
                if ball_on == 1:
                    for b_ in range(nballs):
                        cb = vec3(ball_dat[b_].x, ball_dat[b_].y, ball_dat[b_].z)
                        if (pos - cb).norm() < ball_dat[b_].w + 1.2:
                            h = ti.min(h, ti.max(0.08, 0.25 * ball_dat[b_].w))

            # midpoint (RK2) integrator: with the bending-angle-capped adaptive
            # step the accuracy loss vs RK4 is invisible, and halving the
            # inlined gradient code keeps the kernel small enough for the
            # taichi/LLVM compiler (the giant RK4 kernel crashed it flakily)
            k2x, k2p, _ = geodesic_rhs(pos + 0.5 * h * k1x, pp + 0.5 * h * k1p, pt, a)
            npos = pos + h * k2x
            npp = pp + h * k2p

            if ti.static(extras == 1):
                if lat_on == 1 and done == 0:
                    # --- 3D coordinate lattice: crisp grid lines every 5M,
                    #     3 subsamples per step + super-gaussian line profile ---
                    seg_ = npos - pos
                    sdl_ = seg_.norm() / 3.0
                    for ls in ti.static(range(3)):
                        lp_ = pos + (ls + 0.5) / 3.0 * seg_
                        rr2 = lp_.dot(lp_)
                        if rr2 < 36.0 * 36.0:
                            g_ = ti.abs(ti.math.fract(lp_ / 5.0) - 0.5) * 5.0
                            m_ = ti.max(g_.x, ti.max(g_.y, g_.z))
                            d2_ = g_.dot(g_) - m_ * m_   # dist^2 to nearest line
                            q_ = d2_ / 0.0036            # (0.06 M)^2 core
                            glow = ti.exp(-q_ * q_ * 0.5)
                            fade = 1.0 - ti.math.smoothstep(28.0, 36.0, ti.sqrt(rr2))
                            fade *= ti.math.smoothstep(3.0, 7.0, (lp_ - cam_pos).norm())
                            col += (trans * tint * glow * fade * 0.55 * sdl_
                                    * vec3(0.40, 0.85, 1.0) * exposure)

                # --- reflective metal ball: mirror the ray on the sphere and
                #     keep integrating through curved space (static-mirror
                #     approximation: coordinate-frame reflection) ---
                if ball_on == 1 and done == 0:
                    for b_ in range(nballs):
                        if done == 0:
                            cb = vec3(ball_dat[b_].x, ball_dat[b_].y, ball_dat[b_].z)
                            Rb = ball_dat[b_].w
                            seg_ = npos - pos
                            oc = pos - cb
                            A2_ = seg_.dot(seg_)
                            B2_ = 2.0 * oc.dot(seg_)
                            C2_ = oc.dot(oc) - Rb * Rb
                            disc_ = B2_ * B2_ - 4.0 * A2_ * C2_
                            if disc_ > 0.0 and A2_ > 1e-12:
                                tt = (-B2_ - ti.sqrt(disc_)) / (2.0 * A2_)
                                if tt >= 0.0 and tt <= 1.0:
                                    hit = pos + tt * seg_
                                    nrm_ = (hit - cb).normalized()
                                    dirv = seg_.normalized()
                                    refl = dirv - 2.0 * dirv.dot(nrm_) * nrm_
                                    pt_old = pt
                                    npos = hit + nrm_ * 0.02
                                    npp = refl
                                    pt = solve_pt(npos, npp, a)
                                    e_obs *= ti.abs(pt) / ti.max(ti.abs(pt_old), 1e-9)
                                    tint *= vec3(0.82, 0.85, 0.90)   # steel albedo
                                    nbounce += 1
                                    if nbounce > 4:
                                        done = 1

                # --- star / tidal-debris emission, 3 subsamples per step,
                #     only inside occupied cells; gate follows the ray cutoff
                #     so an inside-horizon camera sees the infalling debris ---
                if star_on == 1 and in_star == 1 and done == 0 and r1 > cap_r + 0.03:
                    seg = npos - pos
                    sdl = seg.norm() / 1.0
                    for ss in ti.static(range(1)):
                        if done == 0:
                            sps = pos + (ss + 0.5) / 1.0 * seg
                            sdens = star_dens_sample(sps)
                            if sdens > 2e-3:
                                # sub-voxel detail: high-frequency noise breaks
                                # the fluffy trilinear blobs into granular gas;
                                # frequency tied to the (adaptive) voxel size,
                                # and a soft ramp removes hard staircase edges
                                raw = sdens
                                nmod = vnoise(sps * (0.45 * SGN / sgf[None]))
                                sdens *= 0.30 + 1.6 * nmod * nmod
                                sdens *= ti.math.smoothstep(2e-3, 9e-3, raw)
                                scp = 0.5 * (pp + npp)
                                gfac = 1.0
                                if redshift_on == 1:
                                    vv = star_vel_sample(sps) / ti.max(raw, 1e-4)
                                    gfac = doppler_g_v(sps, scp, pt, e_obs, vv, a)
                                # local temperature from the particles: blackbody
                                # color + mild T-scaling of the luminosity
                                Tloc = ti.math.clamp(star_temp_sample(sps) / ti.max(raw, 1e-4),
                                                     1200.0, 35000.0)
                                # color-preserving luminance cap: keep the body
                                # below the tonemapper's white point so the
                                # blackbody tint survives instead of clipping
                                lum = ti.min(s_bright * ti.pow(Tloc / 6000.0, 1.6)
                                             * ti.pow(gfac, 4.0 * beam) * exposure, 1.05)
                                src = blackbody_rgb(Tloc * gfac) * lum
                                sab = 1.0 - ti.exp(-16.0 * sdens * sdl)
                                col += trans * tint * src * sab
                                trans *= 1.0 - sab
                                if trans < 0.02:
                                    done = 1

            pos = npos
            pp = npp
            if done == 1:
                break
            if r1 < cap_r:
                break                              # captured / singularity guard
            if pos.dot(pos) > r_esc * r_esc and pos.dot(k1x) > 0.0:
                escaped = 1
                rd_out = k1x.normalized()
                break

        if escaped == 1:
            col += trans * tint * background(rd_out, mode, sky_on, sky_gain)
        elif done == 0 and r_last > cap_r * 1.3 and trans > 0.05:
            # ran out of steps mid-flight: fall back to the background along the
            # current direction instead of returning black (avoids dark arcs)
            col += trans * tint * background(last_dir.normalized(), mode, sky_on, sky_gain)
        hdr[i, j] = col


@ti.kernel
def accumulate(first: ti.i32):
    for i, j in hdr:
        if first == 1:
            acc[i, j] = hdr[i, j]
        else:
            acc[i, j] += hdr[i, j]


# ----------------------------------------------------------------------------
# post-processing: bloom + starburst + tonemap + upscale
# ----------------------------------------------------------------------------
@ti.func
def acc_bilerp(x: ti.f32, y: ti.f32, rw: ti.i32, rh: ti.i32, inv_n: ti.f32) -> vec3:
    x0 = ti.math.clamp(x, 0.0, rw - 1.001)
    y0 = ti.math.clamp(y, 0.0, rh - 1.001)
    i0, j0 = int(x0), int(y0)
    fx, fy = x0 - i0, y0 - j0
    i1, j1 = ti.min(i0 + 1, rw - 1), ti.min(j0 + 1, rh - 1)
    return ti.math.mix(ti.math.mix(acc[i0, j0], acc[i1, j0], fx),
                       ti.math.mix(acc[i0, j1], acc[i1, j1], fx), fy) * inv_n


@ti.kernel
def bright_pass(rw: ti.i32, rh: ti.i32, thresh: ti.f32, inv_n: ti.f32):
    for i, j in bloom0:
        c = acc_bilerp((i + 0.5) * rw / BW - 0.5, (j + 0.5) * rh / BH - 0.5, rw, rh, inv_n)
        luma = c.dot(vec3(0.2126, 0.7152, 0.0722))
        k = ti.max(luma - thresh, 0.0)
        bloom0[i, j] = c * (k / (luma + 1e-4))


@ti.kernel
def blur_pass(src: ti.template(), dst: ti.template(), dx: ti.i32, dy: ti.i32, stride: ti.i32):
    for i, j in dst:
        a_ = 0.227027 * src[i, j]
        w = ti.static([0.1945946, 0.1216216, 0.054054, 0.016216])
        for k in ti.static(range(4)):
            o = (k + 1) * stride
            a_ += w[k] * src[ti.math.clamp(i + o * dx, 0, BW - 1), ti.math.clamp(j + o * dy, 0, BH - 1)]
            a_ += w[k] * src[ti.math.clamp(i - o * dx, 0, BW - 1), ti.math.clamp(j - o * dy, 0, BH - 1)]
        dst[i, j] = a_


@ti.kernel
def streak_pass(strength: ti.f32):
    """4-arm diffraction starburst: long exponential streaks along +-45 deg."""
    for i, j in bloom1:
        c = bloom0[i, j]
        if strength > 1e-4:
            s = vec3(0.0)
            wsum = 0.0
            for k in range(1, 28):
                wk = ti.exp(-k * 0.16)
                wsum += 4.0 * wk
                o = k * 2
                s += wk * bloom0[ti.math.clamp(i + o, 0, BW - 1), ti.math.clamp(j + o, 0, BH - 1)]
                s += wk * bloom0[ti.math.clamp(i - o, 0, BW - 1), ti.math.clamp(j - o, 0, BH - 1)]
                s += wk * bloom0[ti.math.clamp(i + o, 0, BW - 1), ti.math.clamp(j - o, 0, BH - 1)]
                s += wk * bloom0[ti.math.clamp(i - o, 0, BW - 1), ti.math.clamp(j + o, 0, BH - 1)]
            c += strength * s / ti.max(wsum, 1e-4) * 4.0
        bloom1[i, j] = c


@ti.kernel
def composite(rw: ti.i32, rh: ti.i32, bloom_strength: ti.f32, inv_n: ti.f32):
    for i, j in img:
        c = acc_bilerp((i + 0.5) * rw / W - 0.5, (j + 0.5) * rh / H - 0.5, rw, rh, inv_n)
        bx = ti.math.clamp((i + 0.5) * BW / W - 0.5, 0.0, BW - 1.001)
        by = ti.math.clamp((j + 0.5) * BH / H - 0.5, 0.0, BH - 1.001)
        i0, j0 = int(bx), int(by)
        fx, fy = bx - i0, by - j0
        i1, j1 = ti.min(i0 + 1, BW - 1), ti.min(j0 + 1, BH - 1)
        b = ti.math.mix(ti.math.mix(bloom1[i0, j0], bloom1[i1, j0], fx),
                        ti.math.mix(bloom1[i0, j1], bloom1[i1, j1], fx), fy)
        t = aces(c + bloom_strength * b)
        img[i, j] = vec3(ti.pow(t.x, 1.0 / 2.2), ti.pow(t.y, 1.0 / 2.2), ti.pow(t.z, 1.0 / 2.2))


def post_process(rw, rh, bloom_strength, bloom_thresh, star_strength, inv_n):
    bright_pass(rw, rh, bloom_thresh, inv_n)
    blur_pass(bloom0, bloom1, 1, 0, 1)
    blur_pass(bloom1, bloom0, 0, 1, 1)
    blur_pass(bloom0, bloom1, 1, 0, 3)
    blur_pass(bloom1, bloom0, 0, 1, 3)
    streak_pass(star_strength)
    composite(rw, rh, bloom_strength, inv_n)


# ----------------------------------------------------------------------------
# Python-side physics helpers
# ----------------------------------------------------------------------------
def r_isco(a: float) -> float:
    """Prograde ISCO radius, Bardeen-Press-Teukolsky (1972), M = 1."""
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def orbit_EL(r0: float, rp: float, a: float):
    """Exact (E, L) of the equatorial bound orbit with apoapsis r0 and
    pericenter rp: solve g^{mu nu} p_mu p_nu = -1 with p_r = 0 at both radii."""
    def gi(r):
        delta = r * r - 2 * r + a * a
        big_a = (r * r + a * a) ** 2 - delta * a * a
        inv = 1.0 / (r * r * delta)
        return -big_a * inv, -2 * a * r * inv, (delta - a * a) * inv

    def e_of_l(L, r):
        gtt, gtp, gpp = gi(r)
        disc = (2 * gtp * L) ** 2 - 4 * gtt * (gpp * L * L + 1)
        if disc < 0:
            return None
        roots = [(2 * gtp * L + s * math.sqrt(disc)) / (2 * gtt) for s in (1, -1)]
        cands = [e for e in roots if 0.5 < e < 1.2]
        return min(cands) if cands else None

    def f(L):
        E = e_of_l(L, r0)
        if E is None:
            return None
        gtt, gtp, gpp = gi(rp)
        return gtt * E * E - 2 * gtp * E * L + gpp * L * L + 1.0

    grid = np.arange(0.5, 8.0, 0.05)
    vals = [f(L) for L in grid]
    for i in range(len(grid) - 1):
        if vals[i] is not None and vals[i + 1] is not None and vals[i] * vals[i + 1] < 0:
            lo, hi = grid[i], grid[i + 1]
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if fm is None:
                    break
                if fm * vals[i] < 0:
                    hi = mid
                else:
                    lo = mid
            L = 0.5 * (lo + hi)
            return e_of_l(L, r0), L
    return None, None


def np_Hl(pos, a):
    rho2 = float(pos @ pos)
    b = rho2 - a * a
    disc = math.sqrt(b * b + 4 * a * a * pos[2] ** 2)
    r2 = 0.5 * (b + disc)
    r = math.sqrt(max(r2, 1e-12))
    H = r2 * r / max(r2 * r2 + a * a * pos[2] ** 2, 1e-12)
    c = 1.0 / (r2 + a * a)
    l = np.array([1.0, (r * pos[0] + a * pos[1]) * c, (r * pos[1] - a * pos[0]) * c, pos[2] / r])
    return H, l, r


def np_geo_rhs(pos, pp, pt, a):
    """f64 mirror of geodesic_rhs (analytic gradient)."""
    x, y, z = pos
    rho2 = float(pos @ pos)
    b = rho2 - a * a
    disc = math.sqrt(b * b + 4 * a * a * z * z)
    r2 = 0.5 * (b + disc)
    r = math.sqrt(max(r2, 1e-12))
    drv = (r2 * pos + np.array([0.0, 0.0, a * a * z])) / max(r * disc, 1e-9)
    f = r2 * r2 + a * a * z * z
    H = r2 * r / max(f, 1e-12)
    dfv = 4 * r2 * r * drv + np.array([0.0, 0.0, 2 * a * a * z])
    dHv = (3 * r2 * f * drv - r2 * r * dfv) / max(f * f, 1e-12)
    c = 1.0 / (r2 + a * a)
    A_, B_ = r * x + a * y, r * y - a * x
    l = np.array([A_ * c, B_ * c, z / r])
    common = 2 * r * c * c
    dlx = c * (x * drv + np.array([r, a, 0.0])) - A_ * common * drv
    dly = c * (y * drv + np.array([-a, r, 0.0])) - B_ * common * drv
    dlz = np.array([0.0, 0.0, 1.0 / r]) - (z / max(r2, 1e-12)) * drv
    lp = -pt + l @ pp
    dpos = pp - 2 * H * lp * l
    grad_lp = pp[0] * dlx + pp[1] * dly + pp[2] * dlz
    dpp = dHv * lp * lp + 2 * H * lp * grad_lp
    return dpos, dpp, r


def np_vel_to_p(pos, v, a):
    H, l4, r = np_Hl(pos, a)
    l = l4[1:]
    lv = l @ v
    nrm = -((-1 + 2 * H) + 4 * H * lv + v @ v + 2 * H * lv * lv)
    if nrm <= 1e-6:
        return None
    ut = 1.0 / math.sqrt(nrm)
    lpv = 1.0 + lv
    return ut * (-1 + 2 * H * lpv), ut * (v + 2 * H * lpv * l)


def np_advance_coord(pos, p3, pt, a, dtc):
    """Advance a timelike geodesic by coordinate time dtc (f64, midpoint)."""
    nsub = 1 + int(dtc / 0.2)
    h = dtc / nsub
    r = 99.0
    for _ in range(nsub):
        H, l4, _ = np_Hl(pos, a)
        l = l4[1:]
        lp = -pt + l @ p3
        w = h / max(-pt + 2 * H * lp, 1e-5)
        k1x, k1p, _ = np_geo_rhs(pos, p3, pt, a)
        k2x, k2p, r = np_geo_rhs(pos + 0.5 * w * k1x, p3 + 0.5 * w * k1p, pt, a)
        pos = pos + w * k2x
        p3 = p3 + w * k2p
        if r < 0.3:
            break
    return pos, p3, r


def basis_from(yaw, pitch):
    cp_, sp_ = math.cos(pitch), math.sin(pitch)
    f = np.array([cp_ * math.cos(yaw), cp_ * math.sin(yaw), sp_], dtype=np.float32)
    r_ = np.cross(f, np.array([0.0, 0.0, 1.0]))
    n = np.linalg.norm(r_)
    r_ = r_ / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
    u_ = np.cross(r_, f)
    return r_.astype(np.float32), u_.astype(np.float32), f


def lookat_angles(pos):
    f = -pos / np.linalg.norm(pos)
    return math.atan2(f[1], f[0]), math.asin(max(-1.0, min(1.0, float(f[2]))))


def bl_radius(pos, a):
    rho2 = float(pos @ pos)
    b = rho2 - a * a
    return math.sqrt(max(0.5 * (b + math.sqrt(b * b + 4 * a * a * pos[2] ** 2)), 1e-12))


def halton(i, b):
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def save_png(out_dir, fname=None):
    os.makedirs(out_dir, exist_ok=True)
    if fname is None:
        fname = os.path.join(out_dir, f"kerr_{time.strftime('%Y%m%d_%H%M%S')}.png")
    ti.tools.imwrite(img.to_numpy(), fname)
    print(f"saved {fname}")
    return fname



# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    th0 = math.radians(min(max(args.inc, 1.0), 179.0))
    pos0 = np.array([args.dist * math.sin(th0), 0.0, args.dist * math.cos(th0)],
                    dtype=np.float32)
    yaw0, pitch0 = lookat_angles(pos0)
    cam = dict(pos=pos0.copy(), yaw=yaw0, pitch=pitch0)
    st = dict(spin=min(max(args.spin, 0.0), 0.999), fov=55.0, scale=1.0,
              steps=args.steps, exposure=1.0, anim=0.0,
              rs=True, beam=min(max(args.beam, 0.0), 1.0),
              ltype=1 if args.ball else 0, slv=0.35, srad=0.5,
              smu=-4.0,
              snw=10, sgamma=0.08, sbright=1.2, sdist=14.0,
              srp=max(args.srp, 1.0),
              bloom=0.70, bthresh=0.55, star_on=False, star_str=0.5,
              mode=min(max(args.mode, 0), 2), sky_on=HAS_SKY, sky_gain=1.0,
              recn=300)
    h0, r_esc = 1.0, 70.0
    t_anim = args.t_anim
    sgf[None] = min(26.0, SG_MAX)               # initial grid extent, auto-fits later
    grid_key = [None]
    star_n = [0]
    star_epoch = [0]
    balls = []

    def compact_stars():
        """Drop dead particles (captured / escaped) and pack the pool."""
        n = star_n[0]
        if n == 0:
            return
        at = spt.to_numpy()
        alive = at[:n] != 0.0
        na = int(alive.sum())
        if na > 0.8 * n:
            return
        idx = np.nonzero(alive)[0]
        for f in (spx, spp, sT, sbat):
            arr = f.to_numpy()
            arr[:na] = arr[idx]
            f.from_numpy(arr)
        at[:na] = at[idx]
        at[na:n] = 0.0
        spt.from_numpy(at)
        star_n[0] = na
        star_epoch[0] += 1
        print(f"compacted star pool: {n} -> {na} particles")

    sb_next = [0]

    def new_batch():
        """Self-gravity slot for the next star; mu and softening from the UI.
        Returns (batch id, virial velocity dispersion, tidal radius)."""
        mu = 10.0 ** st["smu"]
        b = sb_next[0] % SB_MAX
        sb_next[0] += 1
        sb_mu[b] = mu
        sb_rad[b] = 0.62 * st["srad"]
        sigma = math.sqrt(0.4 * mu / max(0.62 * st["srad"], 0.05))
        rt = st["srad"] * mu ** (-1.0 / 3.0)
        return b, sigma, rt

    def star_T0():
        """Initial temperature from the star's size: a rough main-sequence
        mass-temperature relation, T ~ M^0.5 (radius and particle count stand
        in for the mass).  0.05M dwarf ~ 2000K red, 3M giant ~ 14000K blue."""
        t = 5800.0 * (st["srad"] / 0.5) ** 0.5 * (max(st["snw"], 1) / 10.0) ** 0.12
        return float(min(max(t, 1800.0), 20000.0))

    def pool_slot(n_want):
        off = star_n[0]
        n = min(n_want, NP_STAR_MAX - off)
        if n <= 0:
            compact_stars()
            off = star_n[0]
            n = min(n_want, NP_STAR_MAX - off)
        return off, n

    def do_spawn_star():
        """Place a star on an exact bound orbit with the chosen pericenter."""
        a = st["spin"]
        off, n = pool_slot(int(st["snw"]) * 10000)
        if n <= 0:
            print(f"particle pool full ({NP_STAR_MAX}); clear stars first")
            return
        _, _, cf_ = basis_from(cam["yaw"], cam["pitch"])
        c = cam["pos"].astype(np.float64) + cf_.astype(np.float64) * st["sdist"]
        rmin = 1.0 + math.sqrt(max(1 - a * a, 0)) * 1.6 + 1.5
        if np.linalg.norm(c) < rmin:
            c = c / max(np.linalg.norm(c), 1e-6) * rmin
        rr = np.linalg.norm(c)
        that = np.cross([0.0, 0.0, 1.0], c / rr)
        if np.linalg.norm(that) < 1e-3:
            that = np.array([1.0, 0.0, 0.0])
        that /= np.linalg.norm(that)
        rp = min(st["srp"], rr * 0.85)
        E, L = orbit_EL(rr, rp, a)
        if E is None:
            vmag = 0.6 * rr / (rr ** 1.5 + a)
            print("no bound orbit for these radii: launching a plunge")
        else:
            delta = rr * rr - 2 * rr + a * a
            big_a = (rr * rr + a * a) ** 2 - delta * a * a
            inv = 1.0 / (rr * rr * delta)
            dtdl = big_a * inv * E - 2 * a * rr * inv * L
            dphdl = 2 * a * rr * inv * E + (delta - a * a) * inv * L
            vmag = math.sqrt(rr * rr + a * a) * dphdl / dtdl
            print(f"star: r0={rr:.1f}M  r_p={rp:.2f}M  E={E:.4f}  L={L:.3f}")
        b, sig, rt = new_batch()
        spawn_star(off, n, vec3(*c.astype(np.float32)),
                   vec3(*(vmag * that).astype(np.float32)), st["srad"], sig, a, star_T0(), b)
        star_n[0] = off + n
        star_epoch[0] += 1
        print(f"star spawned: {n} particles, T0={star_T0():.0f}K, "
              f"r_t≈{rt:.1f}M (total {star_n[0]})")

    def do_launch_star():
        """Launch a star / mirror ball from in front of the camera, view axis."""
        a = st["spin"]
        _, _, cf_ = basis_from(cam["yaw"], cam["pitch"])
        c = cam["pos"].astype(np.float64) + cf_.astype(np.float64) * max(3.0, 3.0 * st["srad"])
        r_hor = 1.0 + math.sqrt(max(1 - a * a, 0))
        if bl_radius(c, a) < r_hor + 0.8:
            print("too close to the horizon to launch")
            return
        vb = st["slv"] * cf_.astype(np.float64)
        res = np_vel_to_p(c, vb, a)
        if res is None:
            print("launch frame invalid here; move away from the hole")
            return
        pt0, p30 = res
        E0 = -pt0
        fate = "unbound (escapes unless captured)" if E0 >= 1.0 else "bound orbit"
        if int(st["ltype"]) == 1:
            if len(balls) >= BALL_MAX:
                print(f"ball limit ({BALL_MAX}) reached; clear first")
                return
            balls.append(dict(pos=c.copy(), p3=p30.copy(), pt=pt0, R=st["srad"]))
            print(f"mirror ball launched: beta={st['slv']:.2f}, E={E0:.4f} -> {fate} "
                  f"({len(balls)}/{BALL_MAX})")
            return
        off, n = pool_slot(int(st["snw"]) * 10000)
        if n <= 0:
            print(f"particle pool full ({NP_STAR_MAX}); clear stars first")
            return
        Lz0 = c[0] * p30[1] - c[1] * p30[0]
        b, sig, rt = new_batch()
        spawn_star(off, n, vec3(*c.astype(np.float32)), vec3(*vb.astype(np.float32)),
                   st["srad"], sig, a, star_T0(), b)
        star_n[0] = off + n
        star_epoch[0] += 1
        print(f"star launched: beta={st['slv']:.2f}, E={E0:.4f}, Lz={Lz0:.3f}, "
              f"T0={star_T0():.0f}K, r_t≈{rt:.1f}M -> {fate} (total {star_n[0]})")

    def do_seed_disk():
        """Auto-launch one star on a guaranteed disk-forming orbit: equatorial,
        bound, pericenter deep inside the tidal radius (disruption guaranteed)
        but safely outside capture -- adapts to BOTH spin and mass ratio."""
        a = st["spin"]
        off, n = pool_slot(int(st["snw"]) * 10000)
        if n <= 0:
            print(f"particle pool full ({NP_STAR_MAX}); clear first")
            return
        r0 = 19.0
        phi = np.random.rand() * 2 * math.pi
        c = np.array([r0 * math.cos(phi), r0 * math.sin(phi),
                      0.4 * (np.random.rand() - 0.5)])
        mu = 10.0 ** st["smu"]
        rt_ = st["srad"] * mu ** (-1.0 / 3.0)
        rp = float(np.clip(0.45 * rt_, 1.25 * r_isco(a), 15.0))
        E, L = orbit_EL(r0, rp, a)
        if E is None:
            rp = max(1.7 * r_isco(a), 2.2)
            E, L = orbit_EL(r0, rp, a)
        if E is None:
            print("orbit solver failed; try a different spin")
            return
        print(f"auto orbit: a={a:.3f}  r_isco={r_isco(a):.2f}M  r_t={rt_:.1f}M  -> r_p={rp:.2f}M")
        delta = r0 * r0 - 2 * r0 + a * a
        big_a = (r0 * r0 + a * a) ** 2 - delta * a * a
        inv = 1.0 / (r0 * r0 * delta)
        dtdl = big_a * inv * E - 2 * a * r0 * inv * L
        dphdl = 2 * a * r0 * inv * E + (delta - a * a) * inv * L
        vmag = math.sqrt(r0 * r0 + a * a) * dphdl / dtdl
        that = np.cross([0.0, 0.0, 1.0], c / np.linalg.norm(c))
        that /= np.linalg.norm(that)
        b, sig, rt = new_batch()
        spawn_star(off, n, vec3(*c.astype(np.float32)),
                   vec3(*(vmag * that).astype(np.float32)), st["srad"], sig, a, star_T0(), b)
        star_n[0] = off + n
        star_epoch[0] += 1
        print(f"auto star: r0={r0}M  r_p={rp:.2f}M  E={E:.4f}  L={L:.3f}  T0={star_T0():.0f}K  "
              f"r_t≈{rt:.1f}M ({n} particles, total {star_n[0]}) -- 开时间流速看它被撕碎成盘")

    def update_aim_overlay():
        """While RMB is held: one exactly integrated test geodesic, projected to
        screen (flat pinhole -- a HUD overlay, not a lensed image)."""
        cr_, cu_, cf_ = basis_from(cam["yaw"], cam["pitch"])
        base = cam["pos"].astype(np.float64)
        c = base + cf_.astype(np.float64) * max(3.0, 3.0 * st["srad"])
        predict_traj(vec3(*c.astype(np.float32)),
                     vec3(*(st["slv"] * cf_).astype(np.float32)), st["spin"], 0.30)
        n = int(traj_len[None])
        arr = np.full((2 * TRAJ_N, 2), -10.0, dtype=np.float32)
        carr = np.zeros((2 * TRAJ_N, 3), dtype=np.float32)
        dots = np.full((AIM_DOTS, 2), -10.0, dtype=np.float32)
        dcol = np.zeros((AIM_DOTS, 3), dtype=np.float32)
        if n >= 2:
            d = traj.to_numpy()[:n].astype(np.float64) - base
            xc, yc, zc = d @ cr_, d @ cu_, d @ cf_
            tf = math.tan(math.radians(st["fov"]) / 2)
            zs = np.where(np.abs(zc) < 1e-6, 1e-6, zc)
            uu = 0.5 + xc / (2 * zs * tf * (W / H))
            vv = 0.5 + yc / (2 * zs * tf)
            vis = zc > 0.05
            tpar = np.linspace(0.0, 1.0, n)
            # color = predicted fate: red plunge / orange escape / green bound
            fate = int(traj_fate[None])
            if fate == 0:
                head, tail = np.array([1.0, 0.22, 0.10]), np.array([0.55, 0.05, 0.05])
            elif fate == 1:
                head, tail = np.array([1.0, 0.72, 0.15]), np.array([0.75, 0.35, 0.05])
            else:
                head, tail = np.array([0.25, 1.0, 0.45]), np.array([0.05, 0.55, 0.30])
                if float(traj_rmin[None]) < r_isco(st["spin"]):
                    head = np.array([0.75, 1.0, 0.25])   # marginal: grazes the ISCO
            cols = head[None, :] * (1 - tpar)[:, None] + tail[None, :] * tpar[:, None]
            cols *= (1.0 - 0.70 * tpar)[:, None]
            k = 0
            for i in range(n - 1):
                if vis[i] and vis[i + 1]:
                    arr[k] = (uu[i], vv[i])
                    arr[k + 1] = (uu[i + 1], vv[i + 1])
                    carr[k] = cols[i]
                    carr[k + 1] = cols[i + 1]
                    k += 2
            kd = 0
            for i in range(0, n, 4):
                if kd < AIM_DOTS and vis[i]:
                    dots[kd] = (uu[i], vv[i])
                    dcol[kd] = cols[i] * 1.6
                    kd += 1
        aim_line.from_numpy(arr)
        aim_lcol.from_numpy(np.clip(carr, 0, 1))
        aim_dot.from_numpy(dots)
        aim_dcol.from_numpy(np.clip(dcol, 0, 1))

    def sim_particles(dt_sim):
        """Advance star particles + balls; re-deposit the grids when changed."""
        a = st["spin"]
        if dt_sim > 0.0:
            r_hor = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
            for bi, bd in enumerate(balls[:]):
                p_, m_, r_ = np_advance_coord(bd["pos"], bd["p3"], bd["pt"], a, dt_sim)
                bd["vel"] = (p_ - bd["pos"]) / max(dt_sim, 1e-6)
                bd["pos"], bd["p3"] = p_, m_
                if r_ < 0.45 or np.linalg.norm(p_) > 250:
                    balls.remove(bd)
                    print("ball reached the singularity / escaped")
            for bi, bd in enumerate(balls):
                ball_dat[bi] = [float(bd["pos"][0]), float(bd["pos"][1]),
                                float(bd["pos"][2]), float(bd["R"])]
                bv = bd.get("vel", np.zeros(3))
                ball_vel[bi] = [float(bv[0]), float(bv[1]), float(bv[2])]
            if star_n[0] > 0:
                star_com(star_n[0])               # self-gravity centers
                # particles cross the horizon and keep falling (ingoing KS is
                # regular there); recycled only near the singularity
                update_star_particles(star_n[0], dt_sim, a, st["sgamma"], 0.30, len(balls))
            grid_key[0] = None
        gk = (star_n[0], star_epoch[0])
        if grid_key[0] != gk:
            if star_n[0] > 0:
                for _retry in range(2):
                    sgrid.fill(0.0)
                    svgrid.fill(0.0)
                    stgrid.fill(0.0)
                    bmax[None] = 0.0
                    sg_cur = float(sgf[None])
                    # per-particle mass / cell volume -> number density
                    scatter_star(star_n[0], a, 2e-4 * (SGN / (2 * sg_cur)) ** 3)
                    # auto-fit the grid to the particle bounding box: a compact
                    # scene gets much finer voxels (less staircase aliasing)
                    want = min(max(float(bmax[None]) * 1.12 + 1.0, 12.0), SG_MAX)
                    if abs(want - sg_cur) / sg_cur > 0.12:
                        sgf[None] = want
                        continue                  # re-deposit at the new scale
                    break
                build_occupancy()
            grid_key[0] = gk

    def do_render(steps, hh, sc, jitter_n, first, extras_force=None):
        a = st["spin"]
        r_hor = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
        cr, cu, cf = basis_from(cam["yaw"], cam["pitch"])
        rw, rh = max(int(RW * sc), 16), max(int(RH * sc), 16)
        jx = halton(jitter_n % 64 + 1, 2) if jitter_n > 0 else 0.5
        jy = halton(jitter_n % 64 + 1, 3) if jitter_n > 0 else 0.5
        # camera outside: capture at the horizon (fast); camera inside: rays
        # keep marching and terminate near the singularity instead
        r_cam = bl_radius(cam["pos"].astype(np.float64), a)
        cap_r = r_hor * 1.01 if r_cam > r_hor * 1.06 else 0.22
        for bi, bd in enumerate(balls):
            ball_dat[bi] = [float(bd["pos"][0]), float(bd["pos"][1]),
                            float(bd["pos"][2]), float(bd["R"])]
        star_flag = 1 if star_n[0] > 0 else 0
        lat_flag = 1 if st["mode"] == 2 else 0
        ball_flag = 1 if balls else 0
        extras = 1 if (star_flag or lat_flag or ball_flag) else 0
        if extras_force is not None:
            extras = extras_force
        render(vec3(*cam["pos"]), vec3(*cr), vec3(*cu), vec3(*cf),
               math.tan(math.radians(st["fov"]) / 2),
               extras, a, r_hor, cap_r,
               st["exposure"], steps, hh, r_esc,
               int(st["rs"]), st["beam"], st["mode"],
               star_flag, st["sbright"], lat_flag, ball_flag, len(balls),
               int(st["sky_on"]), st["sky_gain"], jx, jy, rw, rh)
        accumulate(int(first))
        return rw, rh

    def do_post(rw, rh, inv_n):
        post_process(rw, rh, st["bloom"], st["bthresh"],
                     st["star_str"] if st["star_on"] else 0.0, inv_n)

    # ---------------- offline: still frame ----------------
    if args.still:
        print(f"rendering still {RW}x{RH} (ssaa {SS}), a={st['spin']} ...")
        if args.star:
            do_spawn_star()
        if args.launch:
            do_launch_star()
        if args.disk_seed:
            do_seed_disk()
        # chunked evolution so the self-gravity COM / dissipation update along
        # the way (matches the interactive per-frame stepping)
        t_left = t_anim
        while t_left > 0:
            sim_particles(min(2.0, t_left))
            t_left -= 2.0
        if star_n[0] > 0:
            at = spt.to_numpy()[:star_n[0]]
            alive = at != 0.0
            if alive.any():
                rr_ = np.linalg.norm(spx.to_numpy()[:star_n[0]][alive], axis=1)
                print(f"star debug: alive {alive.sum()}/{star_n[0]}, "
                      f"r p10/50/90/max = {np.percentile(rr_, [10, 50, 90]).round(2)} "
                      f"/ {rr_.max():.1f}")
        rw, rh = do_render(st["steps"] * 2, h0 * 0.5, 1.0, 0, True)
        ti.sync()
        print("  [dbg] render ok")
        do_post(rw, rh, 1.0)
        ti.sync()
        print("  [dbg] post ok")
        save_png(args.out)
        return

    # ---------------- offline: animation frames ----------------
    if args.anim > 0:
        out_dir = os.path.join(args.out, f"evolve_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)
        if args.star:
            do_spawn_star()
        if args.launch:
            do_launch_star()
        dphi = math.radians(args.orbit)
        nf, sub = args.anim, max(args.mb, 1)
        yaw_c, pitch_c = cam["yaw"], cam["pitch"]
        print(f"rendering {nf} frames x {sub} subframes -> {out_dir}")
        t0 = time.time()
        for fi in range(nf):
            for s in range(sub):
                frac = (fi + s / sub) / nf
                ang = dphi * frac
                ca_, sa_ = math.cos(ang), math.sin(ang)
                cam["pos"] = np.array([ca_ * pos0[0] - sa_ * pos0[1],
                                       sa_ * pos0[0] + ca_ * pos0[1], pos0[2]],
                                      dtype=np.float32)
                cam["yaw"], cam["pitch"] = lookat_angles(cam["pos"])
                sim_particles(args.rotrate / sub)
                rw, rh = do_render(st["steps"], h0, 1.0, 0, s == 0)
            do_post(rw, rh, 1.0 / sub)
            save_png(out_dir, os.path.join(out_dir, f"frame_{fi:04d}.png"))
            print(f"  frame {fi + 1}/{nf}  ({(time.time() - t0) / (fi + 1):.1f} s/frame)")
        print("ffmpeg example:")
        print(f'  ffmpeg -framerate 30 -i "{out_dir}\\frame_%04d.png" -c:v libx264 -pix_fmt yuv420p blackhole.mp4')
        return

    # ---------------- offline: pre-rendered orbit for the viewer ----------------
    if args.prerender > 0:
        import json
        from PIL import Image
        out_dir = os.path.join("prerender", f"orbit_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(out_dir, exist_ok=True)
        nf = args.prerender
        print(f"pre-rendering {nf} orbit frames at {RW}x{RH} (ssaa {SS}) -> {out_dir}")
        t0 = time.time()
        if args.star:
            do_spawn_star()
        sim_particles(t_anim)
        for k in range(nf):
            ang = 2.0 * math.pi * k / nf
            ca_, sa_ = math.cos(ang), math.sin(ang)
            cam["pos"] = np.array([ca_ * pos0[0] - sa_ * pos0[1],
                                   sa_ * pos0[0] + ca_ * pos0[1], pos0[2]],
                                  dtype=np.float32)
            cam["yaw"], cam["pitch"] = lookat_angles(cam["pos"])
            rw, rh = do_render(st["steps"] * 2, h0 * 0.5, 1.0, 0, True)
            do_post(rw, rh, 1.0)
            arr = (img.to_numpy() * 255.0 + 0.5).astype(np.uint8).transpose(1, 0, 2)[::-1]
            Image.fromarray(arr).save(os.path.join(out_dir, f"frame_{k:04d}.jpg"), quality=93)
            print(f"  {k + 1}/{nf}  ({(time.time() - t0) / (k + 1):.1f} s/frame)")
        with open(os.path.join(out_dir, "meta.json"), "w") as fh:
            json.dump(dict(n=nf, w=W, h=H, spin=st["spin"], inc=args.inc, dist=args.dist), fh)
        print("view it with:")
        print(f'  .venv\\Scripts\\python.exe orbit_viewer.py "{out_dir}"')
        return

    # ---------------- interactive ----------------
    window = ti.ui.Window("KERR // BLACK HOLE SANDBOX", (W, H), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    # pre-warm BOTH kernel variants so feature toggles never JIT-pause; the
    # taichi offline cache makes this near-instant from the second launch on
    print("预编译渲染内核（仅首次启动较慢，之后命中磁盘缓存）...")
    _t0 = time.time()
    do_render(8, 1.0, 0.012, 0, True, extras_force=0)
    print("  [warm] lean ok")
    do_render(8, 1.0, 0.012, 0, True, extras_force=1)
    print("  [warm] full ok")
    do_post(16, 16, 1.0)
    print("  [warm] post ok")
    predict_traj(vec3(20.0, 0.0, 0.0), vec3(-0.3, 0.0, 0.0), st["spin"], 0.3)
    ti.sync()
    print(f"内核就绪 ({time.time() - _t0:.1f} s)")

    last_cursor = None
    t_prev, fps = time.time(), 0.0
    n_accum = 0
    prev_key = None
    rmb_prev = False
    frame_i = 0

    while window.running:
        frame_i += 1
        if frame_i % 900 == 0 and star_n[0] > 0:
            compact_stars()                       # reclaim captured/escaped slots

        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == "p":
                save_png(args.out)

        now = time.time()
        dt = min(now - t_prev, 0.1)

        # mouse look (LMB drag, or while aiming with RMB held)
        moved = False
        if window.is_pressed(ti.ui.LMB) or window.is_pressed(ti.ui.RMB):
            cur = window.get_cursor_pos()
            if last_cursor is not None and (abs(cur[0] - last_cursor[0]) > 1e-5
                                            or abs(cur[1] - last_cursor[1]) > 1e-5):
                cam["yaw"] -= (cur[0] - last_cursor[0]) * 3.2
                cam["pitch"] = min(max(cam["pitch"] + (cur[1] - last_cursor[1]) * 2.4, -1.45), 1.45)
                moved = True
            last_cursor = cur
        else:
            last_cursor = None

        # WASD free-fly; the horizon is NOT a wall -- only the singularity is
        cr_, cu_, cf_ = basis_from(cam["yaw"], cam["pitch"])
        rad = float(np.linalg.norm(cam["pos"]))
        spd = max(0.25, 0.30 * rad) * dt * (4.0 if window.is_pressed(ti.ui.SHIFT) else 1.0)
        mv = np.zeros(3, dtype=np.float32)
        if window.is_pressed("w"):
            mv += cf_
        if window.is_pressed("s"):
            mv -= cf_
        if window.is_pressed("a"):
            mv -= cr_
        if window.is_pressed("d"):
            mv += cr_
        if window.is_pressed("e"):
            mv += np.array([0, 0, 1], dtype=np.float32)
        if window.is_pressed("q"):
            mv -= np.array([0, 0, 1], dtype=np.float32)
        if np.linalg.norm(mv) > 1e-6:
            cam["pos"] = cam["pos"] + mv / np.linalg.norm(mv) * spd
            nr = np.linalg.norm(cam["pos"])
            if nr < 0.35:
                cam["pos"] *= 0.35 / nr           # singularity guard
            if nr > 300.0:
                cam["pos"] *= 300.0 / nr
            moved = True

        # ---- in-game control panel (imgui; English -- the GGUI font has no CJK)
        a_p = st["spin"]
        rh_p = 1.0 + math.sqrt(max(1.0 - a_p * a_p, 0.0))
        rc_p = bl_radius(cam["pos"].astype(np.float64), a_p)
        act = {k: False for k in ("launch", "place", "auto", "clear", "shot", "reset", "rec")}
        with gui.sub_window("KERR // BLACK HOLE", 0.010, 0.012, 0.235, 0.972) as w:
            w.text(f"FPS {fps:5.1f}   SAMPLES {n_accum}")
            w.text(f"r = {rc_p:6.2f} M" + ("   << INSIDE HORIZON >>" if rc_p < rh_p else ""))
            w.text(f"particles {star_n[0] // 10000}0k   balls {len(balls)}/{BALL_MAX}")
            w.text("")
            w.text("--- SPACETIME ---")
            st["spin"] = w.slider_float("spin a/M", st["spin"], 0.0, 0.999)
            st["anim"] = w.slider_float("time flow", st["anim"], 0.0, 30.0)
            st["rs"] = w.checkbox("Doppler + redshift", st["rs"])
            st["beam"] = w.slider_float("beaming (1=real)", st["beam"], 0.0, 1.0)
            w.text("")
            w.text("--- LAUNCH (hold RMB to aim) ---")
            st["ltype"] = 1 if w.checkbox("mirror ball", st["ltype"] == 1) else 0
            st["slv"] = w.slider_float("speed (beta)", st["slv"], 0.05, 0.9)
            st["srad"] = w.slider_float("radius (M)", st["srad"], 0.05, 3.0)
            st["smu"] = w.slider_float("lg mass ratio", st["smu"], -6.0, -2.5)
            st["snw"] = w.slider_int("particles x10k", int(st["snw"]), 1,
                                     max(NP_STAR_MAX // 10000, 100))
            st["sgamma"] = w.slider_float("circularization", st["sgamma"], 0.0, 0.5)
            st["sbright"] = w.slider_float("star brightness", st["sbright"], 0.1, 4.0)
            act["launch"] = w.button("  LAUNCH  ")
            act["auto"] = w.button("  AUTO DISK STAR  ")
            st["srp"] = w.slider_float("place: pericenter", st["srp"], 1.0, 16.0)
            act["place"] = w.button("  PLACE ON ORBIT  ")
            act["clear"] = w.button("  CLEAR ALL  ")
            w.text("")
            w.text("--- DISPLAY ---")
            st["mode"] = w.slider_int("bg: sky/grid/lattice", int(st["mode"]), 0, 2)
            st["scale"] = w.slider_float("render scale", st["scale"], 0.25, 1.0)
            st["steps"] = w.slider_int("ray steps", int(st["steps"]), 100, 1200)
            st["exposure"] = w.slider_float("exposure", st["exposure"], 0.05, 6.0)
            st["bloom"] = w.slider_float("bloom", st["bloom"], 0.0, 2.0)
            st["star_on"] = w.checkbox("starburst flare", st["star_on"])
            w.text("")
            w.text("--- RECORD ---")
            st["recn"] = w.slider_int("frames", int(st["recn"]), 60, 1200)
            act["rec"] = w.button("  REC EVOLUTION  ")
            act["shot"] = w.button("  SCREENSHOT (P)  ")
            act["reset"] = w.button("  RESET VIEW  ")
            w.text("")
            w.text("LMB drag: look   WASD: move")
            w.text("Q/E: down/up     Shift: fast")
            w.text("RMB: aim + launch on release")
            w.text("the horizon is not a wall")

        if act["launch"]:
            do_launch_star()
        if act["place"]:
            do_spawn_star()
        if act["auto"]:
            do_seed_disk()
        if act["shot"]:
            save_png(args.out)
        if act["reset"]:
            cam["pos"] = pos0.copy()
            cam["yaw"], cam["pitch"] = yaw0, pitch0
            moved = True
        if act["clear"]:
            star_n[0] = 0
            star_epoch[0] += 1
            balls.clear()
            print("stars/balls cleared")
        if act["rec"]:
            nf = int(st["recn"])
            out_dir = os.path.join(args.out, f"evolve_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(out_dir, exist_ok=True)
            dt_f = (st["anim"] if st["anim"] > 0 else 8.0) / 30.0
            print(f"recording {nf} frames -> {out_dir} (ESC to abort)")
            aborted = False
            for fi in range(nf):
                for e in window.get_events(ti.ui.PRESS):
                    if e.key == ti.ui.ESCAPE:
                        aborted = True
                if aborted or not window.running:
                    break
                t_anim += dt_f
                sim_particles(dt_f)
                rw0, rh0 = do_render(st["steps"] * 2, h0 * 0.5, 1.0, 0, True)
                do_post(rw0, rh0, 1.0)
                save_png(out_dir, os.path.join(out_dir, f"frame_{fi:04d}.png"))
                canvas.set_image(img)
                with gui.sub_window("REC", 0.4, 0.46, 0.2, 0.08) as wr:
                    wr.text(f"REC {fi + 1}/{nf}  (ESC aborts)")
                window.show()
            print("ffmpeg example:")
            print(f'  ffmpeg -framerate 30 -i "{out_dir}\\frame_%04d.png" '
                  f'-c:v libx264 -pix_fmt yuv420p blackhole.mp4')
            n_accum = 0

        # sim time
        dt_sim = dt * st["anim"]
        if dt_sim > 0:
            t_anim += dt_sim
        sim_particles(dt_sim)

        # right-mouse: hold to aim, release to launch
        rmb = window.is_pressed(ti.ui.RMB)
        if rmb:
            update_aim_overlay()
        elif rmb_prev:
            do_launch_star()
        rmb_prev = rmb

        # temporal accumulation: restart when anything render-relevant changes
        key = (tuple(np.round(cam["pos"], 4)), round(cam["yaw"], 5), round(cam["pitch"], 5),
               st["spin"], st["fov"], st["scale"], st["steps"], st["exposure"],
               st["rs"], st["beam"], st["mode"], st["sky_on"], st["sky_gain"],
               st["sbright"], st["sgamma"], t_anim,
               star_n[0], star_epoch[0], len(balls),
               tuple(np.round(b_["pos"], 3).tolist() for b_ in balls))
        if moved or key != prev_key:
            n_accum = 0
        prev_key = key

        if n_accum < 256:
            rw, rh = do_render(st["steps"], h0, st["scale"], n_accum, n_accum == 0)
            n_accum += 1
        rw, rh = max(int(RW * st["scale"]), 16), max(int(RH * st["scale"]), 16)
        do_post(rw, rh, 1.0 / max(n_accum, 1))

        canvas.set_image(img)
        if rmb:
            canvas.lines(aim_line, 0.006, per_vertex_color=aim_lcol)
            canvas.lines(aim_line, 0.0014, per_vertex_color=aim_lcol)
            canvas.circles(aim_dot, 0.0035, per_vertex_color=aim_dcol)
        window.show()

        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps > 0 else 1.0 / max(now - t_prev, 1e-6)
        t_prev = now


if __name__ == "__main__":
    main()
