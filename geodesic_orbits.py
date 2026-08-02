"""
Kerr / Schwarzschild geodesic orbits and horizon geometry (analysis-grade, f64).

Boyer-Lindquist coordinates, geometric units G = c = M = 1.

Geodesics are integrated with the super-Hamiltonian

    H = (1/2) g^{mu nu} p_mu p_nu          (= 0 null, = -1/2 timelike)

using Hamilton's equations and scipy DOP853 (rtol 1e-10).  p_t = -E and
p_phi = L are exact constants; dp_r/dlam and dp_theta/dlam are obtained from
central finite differences of H in double precision.  Inverse BL metric:

    Sigma = r^2 + a^2 cos^2(th),  Delta = r^2 - 2r + a^2,
    A = (r^2+a^2)^2 - Delta a^2 sin^2(th)
    g^tt = -A/(Sigma Delta),     g^tph = -2 a r/(Sigma Delta),
    g^phph = (Delta - a^2 sin^2 th)/(Sigma Delta sin^2 th),
    g^rr = Delta/Sigma,          g^thth = 1/Sigma.

Figures
-------
1. horizons_ergospheres.png : outer/inner horizon + ergosphere vs spin a.
2. orbits.png :
   (a) Schwarzschild photons near the critical impact parameter b_c = 3*sqrt(3),
   (b) Kerr a=0.95 spherical photon orbit (Bardeen 1973 photon-shell constants
       Lambda(r) = -(r^3 - 3r^2 + a^2 r + a^2)/(a(r-1)),
       Q(r)      = -r^3 (r^3 - 6r^2 + 9r - 4a^2)/(a^2 (r-1)^2),  E = 1),
   (c) frame dragging: zero-angular-momentum particle released at rest,
       a = 0.95 vs a = 0, with the prograde ISCO,
   (d) periapsis precession of a bound Schwarzschild orbit.

Run:  python geodesic_orbits.py [--no-show]
"""

import argparse
import math
import os

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# metric and integrator
# ----------------------------------------------------------------------------
def inv_metric(r, th, a):
    """Nonzero inverse BL metric components (g^tt, g^tph, g^phph, g^rr, g^thth)."""
    s2 = math.sin(th) ** 2
    sig = r * r + a * a * math.cos(th) ** 2
    dlt = r * r - 2.0 * r + a * a
    big_a = (r * r + a * a) ** 2 - dlt * a * a * s2
    return (-big_a / (sig * dlt),
            -2.0 * a * r / (sig * dlt),
            (dlt - a * a * s2) / (sig * dlt * s2),
            dlt / sig,
            1.0 / sig)


def hamiltonian(r, th, pr, pth, E, L, a):
    gtt, gtp, gpp, grr, gthth = inv_metric(r, th, a)
    return 0.5 * (gtt * E * E - 2.0 * gtp * E * L + gpp * L * L
                  + grr * pr * pr + gthth * pth * pth)


def rhs(lam, y, E, L, a):
    """y = (r, th, phi, pr, pth)."""
    r, th, phi, pr, pth = y
    gtt, gtp, gpp, grr, gthth = inv_metric(r, th, a)
    drdl = grr * pr
    dthdl = gthth * pth
    dphdl = -gtp * E + gpp * L          # g^{ph mu} p_mu with p_t = -E
    eps_r = 1e-7 * max(1.0, abs(r))
    eps_t = 1e-7
    dHdr = (hamiltonian(r + eps_r, th, pr, pth, E, L, a)
            - hamiltonian(r - eps_r, th, pr, pth, E, L, a)) / (2 * eps_r)
    dHdth = (hamiltonian(r, th + eps_t, pr, pth, E, L, a)
             - hamiltonian(r, th - eps_t, pr, pth, E, L, a)) / (2 * eps_t)
    return [drdl, dthdl, dphdl, -dHdr, -dHdth]


def integrate(y0, E, L, a, lam_max, r_stop_out=80.0):
    r_hor = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))

    def hit_horizon(lam, y, *f):
        return y[0] - r_hor * 1.003
    hit_horizon.terminal, hit_horizon.direction = True, -1

    def escaped(lam, y, *f):
        return y[0] - r_stop_out
    escaped.terminal, escaped.direction = True, 1

    sol = solve_ivp(rhs, (0.0, lam_max), y0, args=(E, L, a), method="DOP853",
                    rtol=1e-10, atol=1e-12, dense_output=True,
                    events=(hit_horizon, escaped), max_step=lam_max / 200)
    lam = np.linspace(0.0, sol.t[-1], 6000)
    return sol.sol(lam)


def to_xyz(r, th, ph, a):
    """Oblate (Kerr-Schild-like) embedding: x^2+y^2 = (r^2+a^2) sin^2 th."""
    rho = np.sqrt(r * r + a * a) * np.sin(th)
    return rho * np.cos(ph), rho * np.sin(ph), r * np.cos(th)


# ----------------------------------------------------------------------------
# initial-condition helpers
# ----------------------------------------------------------------------------
def photon_equatorial(r0, b, a, ingoing=True):
    """Null, theta = pi/2, E = 1, L = b; p_r from H = 0."""
    E, L = 1.0, b
    gtt, gtp, gpp, grr, _ = inv_metric(r0, math.pi / 2, a)
    pr2 = -(gtt * E * E - 2.0 * gtp * E * L + gpp * L * L) / grr
    pr = -math.sqrt(max(pr2, 0.0)) if ingoing else math.sqrt(max(pr2, 0.0))
    return [r0, math.pi / 2, 0.0, pr, 0.0], E, L


def photon_spherical(r0, a):
    """Spherical photon orbit constants (Bardeen 1973), E = 1."""
    lam_c = -(r0**3 - 3.0 * r0**2 + a * a * r0 + a * a) / (a * (r0 - 1.0))
    q_c = -(r0**3) * (r0**3 - 6.0 * r0**2 + 9.0 * r0 - 4.0 * a * a) / (a * a * (r0 - 1.0) ** 2)
    pth0 = math.sqrt(max(q_c, 0.0))            # at the equator p_th^2 = Q
    return [r0, math.pi / 2, 0.0, 0.0, pth0], 1.0, lam_c


def zamo_drop(r0, a):
    """Timelike, at rest with L = 0 (locally nonrotating release)."""
    gtt, _, _, _, _ = inv_metric(r0, math.pi / 2, a)
    E = math.sqrt(1.0 / -gtt)                  # from g^tt E^2 = -1 with p_r = p_th = 0
    return [r0, math.pi / 2, 0.0, 0.0, 0.0], E, 0.0


def bound_orbit_schw(r_apo, L):
    """Timelike Schwarzschild orbit starting at apoapsis."""
    E = math.sqrt((1.0 - 2.0 / r_apo) * (1.0 + L * L / r_apo**2))
    return [r_apo, math.pi / 2, 0.0, 0.0, 0.0], E, L


def r_isco(a):
    z1 = 1.0 + (1.0 - a * a) ** (1 / 3) * ((1.0 + a) ** (1 / 3) + (1.0 - a) ** (1 / 3))
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    return 3.0 + z2 - math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2))


def r_photon_eq(a, prograde=True):
    """Equatorial circular photon orbit radius (Bardeen-Press-Teukolsky)."""
    s = -1.0 if prograde else 1.0
    return 2.0 * (1.0 + math.cos(2.0 / 3.0 * math.acos(s * a)))


# ----------------------------------------------------------------------------
# figure 1: horizons and ergospheres
# ----------------------------------------------------------------------------
def surface(r_of_th, a, phi_max=0.78 * math.pi, n=80):
    th = np.linspace(0.0, math.pi, n)
    ph = np.linspace(-phi_max, phi_max, n)
    TH, PH = np.meshgrid(th, ph)
    R = r_of_th(TH)
    return to_xyz(R, TH, PH, a)


def fig_geometry(out_dir, show):
    spins = [0.0, 0.6, 0.9, 0.998]
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle("Kerr horizons and ergospheres  (cutaway, M = 1)", fontsize=14)
    for k, a in enumerate(spins):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        rp = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
        rm = 1.0 - math.sqrt(max(1.0 - a * a, 0.0))
        X, Y, Z = surface(lambda TH: rp * np.ones_like(TH), a)
        ax.plot_surface(X, Y, Z, color="#202028", alpha=1.0, linewidth=0)
        X, Y, Z = surface(lambda TH: 1.0 + np.sqrt(np.maximum(1.0 - a * a * np.cos(TH) ** 2, 0.0)), a)
        ax.plot_surface(X, Y, Z, color="#e07b30", alpha=0.30, linewidth=0)
        if a > 0.0 and rm > 1e-3:
            X, Y, Z = surface(lambda TH: rm * np.ones_like(TH), a)
            ax.plot_surface(X, Y, Z, color="#7a2030", alpha=0.85, linewidth=0)
        ax.set_title(f"a = {a}    r+ = {rp:.3f},  r- = {rm:.3f}")
        ax.set_box_aspect((1, 1, 1))
        lim = 2.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_xlabel("x/M"); ax.set_ylabel("y/M"); ax.set_zlabel("z/M")
    fig.text(0.5, 0.02,
             "grey: outer horizon r+   |   orange: ergosphere r = 1 + sqrt(1 - a^2 cos^2 th)   |   "
             "dark red: inner horizon r-", ha="center", fontsize=11)
    path = os.path.join(out_dir, "horizons_ergospheres.png")
    fig.savefig(path, dpi=180)
    print("saved", path)


# ----------------------------------------------------------------------------
# figure 2: orbits
# ----------------------------------------------------------------------------
def draw_horizon_sphere(ax, a, lim):
    rp = 1.0 + math.sqrt(max(1.0 - a * a, 0.0))
    th = np.linspace(0, math.pi, 40)
    ph = np.linspace(0, 2 * math.pi, 60)
    TH, PH = np.meshgrid(th, ph)
    X, Y, Z = to_xyz(rp * np.ones_like(TH), TH, PH, a)
    ax.plot_surface(X, Y, Z, color="black", alpha=0.9, linewidth=0)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xlabel("x/M"); ax.set_ylabel("y/M"); ax.set_zlabel("z/M")


def equatorial_circle(ax, r, a, **kw):
    ph = np.linspace(0, 2 * math.pi, 200)
    x, y, z = to_xyz(np.full_like(ph, r), np.full_like(ph, math.pi / 2), ph, a)
    ax.plot(x, y, z, **kw)


def fig_orbits(out_dir, show):
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle("Geodesics around Schwarzschild and Kerr black holes (M = 1)", fontsize=14)

    # (a) Schwarzschild photons near b_c = 3 sqrt(3) ~ 5.196
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    bc = 3.0 * math.sqrt(3.0)
    for b, col, lab in [(bc + 0.02, "#48b0ff", f"b = b_c + 0.02 (escapes)"),
                        (bc - 0.02, "#ff5050", f"b = b_c - 0.02 (captured)")]:
        y0, E, L = photon_equatorial(30.0, b, 0.0)
        r, th, ph = integrate(y0, E, L, 0.0, 220.0)[0:3]
        ax.plot(*to_xyz(r, th, ph, 0.0), color=col, lw=1.2, label=lab)
    equatorial_circle(ax, 3.0, 0.0, color="#ffd24d", ls="--", lw=1.0, label="photon sphere r = 3")
    draw_horizon_sphere(ax, 0.0, 9.0)
    ax.view_init(elev=90, azim=-90)
    ax.set_title("(a) Schwarzschild: photons near critical impact parameter")
    ax.legend(loc="upper left", fontsize=8)

    # (b) Kerr spherical photon orbit, a = 0.95, r0 = 2.4
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    a = 0.95
    y0, E, L = photon_spherical(2.4, a)
    # spherical photon orbits are unstable: keep lambda short enough that the
    # exponential drift away from r0 stays small (label reports the actual drift)
    r, th, ph = integrate(y0, E, L, a, 28.0)[0:3]
    ax.plot(*to_xyz(r, th, ph, a), color="#48b0ff", lw=1.1,
            label=f"spherical photon orbit r = 2.4 (dr/r < {abs(r - 2.4).max() / 2.4:.0e})")
    equatorial_circle(ax, r_photon_eq(a, True), a, color="#ffd24d", ls="--", lw=1.0,
                      label=f"prograde r_ph = {r_photon_eq(a, True):.2f}")
    equatorial_circle(ax, r_photon_eq(a, False), a, color="#ff9d4d", ls="--", lw=1.0,
                      label=f"retrograde r_ph = {r_photon_eq(a, False):.2f}")
    draw_horizon_sphere(ax, a, 4.5)
    ax.view_init(elev=18, azim=-55)
    ax.set_title(f"(b) Kerr a = {a}: photon shell")
    ax.legend(loc="upper left", fontsize=8)

    # (c) frame dragging: ZAMO release from rest, a = 0.95 vs a = 0
    ax = fig.add_subplot(2, 2, 3, projection="3d")
    for a_c, col, lab in [(0.95, "#48b0ff", "a = 0.95 (dragged)"),
                          (0.0, "#ff5050", "a = 0 (radial plunge)")]:
        y0, E, L = zamo_drop(12.0, a_c)
        r, th, ph = integrate(y0, E, L, a_c, 400.0)[0:3]
        ax.plot(*to_xyz(r, th, ph, a_c), color=col, lw=1.4, label=lab)
    equatorial_circle(ax, r_isco(0.95), 0.95, color="#9dff70", ls="--", lw=1.0,
                      label=f"ISCO(a=0.95) = {r_isco(0.95):.3f}")
    draw_horizon_sphere(ax, 0.95, 13.0)
    ax.view_init(elev=90, azim=-90)
    ax.set_title("(c) frame dragging: particle released at rest (L = 0), r0 = 12")
    ax.legend(loc="upper left", fontsize=8)

    # (d) periapsis precession, Schwarzschild bound orbit
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    y0, E, L = bound_orbit_schw(25.0, 3.9)
    r, th, ph = integrate(y0, E, L, 0.0, 3000.0)[0:3]
    ax.plot(*to_xyz(r, th, ph, 0.0), color="#48b0ff", lw=0.9)
    draw_horizon_sphere(ax, 0.0, 26.0)
    ax.view_init(elev=75, azim=-60)
    ax.set_title(f"(d) Schwarzschild bound orbit, E = {E:.4f}, L = {L}: periapsis precession")

    path = os.path.join(out_dir, "orbits.png")
    fig.savefig(path, dpi=180)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--out", type=str, default="figures")
    a = ap.parse_args()
    if a.no_show:
        matplotlib.use("Agg")
    os.makedirs(a.out, exist_ok=True)
    plt.style.use("dark_background")
    fig_geometry(a.out, not a.no_show)
    fig_orbits(a.out, not a.no_show)
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
