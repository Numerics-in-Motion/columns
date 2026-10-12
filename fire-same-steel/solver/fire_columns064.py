"""Spinoff 064 -- the channel's four same-steel columns in a standard fire.

Four columns, each 0.004 m^2 of steel (the 020 sections: solid rod, round tube, square box,
I-section), 3.0 m, pinned, unprotected, heated by the ISO 834 standard fire. How long until each
column's modelled axial resistance falls below the load it carries?

EVERYTHING IS FROM EN 1993-1-2:2005 (no fitted constant):
  * ISO 834 gas temperature                      theta_g = 20 + 345 log10(8 t + 1), t in min
  * unprotected steel heating, eq. 4.25          d(theta_a) = k_sh (A_m/V)/(c_a rho_a) h_net dt, dt <= 5 s
    h_net = alpha_c (theta_g - theta_a) + Phi eps_m eps_f sigma [(theta_g+273)^4 - (theta_a+273)^4]
    alpha_c = 25 W/m^2K, eps_m = 0.7, eps_f = 1, Phi = 1
  * specific heat c_a(theta), 3.4.1.2;  rho_a = 7850
  * reduction factors k_y,theta and k_E,theta, Table 3.1 (linear interpolation, as the standard allows)
  * flexural buckling in fire, eq. 4.5-4.7, alpha = 0.65 sqrt(235/f_y), gamma_M,fi = 1.0
  * k_sh = 0.9 [A_m/V]_b / [A_m/V] for the I-section, 1 for the convex sections; hollow sections
    are heated on their outside perimeter only.

Material and sections are the PARENT's (020 / samesteel_config.json): E = 200 GPa, f_y = 250 MPa.

Registration: analyze/261011/064_registration_fire_v2.md (frozen before the production solve).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E = 200e9
FY = 250e6
RHO = 7850.0
L = 3.0
AREA = 0.004
EDITION = "EN 1993-1-2:2005"
GAMMA_M_FI = 1.0
ALPHA_C, EPS_F, PHI, SIGMA = 25.0, 1.0, 1.0, 5.67e-8
EPS_M = 0.7
T_TAB = [20, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
KY_TAB = [1, 1, 1, 1, 1, .78, .47, .23, .11, .06, .04, .02, 0]
KE_TAB = [1, 1, .9, .8, .7, .6, .31, .13, .09, .0675, .045, .0225, 0]
KEYS = ("solid_circle", "hollow_tube", "box", "i_section")
LABELS = {"solid_circle": "SOLID ROD", "hollow_tube": "ROUND TUBE", "box": "SQUARE BOX",
          "i_section": "I-SECTION"}
# The parent's published cold capacities (020 results.json, nonlinear first yield, L/1000 bow).
PARENT_COLD_N = {"solid_circle": 249233.01718479025, "hollow_tube": 899508.4397878932,
                 "box": 943972.829828636, "i_section": 356399.43820129655}


def sections():
    """The 020 sections, rebuilt from their outer sizes and the exact area."""
    out = {}
    D = 2 * math.sqrt(AREA / math.pi)
    out["solid_circle"] = dict(A=math.pi * D ** 2 / 4, I=math.pi * D ** 4 / 64, Am=math.pi * D,
                               Am_box=math.pi * D, outer=D, ksh=1.0, wall=None)
    Dt = 0.170
    t = 0.5 * (Dt - math.sqrt(Dt ** 2 - 4 * AREA / math.pi))
    di = Dt - 2 * t
    out["hollow_tube"] = dict(A=math.pi * (Dt ** 2 - di ** 2) / 4, I=math.pi * (Dt ** 4 - di ** 4) / 64,
                              Am=math.pi * Dt, Am_box=math.pi * Dt, outer=Dt, ksh=1.0, wall=t)
    B = 0.170
    tb = 0.5 * (B - math.sqrt(B ** 2 - AREA))
    bi = B - 2 * tb
    out["box"] = dict(A=B ** 2 - bi ** 2, I=(B ** 4 - bi ** 4) / 12, Am=4 * B, Am_box=4 * B, outer=B,
                      ksh=1.0, wall=tb)
    h, bf, r = 0.180, 0.090, 2.0
    a, b, c = -2 * r, 2 * bf * r + h, -AREA
    tw = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    tf = r * tw
    hw = h - 2 * tf
    Iy = (2 * tf * bf ** 3 + hw * tw ** 3) / 12
    Am = 2 * h + 4 * bf - 2 * tw
    Am_box = 2 * (h + bf)
    out["i_section"] = dict(A=2 * bf * tf + hw * tw, I=Iy, Am=Am, Am_box=Am_box, outer=h,
                            ksh=0.9 * Am_box / Am, wall=tw, tf=tf, bf=bf, h=h)
    for s in out.values():
        s["AmV"] = s["Am"] / s["A"]
    return out


def fire_class(s, key):
    """Cross-section class in fire, EN 1993-1-2 4.2.2: eps = 0.85 sqrt(235/f_y), EN 1993-1-1 Table 5.2."""
    eps = 0.85 * math.sqrt(235e6 / FY)
    if key == "solid_circle":
        return 1, "solid"
    if key == "hollow_tube":
        r = s["outer"] / s["wall"]
        lim = [50 * eps ** 2, 70 * eps ** 2, 90 * eps ** 2]
    elif key == "box":
        r = (s["outer"] - 3 * s["wall"]) / s["wall"]
        lim = [33 * eps, 38 * eps, 42 * eps]
    else:   # governing part: outstand flange c/t (internal web checked too)
        c_out = (s["bf"] - s["wall"]) / 2 / s["tf"]
        c_web = (s["h"] - 2 * s["tf"]) / s["wall"]
        k1 = next((i + 1 for i, l in enumerate([9 * eps, 10 * eps, 14 * eps]) if c_out <= l), 4)
        k2 = next((i + 1 for i, l in enumerate([33 * eps, 38 * eps, 42 * eps]) if c_web <= l), 4)
        return max(k1, k2), f"flange c/t {c_out:.2f}, web c/t {c_web:.2f}"
    k = next((i + 1 for i, l in enumerate(lim) if r <= l), 4)
    return k, f"ratio {r:.2f} vs limits {[round(x, 2) for x in lim]}"


def iso834(t_s):
    return 20.0 + 345.0 * math.log10(8.0 * t_s / 60.0 + 1.0)


def ca(th):
    if th < 600:
        return 425 + 7.73e-1 * th - 1.69e-3 * th ** 2 + 2.22e-6 * th ** 3
    if th < 735:
        return 666 + 13002 / (738 - th)
    if th < 900:
        return 545 + 17820 / (th - 731)
    return 650.0


def ky(th):
    return float(np.interp(th, T_TAB, KY_TAB))


def kE(th):
    return float(np.interp(th, T_TAB, KE_TAB))


def n_b_fi(s, th):
    """EN 1993-1-2 eq. 4.5-4.7, minor axis, gamma_M,fi = 1."""
    ncr = math.pi ** 2 * E * s["I"] / L ** 2
    lam = math.sqrt(s["A"] * FY / ncr)
    lt = lam * math.sqrt(ky(th) / max(kE(th), 1e-12))
    al = 0.65 * math.sqrt(235e6 / FY)
    ph = .5 * (1 + al * lt + lt * lt)
    chi = 1 / (ph + math.sqrt(ph * ph - lt * lt))
    return chi * s["A"] * ky(th) * FY / GAMMA_M_FI


def h_net(tg, th, eps_m=EPS_M):
    return ALPHA_C * (tg - th) + PHI * eps_m * EPS_F * SIGMA * ((tg + 273) ** 4 - (th + 273) ** 4)


def heat(s, dt=2.0, t_end=3600.0, eps_m=EPS_M, ksh=None, amv=None):
    """Steel temperature history, explicit eq. 4.25. Returns (t, theta_a, theta_g)."""
    k = s["ksh"] if ksh is None else ksh
    av = s["AmV"] if amv is None else amv
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    th = np.empty(n + 1); th[0] = 20.0
    tg = np.array([iso834(x) for x in t])
    for i in range(n):
        th[i + 1] = th[i] + k * av / (ca(th[i]) * RHO) * h_net(tg[i], th[i], eps_m) * dt
    return t, th, tg


def failure(s, N, **kw):
    """First time the modelled resistance drops below N (linear interpolation between steps)."""
    t, th, _ = heat(s, **kw)
    r = np.array([n_b_fi(s, x) for x in th])
    idx = np.flatnonzero(r < N)
    if not len(idx):
        return math.inf, float(th[-1])
    i = idx[0]
    if i == 0:
        return 0.0, 20.0
    f = (r[i - 1] - N) / (r[i - 1] - r[i])
    return float(t[i - 1] + f * (t[i] - t[i - 1])), float(th[i - 1] + f * (th[i] - th[i - 1]))


def cold(s):
    return n_b_fi(s, 20.0)


def race(rule, x, S=None, **kw):
    S = S or sections()
    if rule == "A":
        N = {k: x * min(cold(s) for s in S.values()) for k in S}
    else:
        N = {k: x * cold(s) for k, s in S.items()}
    return {k: dict(N=N[k], **dict(zip(("t_s", "theta"), failure(S[k], N[k], **kw)))) for k in S}


# ---------------------------------------------------------------- validation (registration v2)

def v_closed_form():
    """Integrator vs an exact solution: constant gas temperature, linear h, constant c_a."""
    s = dict(ksh=1.0, AmV=150.0)
    tg, h, c = 800.0, 40.0, 600.0
    a = s["AmV"] * h / (c * RHO)
    dt, n = 2.0, 900
    th = 20.0
    for _ in range(n):
        th += a * (tg - th) * dt
    exact = tg - (tg - 20.0) * math.exp(-a * dt * n)
    # explicit Euler error is first order; the gate is on the production step
    return abs(th - exact) / (exact - 20.0)


def v_stocky_eq422(mu0):
    """With buckling removed (huge I), the failure temperature must match EN eq. 4.22."""
    s = dict(sections()["box"]); s["I"] = 1e3
    N = mu0 * n_b_fi(s, 20.0)
    lo, hi = 20.0, 1199.0
    for _ in range(80):
        mid = .5 * (lo + hi)
        lo, hi = (mid, hi) if n_b_fi(s, mid) >= N else (lo, mid)
    eq = 39.19 * math.log(1 / (0.9674 * mu0 ** 3.833) - 1) + 482
    return lo, eq


def v_radial_rod(t_end, dr_cells=20, dt=0.02):
    """Independent check for the thickest section: 1-D radial conduction in the solid rod with the
    same boundary heat flux, temperature-dependent conductivity lambda_a = 54 - 3.33e-2 theta
    (EN 1993-1-2 3.4.1.3). Returns (mean temperature, centre, surface) at t_end.
    Explicit stability: dr = R/20 = 1.78 mm, alpha <= 54/(7850*425) = 1.6e-5 m^2/s, so
    dr^2/(4 alpha) = 0.05 s > dt = 0.02 s."""
    cav = np.vectorize(ca)
    R = sections()["solid_circle"]["outer"] / 2
    dr = R / dr_cells
    r_c = (np.arange(dr_cells) + .5) * dr
    r_f = np.arange(dr_cells + 1) * dr
    T = np.full(dr_cells, 20.0)
    vol = math.pi * (r_f[1:] ** 2 - r_f[:-1] ** 2)
    t = 0.0
    while t < t_end - 1e-9:
        lam = np.maximum(54 - 3.33e-2 * T, 27.3)
        lf = 0.5 * (lam[1:] + lam[:-1])
        q_in = np.zeros(dr_cells)
        flux = lf * (T[1:] - T[:-1]) / dr * 2 * math.pi * r_f[1:-1]    # W per m length, outward-positive
        q_in[:-1] += flux
        q_in[1:] -= flux
        q_in[-1] += h_net(iso834(t), T[-1]) * 2 * math.pi * R
        cap = cav(T) * RHO * vol
        T = T + q_in / cap * dt
        t += dt
    mean = float((T * vol).sum() / vol.sum())
    return mean, float(T[0]), float(T[-1])


def run():
    S = sections()
    out = {"edition": EDITION, "gamma_M_fi": GAMMA_M_FI, "E_pa": E, "fy_pa": FY, "length_m": L,
           "sections": {}, "cold": {}, "parent_cold_N": PARENT_COLD_N}
    for k, s in S.items():
        c, why = fire_class(s, k)
        out["sections"][k] = dict(A=s["A"], I=s["I"], AmV=s["AmV"], ksh=s["ksh"],
                                  ksh_AmV=s["ksh"] * s["AmV"], outer=s["outer"], wall=s["wall"],
                                  fire_class=c, class_note=why)
        out["cold"][k] = cold(s)
    out["cold_ranking_EN"] = sorted(KEYS, key=lambda k: -out["cold"][k])
    out["cold_ranking_parent"] = sorted(KEYS, key=lambda k: -PARENT_COLD_N[k])
    # ---- the sweep that decides the claims
    fracs = (0.3, 0.4, 0.5, 0.6, 0.7)
    variants = {"base": {}, "eps_m_0.8": dict(eps_m=0.8)}
    sweep = {}
    for rule in ("A", "B"):
        for x in fracs:
            for vname, kw in variants.items():
                for ksh_I in (None, 1.0):
                    res = {}
                    for k, s in S.items():
                        kk = dict(kw)
                        if k == "i_section" and ksh_I is not None:
                            kk["ksh"] = ksh_I
                        N = (x * min(cold(q) for q in S.values())) if rule == "A" else x * cold(s)
                        t2 = failure(s, N, dt=2.0, **kk)[0]
                        t05 = failure(s, N, dt=0.5, **kk)[0]
                        t5 = failure(s, N, dt=5.0, **kk)[0]
                        res[k] = dict(t2=t2, t05=t05, t5=t5, u=abs(t2 - t05) / t05)
                    sweep[f"{rule}|{x}|{vname}|ksh_I={ksh_I or 'EN'}"] = res
    out["sweep"] = sweep

    def gap(res, a, b):          # relative lead of a over b, on the finest step
        return res[a]["t05"] / res[b]["t05"] - 1

    claims = {"A_rod_outlasts_I": [], "B_rod_outlasts_all": []}
    for key, res in sweep.items():
        u = 3 * max(r["u"] for r in res.values())
        if key.startswith("A|"):
            claims["A_rod_outlasts_I"].append((key, gap(res, "solid_circle", "i_section"), u))
        else:
            nxt = max((k for k in KEYS if k != "solid_circle"), key=lambda k: res[k]["t05"])
            claims["B_rod_outlasts_all"].append((key, gap(res, "solid_circle", nxt), u))
    out["claims"] = {name: dict(min_gap=min(g for _, g, _ in rows),
                                max_bar=max(max(.10, u) for _, _, u in rows),
                                holds=all(g > max(.10, u) for _, g, u in rows),
                                rows=rows) for name, rows in claims.items()}
    # ---- validation
    stocky = {str(m): v_stocky_eq422(m) for m in (0.3, 0.5, 0.7)}
    radial_t = race("B", 0.5, S)["solid_circle"]["t_s"]
    mean, centre, surface = v_radial_rod(radial_t)
    lumped = float(np.interp(radial_t, *heat(S["solid_circle"], dt=0.5)[:2]))
    wrong = dict(S); wrong["solid_circle"] = dict(S["solid_circle"], AmV=S["box"]["AmV"])
    rb_wrong = race("B", 0.5, wrong)
    out["validation"] = {
        "closed_form_rel_err_dt2": v_closed_form(),
        "stocky_vs_eq422": {m: dict(model=v[0], eq422=v[1], rel=abs(v[0] - v[1]) / v[1]) for m, v in stocky.items()},
        "radial_rod_at_ruleB_failure": dict(t_s=radial_t, lumped=lumped, radial_mean=mean,
                                             centre=centre, surface=surface,
                                             rel=abs(lumped - mean) / (mean - 20)),
        "wrong_AmV_flips_ruleB_winner": max(rb_wrong, key=lambda k: rb_wrong[k]["t_s"]) != "solid_circle",
        "classes": {k: out["sections"][k]["fire_class"] for k in KEYS},
    }
    v = out["validation"]
    out["validation_pass"] = (v["closed_form_rel_err_dt2"] < 0.005
                              and all(x["rel"] < 0.02 for x in v["stocky_vs_eq422"].values())
                              and v["radial_rod_at_ruleB_failure"]["rel"] < 0.02
                              and v["wrong_AmV_flips_ruleB_winner"]
                              and all(c <= 3 for c in v["classes"].values()))
    out["nominal"] = {"A": race("A", 0.5, S, dt=0.5), "B": race("B", 0.5, S, dt=0.5)}
    return out


MAG = 10.0          # constant display magnification of the computed sway (registration v2 item 8)
E0 = L / 1000.0     # initial bow amplitude used only for the drawn sway


def sway(s, N, th):
    """delta = e0 / (1 - N / Ncr,theta); None once N >= Ncr,theta (never reached before the resistance crossing)."""
    ncr = kE(th) * math.pi ** 2 * E * s["I"] / L ** 2
    if N >= ncr:
        return None
    return E0 / (1.0 - N / ncr)


def frames(o, dt=0.5, t_end=1800.0):
    """Common-clock arrays for the renderer: gas temperature, each section's steel temperature."""
    S = sections()
    t, _, tg = heat(S["box"], dt=dt, t_end=t_end)
    out = {"time_s": t, "gas_C": tg}
    for k, s in S.items():
        out[f"{k}_steel_C"] = heat(s, dt=dt, t_end=t_end)[1]
    return out


if __name__ == "__main__":
    o = run()
    d = ROOT / "output_spinoff" / "Spinoff_064" / "data"
    d.mkdir(parents=True, exist_ok=True)
    (d / "probe_v2.json").write_text(json.dumps(o, indent=1, default=float), encoding="utf-8")
    res = {k: o[k] for k in ("edition", "gamma_M_fi", "E_pa", "fy_pa", "length_m", "sections", "cold",
                               "parent_cold_N", "cold_ranking_EN", "cold_ranking_parent", "nominal",
                               "validation", "validation_pass")}
    res["claims"] = {k: {kk: v[kk] for kk in ("min_gap", "max_bar", "holds")} for k, v in o["claims"].items()}
    res["display"] = {"MAG": MAG, "E0_m": E0}
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    np.savez_compressed(d / "frames.npz", **frames(o))
    print("cold EN kN", {k: round(v / 1e3, 1) for k, v in o["cold"].items()}, o["cold_ranking_EN"])
    print("parent kN", {k: round(v / 1e3, 1) for k, v in PARENT_COLD_N.items()}, o["cold_ranking_parent"])
    print("classes", {k: (s["fire_class"], s["class_note"]) for k, s in o["sections"].items()})
    print("AmV", {k: round(s["AmV"], 1) for k, s in o["sections"].items()}, "ksh I", round(o["sections"]["i_section"]["ksh"], 3))
    for r in ("A", "B"):
        print(r, {k: round(v["t_s"] / 60, 2) for k, v in o["nominal"][r].items()})
    for name, c in o["claims"].items():
        print(name, "min gap", round(c["min_gap"], 3), "bar", round(c["max_bar"], 3), "HOLDS" if c["holds"] else "FAILS")
    print("validation", json.dumps(o["validation"], default=float)[:900])
    print("validation_pass", o["validation_pass"])
