"""The strut's 44.7 kg of steel: brace the 12 m column, or build it wider or thicker?

037's column (12.0 m, pinned, 170 x 170 x 6.10 mm SHS, A = 0.004 m^2, f_y 250 MPa, L/1000 crookedness) carries 229.2 kN to
first yield; one 3.0 m 100 x 100 x 5 SHS strut at mid-height to an EXISTING RIGID ANCHOR lifts that to 619.3 kN (037).
The strut's modelled member mass is 44.7 kg (A_s L_s = 0.0057 m^3). Spend the same 0.0057 m^3 on the column instead:

  strut    the canonical section + the strut (037's detail, the worse of the two crookedness shapes)
  wider    a wider tube, same 6.10 mm wall:  A + dA, outer width solved from A = 4tB - 4t^2
  thicker  a thicker wall, same 170 mm outside: A + dA, wall solved by section_properties.box_properties
  (bare, 229.2 kN, is drawn as 037's reference only -- not a contender)

Endpoint: axial load at FIRST YIELD of the extreme fibre (the published `first_yield_load`); principal-plane model; an
eigenvalue "instability" return is never counted as first yield. Solver: `column_brace.solve` (037, published, unchanged);
only `column_brace.section` is swapped for the two section contenders. Anchor and connection steel are not counted; installed
cost is not compared; wider/thicker are alternative column designs, not retrofits.

u (relative half-range, quadrature): crookedness amplitude L/2000 vs L/500, yield stress 225 vs 275 MPa, strut spring x0.5 vs
x2 (neighbour kept rigid; saturated band). Each adjacent gap must exceed max(10 %, 3 max(u_a, u_b)).
"""
from __future__ import annotations

import os

# one BLAS thread (set before numpy loads): the solves are small, and a full thread pool per solve made the first full
# run take 139 min instead of minutes (the same reason 037's run_study pins it)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import column_brace as CB
import section_properties as SP

ROOT = Path(__file__).resolve().parents[1]
ORDER = ("strut", "wider", "thicker")
PUBLISHED_037 = {"bare": 229.2e3, "braced": 619.3e3}


def added_area():
    s = CB.strut()
    return s["area"] * s["length"] / CB.LENGTH


def strut_mass_kg(rho=7850.0):
    s = CB.strut()
    return s["area"] * s["length"] * rho


def sections():
    base = CB.section()
    dA = added_area()
    A = CB.AREA + dA
    t0 = base["t"]
    B_wide = (A + 4 * t0 * t0) / (4 * t0)
    return {"bare": base, "strut": base, "wider": SP.box_properties(A, B_wide),
            "thicker": SP.box_properties(A, CB.OUTER)}


@contextmanager
def using_section(sec):
    original = CB.section
    CB.section = lambda: sec
    try:
        yield
    finally:
        CB.section = original


def cfg_with_fy(fy):
    c = copy.deepcopy(CB.config())
    c["material"]["failure_stress_pa"] = float(fy)
    return c


def first_yield(name, imperfection=CB.NOMINAL_IMPERFECTION, fy=250e6, spring=1.0, n=CB.NOMINAL_MESH, keep=False):
    """Worse of the two crookedness shapes for the braced column (037's rule); one wave for an unbraced column."""
    cfg = cfg_with_fy(fy)
    sec = sections()[name]
    k = spring * CB.strut_stiffness(CB.material(cfg).E) if name == "strut" else 0.0
    shapes = CB.SHAPES if name == "strut" else ("one_wave",)
    runs = []
    with using_section(sec):
        for sh in shapes:
            runs.append(CB.solve(n_elements=n, imperfection=imperfection, shape=sh, k=k, cfg=cfg, keep=keep))
    good = [r for r in runs if r["reached_yield"]]
    if not good:
        raise RuntimeError(f"{name}: no first-yield event (modes {[r['mode'] for r in runs]})")
    return min(good, key=lambda r: r["first_yield_n"])


def wall_class(sec, fy=250e6):
    eps = math.sqrt(235e6 / fy)
    c_t = (sec["B"] - 3 * sec["t"]) / sec["t"]
    cls = 1 if c_t <= 33 * eps else 2 if c_t <= 38 * eps else 3 if c_t <= 42 * eps else 4
    return dict(c_over_t=c_t, limit_class1=33 * eps, cls=cls)


def run(out_dir, frames=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    S = sections()
    nominal = {k: first_yield(k, keep=frames) for k in ("bare",) + ORDER}
    P = {k: nominal[k]["first_yield_n"] for k in nominal}
    sweep, u = {}, {}
    for k in ORDER:
        imp = (first_yield(k, imperfection=1 / 2000)["first_yield_n"], first_yield(k, imperfection=1 / 500)["first_yield_n"])
        fy = (first_yield(k, fy=225e6)["first_yield_n"], first_yield(k, fy=275e6)["first_yield_n"])
        rows = {"imperfection": imp, "fy": fy}
        if k == "strut":
            rows["spring"] = (first_yield(k, spring=0.5)["first_yield_n"], first_yield(k, spring=2.0)["first_yield_n"])
        sweep[k] = rows
        u[k] = math.sqrt(sum((abs(b - a) / 2 / P[k]) ** 2 for a, b in rows.values()))
    order = sorted(ORDER, key=lambda k: -P[k])
    gaps = []
    for a, b in zip(order, order[1:]):
        g = (P[a] - P[b]) / P[b]
        th = max(0.10, 3 * max(u[a], u[b]))
        gaps.append(dict(pair=[a, b], gap=g, threshold=th, passes=g > th))
    mesh80 = {k: first_yield(k, n=80)["first_yield_n"] for k in ("bare",) + ORDER}
    ident = CB.identities()
    classes = {k: wall_class(S[k]) for k in ("strut", "wider", "thicker")}
    brackets = {k: all(e["ok"] for e in nominal[k].get("equilibrium", [])) and nominal[k]["bracket"]["ok"] for k in nominal}
    validation = {
        "reproduces_037_bare": abs(P["bare"] - PUBLISHED_037["bare"]) / PUBLISHED_037["bare"] < 2e-3,
        "reproduces_037_braced": abs(P["strut"] - PUBLISHED_037["braced"]) / PUBLISHED_037["braced"] < 2e-3,
        "mesh80_within_0p5pct": all(abs(mesh80[k] - P[k]) / P[k] < 5e-3 for k in mesh80),
        "identities_ok": all(abs(v["bare_over_euler"] - 1) < 0.01 and abs(v["ideal_over_4euler"] - 1) < 0.02
                             for v in ident.values()),
        "yield_brackets_ok": all(brackets.values()),
        "walls_class1": all(c["cls"] == 1 for c in classes.values()),
        "equal_added_steel": abs(S["wider"]["area"] - S["thicker"]["area"]) < 1e-12
                             and abs(S["wider"]["area"] - CB.AREA - added_area()) < 1e-12,
    }
    validation["all"] = all(validation.values())
    claims = {"order_strut_wider_thicker": order == ["strut", "wider", "thicker"],
              "gaps_exceed_max_10pct_3u": all(g["passes"] for g in gaps)}
    res = {
        "length_m": CB.LENGTH, "fy_pa": 250e6, "imperfection": CB.NOMINAL_IMPERFECTION,
        "added_area_m2": added_area(), "added_mass_kg": strut_mass_kg(),
        "sections": {k: {kk: float(v) for kk, v in S[k].items() if isinstance(v, (int, float))} for k in S},
        "first_yield_n": P, "gain_over_bare": {k: P[k] / P["bare"] for k in ORDER},
        "governing_shape": {k: nominal[k]["shape"] for k in nominal},
        "brace_utilisation": nominal["strut"].get("brace_utilisation"),
        "sweep": sweep, "u": u, "gaps": gaps, "order": order, "mesh80": mesh80,
        "identities": {str(k): v for k, v in ident.items()}, "wall_classes": classes,
        "validation": validation, "validation_pass": validation["all"], "claims": claims,
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    if frames:
        fr = {}
        for k in ("bare",) + ORDER:
            r, col = nominal[k]["_result"], nominal[k]["_column"]
            loads = np.asarray(r.load_history)
            keep = loads <= P[k] * 1.0000001
            disp = np.asarray([d[0::3] for d in r.nodal_disp_history])[keep]
            fr[f"{k}_load_n"] = loads[keep]
            fr[f"{k}_sway_m"] = disp
            fr[f"{k}_stress_pa"] = np.asarray(r.max_combined_stress_history)[keep]
            fr[f"{k}_y_m"] = np.asarray(col.nodes[:, 1])
            fr[f"{k}_x0_m"] = np.asarray(col.nodes[:, 0])
        np.savez_compressed(out_dir / "frames.npz", **fr)
    return res


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "out"
    r = run(target)
    print(json.dumps({"first_yield_kN": {k: round(v / 1e3, 1) for k, v in r["first_yield_n"].items()},
                      "u": {k: round(v, 3) for k, v in r["u"].items()},
                      "gaps": [(g["pair"], round(g["gap"], 3), round(g["threshold"], 3), g["passes"]) for g in r["gaps"]],
                      "validation": r["validation"], "claims": r["claims"],
                      "walls": {k: (round(v["c_over_t"], 1), v["cls"]) for k, v in r["wall_classes"].items()},
                      "mesh80_kN": {k: round(v / 1e3, 1) for k, v in r["mesh80"].items()}}, indent=1))
