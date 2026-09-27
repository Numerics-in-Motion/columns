# -*- coding: utf-8 -*-
"""Registered equal-area sections-by-brace study. No winner is encoded.

The first mode is an inexpensive screen, not the publishable result:
    python solver/brace_study.py --probe
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

import column_fem as fem
from column_brace import lateral_spring, strut_stiffness, strut_resistance
from column_geometry import build_all_columns
from inverse_design import first_yield_load
from materials import Material

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "config" / "column_config.json"
PARENT = ROOT / "reference" / "parent_first_crossing.json"
NAMES = ("solid_circle", "hollow_tube", "box", "i_section")
PMAX_N = 1_200_000.0


def config(n: int = 12, amp: float = 1 / 1000) -> dict:
    c = json.loads(CFG.read_text(encoding="utf-8"))
    c = copy.deepcopy(c)
    c["geometry"]["n_elements"] = int(n)
    c["geometry"]["imperfection_ratio"] = float(amp)
    return c


def column(name: str, n: int = 12, amp: float = 1 / 1000,
           shape: str = "one_wave"):
    col = next(x for x in build_all_columns(config(n, amp)) if x.name == name)
    if shape == "two_wave":
        y = col.nodes[:, 1]
        L = col.height
        col.nodes[:, 0] = amp * L * np.sin(2 * np.pi * y / L)
    elif shape != "one_wave":
        raise ValueError(shape)
    return col


def solve(name: str, n: int = 12, amp: float = 1 / 1000,
          shape: str = "one_wave", brace_factor: float = 0.0,
          height: float = 0.5, steps: int = 400,
          tol: float = 1e-8, historical: bool = False) -> dict:
    c = config(n, amp)
    mat = Material.from_config(c["material"])
    col = column(name, n, amp, shape)
    node_float = height * n
    if abs(node_float - round(node_float)) > 1e-9:
        raise ValueError(f"brace height {height} is not on {n}-element mesh")
    node = int(round(node_float))
    k = brace_factor * strut_stiffness(mat.E)
    pe = fem.euler_buckling_load(mat.E, col.inertia, col.height, 1.0)
    if historical:
        if brace_factor:
            raise ValueError("historical parent is unbraced")
        max_factor = 1.5
        steps = 400
    else:
        max_factor = PMAX_N / pe
    with lateral_spring(3 * node, k):
        r = fem.run_nonlinear_column(
            col, mat, steps, max_factor, int(c["solver"]["newton_max_iter"]), tol)
    out = {
        "section": name, "shape": shape, "n": n, "amp": amp,
        "brace_factor": brace_factor, "height": height, "steps": steps,
        "mode": r.failure_mode, "first_crossing_n": r.failure_load,
        "load_steps_completed": r.n_load_steps_completed,
        "euler_n": pe,
    }
    if r.failure_mode == "yield" and len(r.load_history) >= 2:
        p = np.asarray(r.load_history, float)
        s = np.asarray(r.max_combined_stress_history, float)
        hits = np.flatnonzero((s[1:] >= mat.failure_stress)
                              & (s[:-1] < mat.failure_stress))
        if len(hits) == 1:
            j = int(hits[0] + 1)
            py = float(first_yield_load(r, mat.failure_stress))
            if p[j - 1] <= py <= p[j]:
                out["first_yield_n"] = py
                out["bracket_n"] = [float(p[j - 1]), float(p[j])]
                out["bracket_stress_pa"] = [float(s[j - 1]), float(s[j])]
                out["rung_n"] = float(p[j] - p[j - 1])
                if k:
                    u = r.nodal_disp_history[j]
                    force = abs(k * float(u[3 * node]))
                    out["brace_force_n"] = force
                    out["brace_utilisation"] = force / strut_resistance(
                        mat.E, mat.failure_stress)
    return out


def probe() -> int:
    pub = json.loads(PARENT.read_text(encoding="utf-8"))["real_N"]
    baseline = {name: solve(name, historical=True) for name in NAMES}
    for name, r in baseline.items():
        rel = abs(r["first_crossing_n"] - pub[name]) / pub[name]
        print(f"parent {name:14s}: {r['first_crossing_n']:11.2f} N "
              f"reference {pub[name]:11.2f} N rel={rel:.4%}")
        if rel > 0.001 or r["mode"] != "yield":
            print("STOP: parent identity failed")
            return 2
    k = strut_stiffness(Material.from_config(config()["material"]).E)
    print(f"identical spring k={k/1e6:.6f} MN/m")
    rows = {}
    for name in NAMES:
        bare = solve(name)
        one = solve(name, brace_factor=1.0)
        two = solve(name, shape="two_wave", brace_factor=1.0)
        rows[name] = {"bare": bare, "one": one, "two": two}
        values = [x.get("first_yield_n") for x in (bare, one, two)]
        if any(x is None for x in values):
            print(f"{name}: non-yield or missing bracket: "
                  f"{bare['mode']}/{one['mode']}/{two['mode']}")
            print("STOP: nominal first-yield condition failed")
            return 3
        p0, p1, p2 = values
        ratio, adverse = p1 / p0, min(p1, p2) / p0
        print(f"{name:14s} bare={p0/1e3:8.2f} kN "
              f"brace1={p1/1e3:8.2f} brace2={p2/1e3:8.2f} "
              f"R={ratio:.4f} Rw={adverse:.4f} "
              f"strut_util={max(one['brace_utilisation'],two['brace_utilisation']):.3%}")
    matched = {name: r["one"]["first_yield_n"] / r["bare"]["first_yield_n"]
               for name, r in rows.items()}
    adverse = {name: min(r["one"]["first_yield_n"],
                         r["two"]["first_yield_n"]) / r["bare"]["first_yield_n"]
               for name, r in rows.items()}
    winner = max(matched, key=matched.get)
    lagger = min(matched, key=matched.get)
    print(f"screening C={matched[winner]/matched[lagger]:.4f} "
          f"matched extremes={winner}/{lagger}; "
          f"adverse extremes={max(adverse,key=adverse.get)}/{min(adverse,key=adverse.get)}")
    print("Screening is NOT the 80-element result or an uncertainty/robustness gate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    if args.probe:
        return probe()
    ap.error("only --probe is implemented; production cannot be inferred from screening")


if __name__ == "__main__":
    raise SystemExit(main())
