"""Reproduce the column-brace calculation with frozen published sources.

    python reproduce.py             # 12 nominal 80-element solves + source checks
    python reproduce.py --full      # every v3 and v4 solve; several minutes

This package uses only its own files and the column solver in the parent
public repository. No private project path or cached numerical answer is used
as a substitute for a solve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
RIG = HERE.parent
sys.path.insert(0, str(RIG))
sys.path.insert(0, str(HERE / "solver"))
os.environ["BRACE_CONFIG"] = str(HERE / "config" / "column_config.json")
os.environ["BRACE_DATA"] = str(HERE / "data")

from brace_study import NAMES, solve  # noqa: E402


def read(name):
    return json.loads((HERE / "reference" / name).read_text(encoding="utf-8"))


def verify_sources():
    hashes = read("source_sha256.json")
    for relative, expected in hashes.items():
        target = (RIG / relative[3:] if relative.startswith("../")
                  else HERE / relative)
        got = hashlib.sha256(target.read_bytes()).hexdigest()
        if got != expected:
            raise RuntimeError(f"source changed: {relative}")
    print(f"verified {len(hashes)} frozen source/config/reference files")


def near(got, want, label, tolerance=0.001):
    rel = abs(got / want - 1)
    if rel > tolerance:
        raise RuntimeError(f"{label} moved by {rel:.4%}: {got} vs {want}")


def quick():
    ref = read("v3_failed_results.json")
    nominal = ref["nominal"]["values"]
    for name in NAMES:
        got = {
            "bare": solve(name, n=80, amp=0.001, shape="one_wave",
                          brace_factor=0., height=.5, steps=2400),
            "braced_one": solve(name, n=80, amp=0.001,
                                shape="one_wave", brace_factor=1.,
                                height=.5, steps=2400),
            "braced_two": solve(name, n=80, amp=0.001,
                                shape="two_wave", brace_factor=1.,
                                height=.5, steps=2400),
        }
        for role, row in got.items():
            if row["mode"] != "yield" or "first_yield_n" not in row:
                raise RuntimeError(f"missing first-yield bracket: {name} {role}")
            near(row["first_yield_n"], nominal[name][role]["P"],
                 f"{name} {role}")
        p0 = got["bare"]["first_yield_n"]
        p1 = got["braced_one"]["first_yield_n"]
        p2 = got["braced_two"]["first_yield_n"]
        print(f"{name:14s} first yield kN: {p0/1000:.2f} -> "
              f"{p1/1000:.2f} / {p2/1000:.2f}; "
              f"R={p1/p0:.3f}, Rw={min(p1,p2)/p0:.3f}")
    if ref["all_pass"] or ref["gates"]["adverse_extremes"]:
        raise RuntimeError("frozen v3 failure has been lost")
    print("quick reproduction PASS; v3 four-way-ranking gate remains FAILED")


def full(workers):
    from brace_batches import PHASES, run as old_run
    from brace_holdout import run as holdout_run
    from brace_validation import run as mechanics_run
    from brace_results import run as old_result
    from brace_redesign_results import main as new_result

    for phase in PHASES:
        old_run(phase, workers)
    mechanics_run()
    old_code = old_result()
    if old_code != 2:
        raise RuntimeError("v3 should FAIL its own four-way-ranking gate")
    for phase in PHASES:
        holdout_run(phase, workers)
    if new_result() != 0:
        raise RuntimeError("v4 leader study did not pass")
    old = json.loads((HERE / "data" / "results.json").read_text())
    new = json.loads((HERE / "data" / "redesign_v4_results.json").read_text())
    old_ref, new_ref = read("v3_failed_results.json"), read("v4_passed_results.json")
    for name in NAMES:
        for role in ("bare", "braced_one", "braced_two"):
            near(old["nominal"]["values"][name][role]["P"],
                 old_ref["nominal"]["values"][name][role]["P"],
                 f"nominal {name} {role}")
    for c, wanted in zip(new["conditions"], new_ref["conditions"]):
        for name in NAMES:
            for role in ("bare", "braced_one", "braced_two"):
                near(c["values"][name][role]["P"],
                     wanted["values"][name][role]["P"],
                     f"holdout {c['amp']} {name} {role}")
    if old["all_pass"] or not new["all_pass"]:
        raise RuntimeError("release status changed")
    print("full reproduction PASS: old v3 failed, narrower v4 passed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--verify-only", action="store_true",
                   help="check frozen source and reference hashes without solves")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    verify_sources()
    if not args.verify_only:
        full(args.workers) if args.full else quick()
