"""One braced level.

    python reproduce.py            # all 199 solves, a few minutes
    python reproduce.py --quick    # the two headline solves only, seconds

WHAT IS CLAIMED

In this principal-plane model of a 12 m square steel tube, the bare one-wave
reference first yielded at 229 kN; with one mid-height lateral brace, the less
favourable of two braced crookedness shapes first yielded at 619 kN. Every
registered crookedness-size, shape, and brace-height case remained above twice
its corresponding one-wave bare reference.

WHAT IS NOT

First yield of the extreme fibre only: the post-yield response was not
computed. One principal bending plane: no out-of-plane response or biaxial
stress interaction was computed. No local buckling of the tube wall, residual
stress or twisting; the strut connection is assumed not to slip and its
neighbour rigid. The strut is about 390 times the closed-form ideal-brace
threshold, so its 0.5x-2x band is a saturated control, not a sensitivity study.

WHAT THIS DOES

Checks every module against the SHA-256 frozen with the reference, re-runs the
tangent identities and every solve, re-applies every gate to the new rows, and
fails if a gated number has moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "solver"))
sys.path.insert(0, RIG)
os.environ["BRACE_CONFIG"] = os.path.join(HERE, "config", "column_config.json")
RUN = tempfile.mkdtemp(prefix="brace-")
os.environ["BRACE_DATA"] = RUN

import column_brace as CB                 # noqa: E402
import column_brace_gates as G            # noqa: E402

REF = os.path.join(HERE, "reference", "reference_canonical.json")
# the same solve on another machine's BLAS may differ in the last digits
TOL = 1e-6


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _where(k):
    p = os.path.join(HERE, "solver", k + ".py")
    return p if os.path.exists(p) else os.path.join(RIG, k + ".py")


def check_sources(ref):
    bad = [k for k, want in ref["source_sha256"].items()
           if sha(_where(k)) != want]
    if sha(os.environ["BRACE_CONFIG"]) != ref["config_sha256"]:
        bad.append("config/column_config.json")
    if bad:
        raise SystemExit("these do not match the reference: %s"
                         % ", ".join(bad))
    print("   %d modules and the config match the reference by SHA-256"
          % len(ref["source_sha256"]))


def moved(got, want):
    return abs(got / want - 1.0)


def quick(ref):
    kb = CB.strut_stiffness(CB.material().E)
    bare = CB.solve(k=0.0, shape="one_wave")["first_yield_n"]
    braced = CB.solve(k=kb, shape="two_waves")["first_yield_n"]
    print("   bare    %.1f kN" % (bare / 1e3))
    print("   braced  %.1f kN  (two half-waves)  x%.3f" % (braced / 1e3,
                                                          braced / bare))
    worst = max(moved(bare, ref["bare_first_yield_n"]),
                moved(braced, ref["braced_first_yield_worse_shape_n"]))
    print("   reproduce the reference to %.1e" % worst)
    return 0 if worst <= TOL else 1


def full(ref):
    CB.main()
    res = G.run(write=True)
    print()
    for g in res["gates"]:
        print("   %-12s %s" % (g["name"], "PASS" if g["passes"] else "FAIL"))
    g = {x["name"]: x for x in res["gates"]}
    ship = g["ship"]
    pairs = [(ship["bare_n"], ref["bare_first_yield_n"]),
             (ship["braced_n"], ref["braced_first_yield_one_wave_n"]),
             (ship["nominal_worse_shape"], ref["ratio_worse_shape"]),
             (ship["min_ratio"], ref["min_ratio"])]
    worst = max(moved(a, b) for a, b in pairs)
    print()
    print("   bare %.1f kN, braced %.1f kN (less favourable shape), x%.3f; smallest "
          "ratio in the sweep x%.3f"
          % (ship["bare_n"] / 1e3, ship["bare_n"] * ship["nominal_worse_shape"]
             / 1e3, ship["nominal_worse_shape"], ship["min_ratio"]))
    print("   the gated numbers reproduce the reference to %.1e" % worst)
    if worst > TOL:
        raise SystemExit("the answer moved against the frozen reference")
    print()
    print("   claim: %s" % res["claim"])
    return 0 if res["passed"] == res["total"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    print("ONE BRACED LEVEL")
    ref = json.load(open(REF, encoding="utf-8"))
    check_sources(ref)
    return quick(ref) if a.quick else full(ref)


if __name__ == "__main__":
    raise SystemExit(main())
