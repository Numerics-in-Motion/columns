"""The strut's 44.7 kg: brace the 12 m column, or build it wider or thicker?

    python reproduce.py            # every solve (nominal, sweep, mesh 80, identities), well under a minute on one core
    python reproduce.py --quick    # the four nominal solves only

Checks every source against the SHA-256 frozen with the reference, re-runs the study, and fails if a published number moved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "solver"))
sys.path.insert(0, RIG)
os.environ["BRACE_CONFIG"] = os.path.join(HERE, "config", "column_config.json")
os.environ["BRACE_DATA"] = tempfile.mkdtemp(prefix="strut-")
TOL = 1e-6          # another machine's BLAS may differ in the last digits


def sha(p):
    with open(p, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    ref = json.load(open(os.path.join(HERE, "reference", "result.json"), encoding="utf-8"))
    for rel, want in ref["source_sha256"].items():
        if sha(os.path.join(HERE, rel)) != want:
            raise SystemExit(f"source changed: {rel}")
    import strut_vs_section as SV
    if a.quick:
        got = {k: SV.first_yield(k)["first_yield_n"] for k in ("bare", "strut", "wider", "thicker")}
    else:
        o = SV.run(tempfile.mkdtemp(prefix="strut-out-"), frames=False)
        got = o["first_yield_n"]
        bad = [k for k, v in ref["u"].items() if abs(o["u"][k] - v) > 1e-6]
        if bad or o["claims"] != ref["claims"] or not o["validation_pass"]:
            raise SystemExit(f"MISMATCH: u {bad} claims {o['claims']} validation {o['validation_pass']}")
        for g in o["gaps"]:
            print(f"{g['pair'][0]} -> {g['pair'][1]}: gap {g['gap']:.1%} > max(10%, 3u) = {g['threshold']:.1%} -> holds")
    for k, v in ref["first_yield_n"].items():
        if abs(got[k] / v - 1) > TOL:
            raise SystemExit(f"MISMATCH: {k} {got[k]} vs {v}")
    for k in ("strut", "wider", "thicker"):
        print(f"{k:8s} first yield {got[k] / 1e3:6.1f} kN  (x{got[k] / got['bare']:.2f} over the bare {got['bare'] / 1e3:.1f} kN)")
    print("reproduced" + (" (quick)" if a.quick else "; all validation checks pass"))


if __name__ == "__main__":
    main()
