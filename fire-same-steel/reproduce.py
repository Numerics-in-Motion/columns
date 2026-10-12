"""Reproduce the four-column standard-fire comparison (Spinoff 064) from frozen sources.

    python reproduce.py

Re-runs the full sweep (both load rules, five load levels, two emissivities, two I-section shadow
factors, three time steps), the four validation checks, and compares every published number with
reference/result.json. Exits nonzero, naming the quantity, if anything disagrees or a source changed.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "solver"))


def sha(p):
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()   # line-ending independent


def main():
    ref = json.loads((HERE / "reference" / "result.json").read_text(encoding="utf-8"))
    for rel, want in ref["source_sha256"].items():
        if sha(HERE / rel) != want:
            raise SystemExit(f"source changed: {rel}")
    import fire_columns064 as fc
    o = fc.run()
    bad = []
    for rule in ("A", "B"):
        for k, v in ref["nominal"][rule].items():
            got = o["nominal"][rule][k]["t_s"]
            if abs(got - v) > 1e-6 * max(1.0, v):
                bad.append(f"rule {rule} {k}: {got} vs {v}")
    for name, c in ref["claims"].items():
        if o["claims"][name]["holds"] != c["holds"] or abs(o["claims"][name]["min_gap"] - c["min_gap"]) > 1e-9:
            bad.append(f"claim {name}")
    if not o["validation_pass"]:
        bad.append("validation")
    if bad:
        raise SystemExit("MISMATCH: " + "; ".join(bad))
    for rule, txt in (("B", "each at half its own cold resistance"), ("A", "the same load on all four")):
        print(f"rule {rule} ({txt}):", ", ".join(f"{k} {v / 60:.1f} min" for k, v in ref["nominal"][rule].items()))
    for name, c in o["claims"].items():
        print(f"{name}: smallest gap {c['min_gap']:.1%} > bar {c['max_bar']:.0%} -> holds")
    print("all validation checks pass; reproduced", ref["edition"])


if __name__ == "__main__":
    main()
