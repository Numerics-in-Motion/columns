# -*- coding: utf-8 -*-
"""Separate v4 release decision; NEVER overwrites the failed v3 result."""
from __future__ import annotations

import json

from brace_holdout import CONDITIONS, OUT, key, load
from brace_batches import DATA, PHASES
from brace_results import adverse, estimate, interval, separated
from brace_study import NAMES


def holdout_cell(rows, condition):
    amp, height, factor = condition
    values = {}
    for name in NAMES:
        bare = estimate(rows, key((name, amp, 0.5, 0., "one_wave")))
        one = estimate(rows, key((name, amp, height, factor, "one_wave")))
        two = estimate(rows, key((name, amp, height, factor, "two_wave")))
        values[name] = dict(
            bare=bare, braced_one=one, braced_two=two,
            R=interval(one["P"], one["e"], bare["P"], bare["e"]),
            Rw=adverse(one["P"], one["e"], two["P"], two["e"],
                       bare["P"], bare["e"]),
        )
    return dict(amp=amp, height=height, factor=factor, values=values)


def leader_resolved(cell, quantity):
    values = cell["values"]
    rod = values["solid_circle"][quantity]
    return all(separated(rod, values[n][quantity]) for n in NAMES
               if n != "solid_circle")


def uncertainty(cell):
    return max(max(1 - v[q]["low"] / v[q]["value"],
                   v[q]["high"] / v[q]["value"] - 1)
               for v in cell["values"].values() for q in ("R", "Rw"))


def main():
    old = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    rows = {p: load(OUT / f"{p}.jsonl") for p in PHASES}
    gates = {
        "v3_preserved_failed": not old["all_pass"] and
             not old["gates"]["adverse_extremes"] and
             not old["gates"]["sensitivity_extremes"],
        "old_physical_numerical": all(v for k, v in old["gates"].items()
                                      if k not in ("adverse_extremes",
                                                   "sensitivity_extremes")),
        "old_nominal_material_contrast":
            old["nominal"]["extremes"]["contrast_low"] >= 1.5,
        "complete_new_phases": all(len(r) == 24 for r in rows.values()),
        "old_nominal_rod_leader_R": leader_resolved(old["nominal"], "R"),
        "old_nominal_rod_leader_Rw": leader_resolved(old["nominal"], "Rw"),
        "old_grid_rod_leader_R": all(
            c["extremes"]["leader"] == "solid_circle" and
            c["extremes"]["resolved_leader"]
            for c in old["sensitivity"]),
        "old_grid_rod_leader_Rw": all(
            c["adverse_extremes"]["leader"] == "solid_circle" and
            c["adverse_extremes"]["resolved_leader"]
            for c in old["sensitivity"]),
    }
    cells = []
    failures = []
    if gates["complete_new_phases"]:
        for condition in CONDITIONS:
            try:
                cells.append(holdout_cell(rows, condition))
            except (ValueError, KeyError, ZeroDivisionError) as exc:
                failures.append(dict(condition=condition, reason=str(exc)))
    gates["new_yield_complete"] = len(cells) == 2 and not failures
    gates["new_rod_leader_R"] = gates["new_yield_complete"] and all(
        leader_resolved(c, "R") for c in cells)
    gates["new_rod_leader_Rw"] = gates["new_yield_complete"] and all(
        leader_resolved(c, "Rw") for c in cells)
    gates["new_numerical_u"] = gates["new_yield_complete"] and all(
        uncertainty(c) <= 0.03 for c in cells)
    gates["new_matched_benefits"] = gates["new_yield_complete"] and all(
        v["R"]["low"] > 1 for c in cells for v in c["values"].values())
    gates["new_rod_adverse_above_one"] = gates["new_yield_complete"] and all(
        c["values"]["solid_circle"]["Rw"]["low"] > 1 for c in cells)
    braced = [r for phase in rows.values() for r in phase.values()
              if r["brace_factor"] > 0]
    gates["new_strut_under_80pct"] = (len(braced) == 80 and
        all(r.get("brace_utilisation", 1) < 0.8 for r in braced))
    result = dict(
        status="new post-hoc narrowed study with prospective parameter checks",
        old_v3_failed=True, old_v3_file=str(DATA / "results.json"),
        registration="reference/redesign_registration_v4.md",
        conditions=cells, failures=failures,
        uncertainty_by_condition=[uncertainty(c) for c in cells],
        gates=gates, all_pass=all(gates.values()),
    )
    dest = DATA / "redesign_v4_results.json"
    dest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for k, v in gates.items():
        print("PASS" if v else "FAIL", k)
    print("wrote", dest, "all_pass", result["all_pass"])
    return 0 if result["all_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
