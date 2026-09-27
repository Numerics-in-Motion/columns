# -*- coding: utf-8 -*-
"""Evaluate every registered brace cell from five complete batches.

This code never selects a condition by outcome. Missing cells fail the release.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brace_batches import AMPS, DATA, FACTORS, HEIGHTS, PHASES, key, load_done
from brace_study import NAMES, PARENT, column, solve
from column_brace import strut_resistance, strut_stiffness
from materials import Material
from brace_study import config


def interval(num, num_e, den, den_e):
    if not 0 <= num_e < 1 or not 0 <= den_e < 1:
        raise ValueError("invalid relative numerical envelope")
    r = num / den
    return dict(value=r, low=r * (1-num_e) / (1+den_e),
                high=r * (1+num_e) / (1-den_e))


def adverse(one, one_e, two, two_e, den, den_e):
    return dict(value=min(one,two)/den,
                low=min(one*(1-one_e), two*(1-two_e))/(den*(1+den_e)),
                high=min(one*(1+one_e), two*(1+two_e))/(den*(1-den_e)),
                governing="one_wave" if one <= two else "two_wave")


def separated(a, b):
    """True only if both the ratio intervals and 3% point-estimate rule resolve."""
    return a["low"] > b["high"] and a["value"] > b["value"] * 1.03


def value(r):
    if r.get("mode") != "yield" or "first_yield_n" not in r:
        raise ValueError(f"no bracketed first yield: {r.get('section')} {r.get('mode')}")
    lo, hi = r["bracket_n"]
    if not lo <= r["first_yield_n"] <= hi:
        raise ValueError("interpolated first yield outside bracket")
    return float(r["first_yield_n"])


def estimate(rows, k):
    p = rows["production"][k]
    P = value(p)
    controls = ("mesh", "steps", "tol_lo", "tol_hi")
    pc = {x: value(rows[x][k]) for x in controls}
    e_mesh = abs(P-pc["mesh"]) / P
    e_steps = abs(P-pc["steps"]) / P
    e_tol = max(abs(P-pc["tol_lo"]), abs(P-pc["tol_hi"])) / P
    e_rung = float(p["rung_n"]) / P
    e = e_mesh + e_steps + e_tol + e_rung
    return dict(P=P, e=e, components=dict(mesh=e_mesh,steps=e_steps,
                                           tol=e_tol,rung=e_rung),
                strut_util=p.get("brace_utilisation"))


def condition(rows, amp, height, factor):
    out = {}
    for name in NAMES:
        b = estimate(rows, key((name,amp,0.5,0.0,"one_wave")))
        o = estimate(rows, key((name,amp,height,factor,"one_wave")))
        t = estimate(rows, key((name,amp,height,factor,"two_wave")))
        out[name] = dict(bare=b, braced_one=o, braced_two=t,
                         R=interval(o["P"],o["e"],b["P"],b["e"]),
                         Rw=adverse(o["P"],o["e"],t["P"],t["e"],b["P"],b["e"]))
    return out


def extrema(cell, quantity):
    order = sorted(NAMES, key=lambda n: cell[n][quantity]["value"])
    low_name, high_name = order[0], order[-1]
    high = cell[high_name][quantity]
    low = cell[low_name][quantity]
    resolved_high = all(separated(high, cell[n][quantity]) for n in NAMES
                        if n != high_name)
    resolved_low = all(separated(cell[n][quantity], low) for n in NAMES
                       if n != low_name)
    return dict(leader=high_name, lagger=low_name,
                contrast=high["value"]/low["value"],
                contrast_low=high["low"]/low["high"],
                resolved_leader=resolved_high, resolved_lagger=resolved_low,
                order=order)


def run():
    rows = {p: load_done(DATA / f"{p}.jsonl") for p in PHASES}
    if any(len(r) != 240 for r in rows.values()):
        raise RuntimeError("five complete 240-cell batches required: " +
                           str({p:len(r) for p,r in rows.items()}))
    parent = json.loads(PARENT.read_text(encoding="utf-8"))["real_N"]
    parent_check = {n:solve(n,historical=True) for n in NAMES}
    parent_rel = {n:abs(parent_check[n]["first_crossing_n"]-parent[n])/parent[n]
                  for n in NAMES}
    gates = {"parent_identity":max(parent_rel.values()) < 0.001}

    cells = []
    failures = []
    for amp in AMPS:
        for height in HEIGHTS:
            for factor in FACTORS:
                try:
                    values = condition(rows,amp,height,factor)
                    er = extrema(values,"R")
                    ew = extrema(values,"Rw")
                    cell = dict(amp=amp,height=height,factor=factor,
                                values=values,extremes=er,adverse_extremes=ew)
                    cells.append(cell)
                except (ValueError,KeyError,ZeroDivisionError) as exc:
                    failures.append(dict(amp=amp,height=height,factor=factor,
                                         reason=str(exc)))
    gates["complete_grid"] = len(cells)==27 and not failures
    nominal = next((c for c in cells if c["amp"]==0.001 and
                    c["height"]==0.5 and c["factor"]==1.0),None)
    if nominal:
        u = max(max(1-x[q]["low"]/x[q]["value"],
                    x[q]["high"]/x[q]["value"]-1)
                for x in nominal["values"].values() for q in ("R","Rw"))
        gates["numerical_u"] = u <=0.03
        gates["benefit_positive"] = all(x["R"]["low"] >1 for x in
                                         nominal["values"].values())
        e = nominal["extremes"]
        w = nominal["adverse_extremes"]
        gates["nominal_contrast"] = e["contrast_low"] >=1.50
        gates["nominal_extremes"] = e["resolved_leader"] and e["resolved_lagger"]
        gates["adverse_extremes"] = (w["leader"]==e["leader"] and
                                      w["lagger"]==e["lagger"] and
                                      w["resolved_leader"] and w["resolved_lagger"])
        gates["sensitivity_extremes"] = all(
            c["extremes"]["leader"]==e["leader"] and
            c["extremes"]["lagger"]==e["lagger"] and
            c["extremes"]["resolved_leader"] and
            c["extremes"]["resolved_lagger"] and
            c["adverse_extremes"]["leader"]==e["leader"] and
            c["adverse_extremes"]["lagger"]==e["lagger"] and
            c["adverse_extremes"]["resolved_leader"] and
            c["adverse_extremes"]["resolved_lagger"]
            for c in cells)
        print("nominal u",u,"C_low",e["contrast_low"],
              "extremes",e["leader"],e["lagger"])
    else:
        gates.update({k:False for k in ("numerical_u","benefit_positive",
                                        "nominal_contrast","nominal_extremes",
                                        "adverse_extremes","sensitivity_extremes")})
        u=None
    mat=Material.from_config(config()["material"])
    cols={n:column(n) for n in NAMES}
    gates["equal_steel"] = (max(x.area for x in cols.values())-
                             min(x.area for x in cols.values()) <1e-12 and
                             all(abs(x.area-0.004)<1e-12 for x in cols.values()))
    gates["identical_brace"] = abs(strut_stiffness(mat.E)-126666666.66666667)<1e-3
    util=[r.get("brace_utilisation",0.0) for r in rows["production"].values()
          if r["brace_factor"]>0]
    gates["strut_under_80pct"] = bool(util) and max(util)<0.8
    validation_path=DATA/"validation.json"
    if not validation_path.exists():
        raise RuntimeError("independent mechanical validation has not run")
    validation=json.loads(validation_path.read_text(encoding="utf-8"))
    for k,passed in validation["gates"].items():
        gates["mechanics_"+k]=bool(passed)
    no_go_keys=("parent_identity","complete_grid","numerical_u",
                "benefit_positive","nominal_contrast","nominal_extremes",
                "adverse_extremes","sensitivity_extremes",
                "strut_under_80pct","mechanics_bare_euler",
                "mechanics_ideal_brace_euler","mechanics_yield_equilibrium")
    no_go={k:gates[k] for k in no_go_keys}
    doc=dict(registration="reference/original_registration_v3.md",
             parent_relative=parent_rel,nominal=nominal,
             sensitivity=[dict(amp=c["amp"],height=c["height"],factor=c["factor"],
                               extremes=c["extremes"],
                               adverse_extremes=c["adverse_extremes"])
                          for c in cells], failures=failures,u=u,
             max_strut_util=max(util) if util else None,
             gates=gates,no_go=no_go,all_pass=all(gates.values()))
    target=DATA/"results.json"
    target.write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    for name,passed in gates.items():
        print("PASS" if passed else "FAIL",name)
    print("wrote",target,"all_pass",doc["all_pass"])
    return 0 if doc["all_pass"] else 2


if __name__=="__main__":
    raise SystemExit(run())
