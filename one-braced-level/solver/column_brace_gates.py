# -*- coding: utf-8 -*-
"""One braced level -- the gates.

    python src/column_brace_gates.py      # reads data/column_brace.json, writes gates.json

Every gate reads the study's own rows and checks. The lines are identities,
the solver's own stopping rule or resolution, or the ship line the design
review fixed before any braced result was solved.

  identity_gate     the solver's own tangent: bare = pi^2 EI/L^2, ideal brace
                    = 4 pi^2 EI/L^2, at every mesh, within 0.1 %
  yield_gate        every eligible case reached first yield, and EXACTLY the
                    nine bare two-half-wave controls did not (they go unstable
                    at the Euler load) -- no other exclusion
  bracket_gate      every first-yield load lies between the rungs that bracket
                    yield, in load and in stress
  equilibrium_gate  at both bracketing rungs of every eligible case: free-DOF
                    residual, brace-force closure, sideways reaction closure
                    and vertical closure, all within the Newton stopping bound
  spring_gate       the brace wrapper adds k at one diagonal entry only, its
                    energy is 1/2 k u_b^2, and the published assembly is
                    restored, at every mesh and brace height
  monotone_gate     the solver's buckling load never falls as k rises
  mirror_gate       flipping the crookedness sign changes no first-yield load
  resolution_gate   u = the largest relative change of the braced/bare ratio
                    under mesh (40 -> 80; 20 -> 40 recorded), load-step
                    (600 -> 1200) and Newton-tolerance (1e-8 -> 1e-7, 1e-9)
                    refinement. Must be <= 3 %
  brace_gate        the strut stays below 80 % of its own squash and Euler
                    resistance at the column's first yield, in every case
  ship_gate         min over imperfection size x shape (worse kept) x brace
                    stiffness x brace height of P_y(braced)/P_y(bare) >= 2 + 3u,
                    and the nominal ratio >= 2.5
  wording_gate      nothing shipped says collapse, capacity, failure load,
                    design resistance, "safe", or that the brace works both
                    ways -- the model is planar. The renderer's and the SNS
                    copy's own text is audited by test against FORBIDDEN too.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import column_brace as CB             # noqa: E402

CLAIM = ("In this principal-plane model of a 12 m square steel tube, the bare "
         "one-wave reference first yielded at 229 kN; with one mid-height "
         "lateral brace, the less favourable of two braced crookedness shapes "
         "first yielded at 619 kN. Every registered crookedness-size, shape, "
         "and brace-height case remained above twice its corresponding "
         "one-wave bare reference.")
ON_SCREEN = dict(
    setup="12 m STEEL COLUMN, PINNED AT BOTH ENDS — PRINCIPAL-PLANE MODEL",
    measure="THE NUMBER: THE LOAD WHEN THE STEEL FIRST YIELDS",
    brace="ONE MID-HEIGHT BRACE IN THE SHOWN PLANE: A 3 m STEEL STRUT",
    why="A MID-HEIGHT RESTRAINT SPLITS THE ELASTIC BUCKLING WAVE INTO TWO 6 m HALVES",
    result="MORE THAN DOUBLE ACROSS EVERY CROOKEDNESS SIZE, SHAPE, AND BRACE HEIGHT TRIED",
    braced_note="LESS FAVOURABLE OF TWO BRACED CROOKEDNESS SHAPES",
    why_note="SCHEMATIC ELASTIC BUCKLING SHAPES — AMPLITUDE NOT TO SCALE",
    limits=["FIRST YIELD ONLY: POST-YIELD RESPONSE NOT COMPUTED",
            "PLANAR, STRAIGHT PUSH: OUT-OF-PLANE BEHAVIOUR NOT COMPUTED",
            "NO WALL BUCKLING OR RESIDUAL STRESS",
            "STRUT CONNECTION ASSUMED NO-SLIP; NEIGHBOUR RIGID"],
    # the renderer's headings, gated here so no heading escapes the audit
    heads=dict(
        brace="at mid-height, in the shown plane",
        braced="same section and length; less favourable braced shape",
        braced_short="less favourable braced shape",
        why="elastic buckling shapes",
        result="bare reference / less favourable braced shape"),
)
SILENT = dict(
    result="more than double across all registered size/shape/height cases",
    limits=["first yield only",
            "principal-plane model — out-of-plane behaviour not computed",
            "strut connection no-slip; neighbour rigid"],
)
NARRATION = dict(
    a1="A twelve metre steel column, pinned at both ends.",
    a2="Push down until the steel first yields.",
    b1="Two hundred and twenty-nine kilonewtons.",
    c1="Now brace the comparison column at mid-height with one steel strut.",
    e1="Six hundred and nineteen kilonewtons.",
    d1="The restraint splits the elastic buckling wave into two six metre halves.",
    e2="More than double throughout the registered sweep.",
    z1="That is first yield. This model stops there.",
    z2="One brace, in the modelled plane.",
)
IDENTITY = ["WORKER DRAWN TO SCALE, 1.70 m - AN ILLUSTRATION"]
FORBIDDEN = [r"\bcollaps", r"\bcapacity\b", r"\bfailure load", r"\bfails?\b",
             r"\bdesign resistance", r"\bsafe\b", r"\bunsafe\b",
             r"\bstrength\b", r"\bbreak", r"\bboth ways\b", r"\btwo-way\b"]
MIRROR_LINE = 1e-9            # an exact symmetry: machine scale


def load(path=None):
    path = path or os.path.join(CB.OUT, "column_brace.json")
    return json.load(io.open(path, encoding="utf-8"))


def _tol():
    return float(CB.config()["solver"]["newton_tol"])


def _pick(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            rv = r.get(k)
            if isinstance(v, float):
                ok = ok and rv is not None and abs(rv - v) <= 1e-9 * max(1.0, abs(v))
            else:
                ok = ok and rv == v
        if ok:
            out.append(r)
    return out


def identity_gate(doc):
    worst = 0.0
    for n, d in doc["identities"].items():
        worst = max(worst, abs(d["bare_over_euler"] - 1.0),
                    abs(d["ideal_over_4euler"] - 1.0))
    return dict(name="identity", by_mesh=doc["identities"], worst=worst,
                passes=bool(worst < 1e-3))


def _is_control(r):
    return r["k"] == 0.0 and r["shape"] == "two_waves"


def yield_gate(rows):
    expected = sorted((n, float(imp)) for n in CB.MESHES
                      for imp in CB.IMPERFECTIONS)
    controls = sorted((r["n_elements"], r["imperfection"]) for r in rows
                      if _is_control(r))
    not_yield = [r for r in rows if not r["reached_yield"]]
    excluded = sorted((r["n_elements"], r["imperfection"]) for r in not_yield
                      if _is_control(r))
    other = [(r["n_elements"], r["imperfection"], r["shape"], r["k"],
              r["height"], r["mode"]) for r in not_yield if not _is_control(r)]
    eligible = len(rows) - len(controls)
    return dict(name="yield", cases=len(rows), eligible=eligible,
                reached=eligible - len(other), controls=len(controls),
                control_modes=sorted({r["mode"] for r in rows if _is_control(r)}),
                not_yield_other=other,
                passes=bool(controls == expected and excluded == expected
                            and not other))


def bracket_gate(rows):
    bad = [(r["n_elements"], r["imperfection"], r["shape"], r["k"], r["height"])
           for r in rows if r["reached_yield"] and not r["bracket"]["ok"]]
    n = sum(1 for r in rows if r["reached_yield"])
    return dict(name="bracket", cases=n, bad=bad, passes=bool(n and not bad))


def equilibrium_gate(rows):
    worst = dict(free_residual_n=0.0, brace_closure_n=0.0,
                 sideways_closure_n=0.0, vertical_closure_n=0.0)
    worst_ratio, bad, n = 0.0, [], 0
    for r in rows:
        if not r["reached_yield"]:
            continue
        for e in r["equilibrium"]:
            n += 1
            for k in worst:
                worst[k] = max(worst[k], e[k])
                worst_ratio = max(worst_ratio, e[k] / e["bound_n"])
            if not e["ok"]:
                bad.append((r["n_elements"], r["imperfection"], r["shape"],
                            r["k"], r["height"], e["rung"]))
    return dict(name="equilibrium", rungs=n, worst=worst,
                worst_over_bound=worst_ratio, bad=bad,
                passes=bool(n and not bad))


def spring_gate(doc):
    s = doc["spring_checks"]
    return dict(name="spring", cases=len(s),
                worst_diagonal_error=max(abs(x["diagonal_error"]) for x in s),
                worst_energy_error=max(x["energy_error"] for x in s),
                passes=bool(s and all(x["ok"] for x in s)))


def monotone_gate(doc):
    m = doc["eigen_monotone"]
    return dict(name="monotone", k=m["k"], over_euler=m["over_euler"],
                ideal_threshold=m["ideal_threshold"],
                strut_over_threshold=m["strut_over_threshold"],
                worst_drop=m["worst_drop"], passes=bool(m["ok"]))


def mirror_gate(rows):
    E = CB.material().E
    kb = CB.strut_stiffness(E)
    out = []
    for shape, k in (("one_wave", 0.0), ("one_wave", kb), ("two_waves", kb)):
        kw = dict(n_elements=CB.NOMINAL_MESH, shape=shape, k=float(k),
                  height=0.5, steps=CB.LOAD_STEPS, tol=_tol())
        a = _pick(rows, imperfection=float(CB.NOMINAL_IMPERFECTION), **kw)
        b = _pick(rows, imperfection=-float(CB.NOMINAL_IMPERFECTION), **kw)
        assert a and b, (shape, k)
        out.append(dict(shape=shape, k=k, plus_n=a[0]["first_yield_n"],
                        minus_n=b[0]["first_yield_n"],
                        rel=abs(b[0]["first_yield_n"] / a[0]["first_yield_n"] - 1)))
    worst = max(x["rel"] for x in out)
    return dict(name="mirror", cases=out, worst=worst,
                passes=bool(worst <= MIRROR_LINE))


def ratio(rows, n, imp, shape, k, h, steps=CB.LOAD_STEPS, tol=None):
    # the BARE reference always carries its own governing imperfection, the
    # single half-wave. Its two-half-wave case is orthogonal to its first mode
    # and goes unstable before yield: it is not a column anyone could build
    # to compare against. The worse-shape rule applies to the BRACED column.
    tol = _tol() if tol is None else float(tol)
    b = _pick(rows, n_elements=n, imperfection=float(imp), shape="one_wave",
              k=0.0, height=0.5, steps=steps, tol=tol)
    r = _pick(rows, n_elements=n, imperfection=float(imp), shape=shape,
              k=float(k), height=float(h), steps=steps, tol=tol)
    assert b and r, (n, imp, shape, k, h, len(b), len(r))
    # the load-step control re-solves the nominal case; a deterministic solve
    # repeated must agree exactly, and then either copy is the same number
    for grp in (b, r):
        assert all(x["first_yield_n"] == grp[0]["first_yield_n"] for x in grp), grp
    return r[0]["first_yield_n"] / b[0]["first_yield_n"], b[0], r[0]


def resolution_gate(rows):
    E = CB.material().E
    kb = CB.strut_stiffness(E)
    coarse, fine = sorted(CB.MESHES)[0], max(CB.MESHES)
    mesh = {"%d->%d" % (coarse, CB.NOMINAL_MESH): 0.0,
            "%d->%d" % (CB.NOMINAL_MESH, fine): 0.0}
    for imp in CB.IMPERFECTIONS:
        for shape in CB.SHAPES:
            for f in CB.BRACE_FACTORS:
                for h in CB.BRACE_HEIGHTS:
                    a = ratio(rows, coarse, imp, shape, f * kb, h)[0]
                    b = ratio(rows, CB.NOMINAL_MESH, imp, shape, f * kb, h)[0]
                    c = ratio(rows, fine, imp, shape, f * kb, h)[0]
                    k1, k2 = list(mesh)
                    mesh[k1] = max(mesh[k1], abs(a / b - 1.0))
                    mesh[k2] = max(mesh[k2], abs(b / c - 1.0))
    nom = CB.NOMINAL_IMPERFECTION
    a = ratio(rows, CB.NOMINAL_MESH, nom, "one_wave", kb, 0.5)[0]
    b = ratio(rows, CB.NOMINAL_MESH, nom, "one_wave", kb, 0.5,
              steps=2 * CB.LOAD_STEPS)[0]
    steps = abs(a / b - 1.0)
    tol = 0.0
    for shape in CB.SHAPES:
        base = ratio(rows, CB.NOMINAL_MESH, nom, shape, kb, 0.5)[0]
        for t in CB.TOLERANCES:
            other = ratio(rows, CB.NOMINAL_MESH, nom, shape, kb, 0.5, tol=t)[0]
            tol = max(tol, abs(base / other - 1.0))
    production = mesh["%d->%d" % (CB.NOMINAL_MESH, fine)]
    u = max(production, steps, tol)
    return dict(name="resolution", mesh=mesh, load_steps=steps, tolerance=tol,
                u=u, passes=bool(u <= 0.03))


def brace_gate(rows):
    worst = max((r.get("brace_utilisation") or 0.0) for r in rows
                if 0.0 < r["k"] < 100 * CB.strut_stiffness(CB.material().E))
    return dict(name="brace", worst_utilisation=worst, passes=bool(worst < 0.8))


def ship_gate(rows, u):
    E = CB.material().E
    kb = CB.strut_stiffness(E)
    n = CB.NOMINAL_MESH
    cells = []
    for imp in CB.IMPERFECTIONS:
        for f in CB.BRACE_FACTORS:
            for h in CB.BRACE_HEIGHTS:
                per_shape = {s: ratio(rows, n, imp, s, f * kb, h)[0]
                             for s in CB.SHAPES}
                worse = min(per_shape, key=per_shape.get)
                cells.append(dict(imperfection=imp, brace_factor=f, height=h,
                                  by_shape=per_shape, worse_shape=worse,
                                  ratio=per_shape[worse]))
    rmin = min(c["ratio"] for c in cells)
    nominal = ratio(rows, n, CB.NOMINAL_IMPERFECTION, "one_wave", kb, 0.5)
    nominal_worse = min(ratio(rows, n, CB.NOMINAL_IMPERFECTION, s, kb, 0.5)[0]
                        for s in CB.SHAPES)
    line = 2.0 + 3.0 * u
    worst_cell = min(cells, key=lambda c: c["ratio"])
    return dict(name="ship", line=line, min_ratio=rmin, worst_cell=worst_cell,
                nominal_one_wave=nominal[0], nominal_worse_shape=nominal_worse,
                bare_n=nominal[1]["first_yield_n"],
                braced_n=nominal[2]["first_yield_n"], cells=cells,
                passes=bool(rmin >= line and nominal_worse >= 2.5))


def hits(text):
    """Every forbidden expression in `text` -- also used by the tests that
    audit the renderer's literals and the SNS copy."""
    out = []
    for pat in FORBIDDEN:
        for m in re.finditer(pat, text, re.I):
            out.append((m.group(0), text[max(0, m.start() - 30):m.end() + 30]))
    return out


def wording_gate(strings):
    found = [h for s in strings for h in hits(s)]
    return dict(name="wording", hits=found, passes=not found)


def _flatten(v):
    if isinstance(v, dict):
        return [x for vv in v.values() for x in _flatten(vv)]
    if isinstance(v, list):
        return [x for vv in v for x in _flatten(vv)]
    return [v]


def _strings():
    return ([CLAIM] + IDENTITY + list(NARRATION.values())
            + _flatten(ON_SCREEN) + _flatten(SILENT))


def run(write=True):
    doc = load()
    rows = doc["rows"]
    res = resolution_gate(rows)
    gates = [identity_gate(doc), yield_gate(rows), bracket_gate(rows),
             equilibrium_gate(rows), spring_gate(doc), monotone_gate(doc),
             mirror_gate(rows), res, brace_gate(rows),
             ship_gate(rows, res["u"]), wording_gate(_strings())]
    out = dict(claim=CLAIM, on_screen=ON_SCREEN, silent=SILENT,
               narration=NARRATION, identity=IDENTITY, gates=gates,
               passed=sum(bool(g["passes"]) for g in gates), total=len(gates))
    if write:
        json.dump(out, io.open(os.path.join(CB.OUT, "gates.json"), "w",
                               encoding="utf-8"), indent=1, default=float,
                  ensure_ascii=False)
    return out


def main():
    out = run()
    for g in out["gates"]:
        if g["name"] == "ship":
            print("ship         %s  line %.3f  min ratio %.3f (worst: imp %.4f, brace x%.1f, h %.2f, %s)  nominal %.3f / worse shape %.3f  bare %.1f kN braced %.1f kN"
                  % ("PASS" if g["passes"] else "FAIL", g["line"], g["min_ratio"],
                     g["worst_cell"]["imperfection"], g["worst_cell"]["brace_factor"],
                     g["worst_cell"]["height"], g["worst_cell"]["worse_shape"],
                     g["nominal_one_wave"], g["nominal_worse_shape"],
                     g["bare_n"] / 1e3, g["braced_n"] / 1e3))
            continue
        extra = {k: v for k, v in g.items()
                 if k not in ("name", "passes", "cells", "by_mesh", "k", "over_euler")}
        print("%-12s %s  %s" % (g["name"], "PASS" if g["passes"] else "FAIL",
                                json.dumps(extra, default=float)[:240]))
    print("%d / %d" % (out["passed"], out["total"]))
    return 0 if out["passed"] == out["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
