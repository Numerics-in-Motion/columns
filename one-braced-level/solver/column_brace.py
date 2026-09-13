# -*- coding: utf-8 -*-
"""One braced level: what does a single mid-height brace do to a 12 m column?

THE QUESTION

A 12.0 m steel column -- the channel's canonical 170 x 170 x 6.10 mm square
hollow section, A = 0.004 m2, pinned at both ends, crooked by L/1000 -- is
pushed from above until its extreme fibre first yields. Then ONE lateral brace
is added at mid-height, in the modelled bending plane: a 3.0 m
100 x 100 x 5 mm SHS strut to a rigid neighbour, its connection assumed not to
slip. Same section, same length, same load. How much more load does it carry
to first yield?

A PRINCIPAL-PLANE MODEL. The solver is planar. Nothing out of that plane --
restraint in the other direction, biaxial bending, the tube's diagonal fibre
-- is computed, and no result here is a two-way result.

THE ANCHOR IS AN IDENTITY

A perfect pinned column buckles at P = pi^2 EI / L^2. Hold its middle still and
each half is a pinned column of length L/2, so P = 4 pi^2 EI / L^2: exactly four
times. `identity` checks the solver's own tangent against both, and the
braced column's brace stiffness against the ideal limit.

WHAT IS MEASURED, AND WHAT IS NOT

The axial load at FIRST YIELD of the extreme fibre, interpolated between load
rungs by the published `first_yield_load`. Not a collapse load, not an ultimate
capacity, not a code design resistance: a real column carries past first
yield, and residual stress, local buckling of the wall, connection slip and
three-dimensional effects are not modelled.

THE SOLVER IS THE PUBLISHED ONE

`column_fem.run_nonlinear_column` (P-delta, geometric stiffness, Newton) is
called unchanged. The brace enters as a lateral spring added to the linear
stiffness the published function assembles; with the spring at zero the call
is the published call. No module is edited.

WHAT IS SWEPT, BEFORE ANY BRACED RESULT WAS SOLVED
(a closed-form Perry-Robertson forecast had been seen: about 229 kN bare,
610-710 kN braced)

  imperfection size    L/2000, L/1000, L/500
  imperfection shape   the bare column's single half-wave, and the braced
                       column's own two half-waves -- the WORSE is kept
  brace stiffness      0.5x, 1x, 2x the real strut (and zero and ideal as
                       controls). The strut is ~390x the closed-form
                       ideal-brace threshold, so this is a saturated
                       stiff-brace control band, not a sensitivity study
  brace height         0.45 L, 0.50 L, 0.55 L
  mesh                 20, 40, 80 elements (a node on every brace height;
                       160 trips the published solver's conditioning check)

SHIP LINE (pre-registered in the design review, before any braced solve)

  min over the sweep of P_y(braced) / P_y(bare)  >=  2.00 + 3u,  u <= 3 %,
  the nominal ratio >= 2.5, and the brace below 80 % of its own yield and
  buckling resistance at the column's first yield. Otherwise stop.
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import column_fem as CF                                  # noqa: E402
import section_properties as SP                          # noqa: E402
from column_geometry import build_column                 # noqa: E402
from inverse_design import first_yield_load              # noqa: E402
from materials import Material                           # noqa: E402

# ---------------------------------------------------------------------------
# THE FREEZE
# ---------------------------------------------------------------------------
LENGTH = 12.0
AREA = 0.004
OUTER = 0.170                          # the canonical box, t = 6.10 mm
IMPERFECTIONS = (1 / 2000, 1 / 1000, 1 / 500)
NOMINAL_IMPERFECTION = 1 / 1000
SHAPES = ("one_wave", "two_waves")
BRACE_FACTORS = (0.5, 1.0, 2.0)
BRACE_HEIGHTS = (0.45, 0.50, 0.55)
# 20 / 40 / 80: production 40, refinement 80. NOT 160 -- at 160 elements the
# published solver's tangent check (smallest eigenvalue <= 1e-9 of the
# largest) trips on conditioning alone and reports "instability": the bare
# column came back 160.4 kN at 160 elements against 229.2 kN at both 40 and 80.
# That is the solver's threshold, not the column, so the finest honest mesh is 80
MESHES = (20, 40, 80)
NOMINAL_MESH = 40
LOAD_STEPS = 600
LOAD_MAX_FACTOR = 4.5                  # times the BARE Euler load, both columns
# the strut
STRUT_OUTER, STRUT_T, STRUT_LENGTH = 0.100, 0.005, 3.0
IDEAL_FACTOR = 1000.0
TOLERANCES = (1e-7, 1e-9)              # either side of the published 1e-8

CONFIG = os.environ.get("BRACE_CONFIG") or os.path.join(
    ROOT, "config", "column_config.json")
OUT = os.environ.get("BRACE_DATA") or os.path.join(ROOT, "data")


def config():
    return json.load(io.open(CONFIG, encoding="utf-8"))


def material(cfg=None):
    return Material.from_config((cfg or config())["material"])


def section():
    return SP.box_properties(AREA, OUTER)


def strut():
    b, t = STRUT_OUTER, STRUT_T
    A = b * b - (b - 2 * t) ** 2
    I = (b ** 4 - (b - 2 * t) ** 4) / 12.0
    return dict(area=A, inertia=I, length=STRUT_LENGTH)


def strut_stiffness(E):
    s = strut()
    return E * s["area"] / s["length"]


def strut_resistance(E, fy):
    """The smaller of the strut's squash load and its pinned Euler load."""
    s = strut()
    return min(s["area"] * fy, np.pi ** 2 * E * s["inertia"] / s["length"] ** 2)


# ---------------------------------------------------------------------------
# the column and the brace
# ---------------------------------------------------------------------------
def column(n_elements, imperfection, shape):
    col = build_column("brace", section(), LENGTH, int(n_elements),
                       float(imperfection))
    if shape == "two_waves":
        ys = col.nodes[:, 1]
        col.nodes = np.column_stack([imperfection * LENGTH
                                     * np.sin(2 * np.pi * ys / LENGTH), ys])
    elif shape != "one_wave":
        raise KeyError(shape)
    return col


@contextmanager
def lateral_spring(dof, k):
    """Add `k` to the lateral DOF of the brace node in the published assembly."""
    original = CF.assemble_global

    def with_spring(nodes, members, E, A, I):
        K, cache = original(nodes, members, E, A, I)
        K = K.copy()
        K[dof, dof] += k
        return K, cache

    CF.assemble_global = with_spring
    try:
        yield
    finally:
        CF.assemble_global = original


def brace_node(col, height):
    n = len(col.nodes) - 1
    k = height * n
    if abs(k - round(k)) > 1e-9:
        raise ValueError("brace height %.3f L is not on a node of %d elements"
                         % (height, n))
    return int(round(k))


def solve(n_elements=NOMINAL_MESH, imperfection=NOMINAL_IMPERFECTION,
          shape="one_wave", k=0.0, height=0.50, steps=LOAD_STEPS, cfg=None,
          keep=False, tol=None):
    cfg = cfg or config()
    mat = material(cfg)
    s = cfg["solver"]
    tol = float(s["newton_tol"] if tol is None else tol)
    col = column(n_elements, imperfection, shape)
    node = brace_node(col, height)
    with lateral_spring(3 * node, float(k)):
        t0 = time.time()
        r = CF.run_nonlinear_column(col, mat, int(steps), LOAD_MAX_FACTOR,
                                    int(s["newton_max_iter"]), tol)
    # a case that never completes a load step has no first-yield load; it is
    # recorded as not reaching yield instead of taking the sweep down
    Py = (first_yield_load(r, mat.failure_stress) if r.load_history
          else float("nan"))
    out = dict(n_elements=int(n_elements), imperfection=float(imperfection),
               shape=shape, k=float(k), height=float(height), steps=int(steps),
               tol=tol, first_yield_n=float(Py), mode=r.failure_mode,
               reached_yield=bool(r.failure_mode == "yield"),
               euler_bare_n=float(r.analytical_p_cr),
               runtime_s=time.time() - t0)
    # the brace force at the rung nearest first yield
    if r.nodal_disp_history:
        loads = np.asarray(r.load_history)
        i = int(np.argmin(np.abs(loads - Py)))
        u = r.nodal_disp_history[i]
        ub = float(u[3 * node])
        out["brace_force_n"] = float(abs(k * ub))
        out["brace_utilisation"] = float(abs(k * ub)
                                         / strut_resistance(mat.E, mat.failure_stress))
    if out["reached_yield"]:
        out["bracket"] = yield_bracket(r, Py, mat.failure_stress)
        out["equilibrium"] = [equilibrium(r, col, mat, node, float(k), i, tol)
                              for i in out["bracket"]["rungs"]]
    if keep:
        out["_result"] = r
        out["_column"] = col
        out["_node"] = node
    return out


# ---------------------------------------------------------------------------
# the checks on every solve
# ---------------------------------------------------------------------------
def yield_bracket(r, Py, fy):
    """First yield lies between the last rung below yield and the first at or
    above it, in load AND in stress -- what `first_yield_load` interpolates."""
    P = np.asarray(r.load_history)
    S = np.asarray(r.max_combined_stress_history)
    i = int(np.argmax(S >= fy))
    ok = bool(i >= 1 and S[i] >= fy and S[i - 1] < fy
              and P[i - 1] <= Py <= P[i])
    return dict(rungs=[i - 1, i], load=[float(P[i - 1]), float(P[i])],
                stress=[float(S[i - 1]), float(S[i])], ok=ok)


def equilibrium(r, col, mat, node, k, rung, tol):
    """Equilibrium of a converged rung, rebuilt from the published element
    functions WITHOUT the spring.

    r = (K0 + Kg(u)) u - F. At every free DOF except the brace's it must be
    zero; at the brace DOF it must be the spring's pull, -k u_b. Summed over
    every sideways DOF it is zero (a rigid translation carries no force), so
    the support reactions and the brace force close; summed over the vertical
    DOFs it is the applied load. The line is the solver's own stopping rule:
    Newton stops when an increment is below `tol` of the largest displacement,
    and the residual of such an iterate is bounded by the tangent's row norm
    times that increment.
    """
    E = mat.E
    u = np.asarray(r.nodal_disp_history[rung], float)
    P = float(r.load_history[rung])
    K0, cache = CF.assemble_global(col.nodes, col.members, E, col.area,
                                   col.inertia)
    Kt = K0.copy()
    for kk, (i, j) in enumerate(col.members):
        T, kl, Le = cache[kk]
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        N = (kl @ (T @ u[dofs]))[3]
        Kt[np.ix_(dofs, dofs)] += T.T @ CF.geometric_stiffness_local(N, Le) @ T
    F = np.zeros_like(u)
    F[3 * col.top_node + 1] = -P
    res = Kt @ u - F
    fixed = set(CF.column_constrained_dofs(col))
    b = 3 * node
    free = [d for d in range(len(u)) if d not in fixed and d != b]
    bound = float(np.abs(Kt).sum(axis=1).max() * tol * np.abs(u).max())
    xs = [d for d in range(len(u)) if d % 3 == 0]
    ys = [d for d in range(len(u)) if d % 3 == 1]
    out = dict(rung=int(rung), load_n=P, bound_n=bound,
               free_residual_n=float(np.abs(res[free]).max()),
               brace_closure_n=float(abs(res[b] + k * u[b])),
               brace_force_n=float(k * u[b]),
               sideways_closure_n=float(abs(res[xs].sum())),
               vertical_closure_n=float(abs(res[ys].sum() - P)),
               bottom_reaction_n=float(res[3 * col.bottom_node + 1]))
    out["ok"] = bool(max(out["free_residual_n"], out["brace_closure_n"],
                         out["sideways_closure_n"],
                         out["vertical_closure_n"]) <= bound)
    return out


def spring_checks(cfg=None):
    """The wrapper adds k at one diagonal entry and nothing else, its energy is
    1/2 k u_b^2 for any u, and the published assembly is restored on exit."""
    cfg = cfg or config()
    E = material(cfg).E
    kb = strut_stiffness(E)
    out = []
    for n in MESHES:
        for h in BRACE_HEIGHTS:
            col = column(n, NOMINAL_IMPERFECTION, "one_wave")
            node = brace_node(col, h)
            original = CF.assemble_global
            Ka, _ = CF.assemble_global(col.nodes, col.members, E, col.area,
                                       col.inertia)
            with lateral_spring(3 * node, kb):
                Kb, _ = CF.assemble_global(col.nodes, col.members, E,
                                           col.area, col.inertia)
            restored = CF.assemble_global is original
            D = Kb - Ka
            nz = [[int(v) for v in x] for x in np.argwhere(D != 0)]
            scale = float(np.abs(Ka).max())
            u = np.random.default_rng(n).standard_normal(len(D))
            energy = float(abs(0.5 * u @ D @ u - 0.5 * kb * u[3 * node] ** 2))
            ok = bool(restored and nz == [[3 * node, 3 * node]]
                      and abs(D[3 * node, 3 * node] - kb) <= 1e-12 * scale
                      and energy <= 1e-12 * scale * float(u @ u))
            out.append(dict(n_elements=n, height=h, nonzeros=nz,
                            diagonal_error=float(D[3 * node, 3 * node] - kb),
                            energy_error=energy, restored=restored, ok=ok))
    return out


def eigen_monotone(cfg=None):
    """The solver's own buckling load of the perfect column, as the brace
    stiffens: it may not fall, and it is recorded against the closed-form
    ideal-brace threshold 2P/L_b = 16 P_E / L."""
    cfg = cfg or config()
    E = material(cfg).E
    kb = strut_stiffness(E)
    pe = np.pi ** 2 * E * section()["I_min"] / LENGTH ** 2
    beta = 16.0 * pe / LENGTH
    ks = [0.0, 0.25 * beta, 0.5 * beta, beta, 2 * beta, 4 * beta, 0.5 * kb,
          kb, 2 * kb, IDEAL_FACTOR * kb]
    loads = [tangent_eigen_load(NOMINAL_MESH, k, cfg=cfg) for k in ks]
    drops = [max(0.0, (a - b) / a) for a, b in zip(loads, loads[1:])]
    return dict(k=ks, over_euler=[x / pe for x in loads],
                ideal_threshold=beta, strut_over_threshold=kb / beta,
                worst_drop=max(drops), ok=bool(max(drops) <= 1e-9))


# ---------------------------------------------------------------------------
# the identities
# ---------------------------------------------------------------------------
def tangent_eigen_load(n_elements, k, height=0.5, cfg=None):
    """Elastic buckling load of the PERFECT column from the solver's own
    linear and geometric stiffness: det(K0 + P Kg1) = 0 with a unit-compression
    geometric stiffness. Uses the published element functions only."""
    from scipy.linalg import eigh
    cfg = cfg or config()
    mat = material(cfg)
    col = column(n_elements, 0.0, "one_wave")
    node = brace_node(col, height)
    E, A, I = mat.E, col.area, col.inertia
    with lateral_spring(3 * node, float(k)):
        K0, cache = CF.assemble_global(col.nodes, col.members, E, A, I)
    ndof = 3 * col.n_nodes
    Kg = np.zeros((ndof, ndof))
    for kk, (i, j) in enumerate(col.members):
        T, kl, Le = cache[kk]
        g = T.T @ CF.geometric_stiffness_local(-1.0, Le) @ T
        dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
        Kg[np.ix_(dofs, dofs)] += g
    fixed = set(CF.column_constrained_dofs(col))
    free = [d for d in range(ndof) if d not in fixed]
    # lateral + rotational DOFs only: the axial chain decouples for a
    # straight column and would add spurious zero-geometric-stiffness rows
    lat = [d for d in free if d % 3 != 1]
    vals = eigh(K0[np.ix_(lat, lat)], -Kg[np.ix_(lat, lat)], eigvals_only=True)
    vals = vals[vals > 0]
    return float(vals.min())


def identities(cfg=None):
    cfg = cfg or config()
    mat = material(cfg)
    sec = section()
    EI = mat.E * sec["I_min"]
    pe = np.pi ** 2 * EI / LENGTH ** 2
    out = {}
    for n in MESHES:
        bare = tangent_eigen_load(n, 0.0, cfg=cfg)
        ideal = tangent_eigen_load(n, 1e14, cfg=cfg)
        real = tangent_eigen_load(n, strut_stiffness(mat.E), cfg=cfg)
        out[n] = dict(bare_over_euler=bare / pe, ideal_over_4euler=ideal / (4 * pe),
                      real_over_4euler=real / (4 * pe))
    return out


# ---------------------------------------------------------------------------
# the study
# ---------------------------------------------------------------------------
def _job(args):
    return solve(**args)


def jobs():
    E = material().E
    kb = strut_stiffness(E)
    js = []
    for n in MESHES:
        for imp in IMPERFECTIONS:
            for shape in SHAPES:
                js.append(dict(n_elements=n, imperfection=imp, shape=shape, k=0.0,
                               height=0.50))
                for f in BRACE_FACTORS:
                    for h in BRACE_HEIGHTS:
                        js.append(dict(n_elements=n, imperfection=imp, shape=shape,
                                       k=f * kb, height=h))
    # controls at the nominal mesh: ideal brace, and a load-step refinement
    # "ideal": 1000x the real strut. At 1e14 N/m the published tangent check
    # (smallest eigenvalue <= 1e-9 of the largest) called the column unstable
    # at the first rung -- a conditioning artefact, not a buckling
    for imp in IMPERFECTIONS:
        for shape in SHAPES:
            js.append(dict(n_elements=NOMINAL_MESH, imperfection=imp, shape=shape,
                           k=IDEAL_FACTOR * kb, height=0.50))
    for steps in (LOAD_STEPS, 2 * LOAD_STEPS):
        js.append(dict(n_elements=NOMINAL_MESH, imperfection=NOMINAL_IMPERFECTION,
                       shape="one_wave", k=0.0, height=0.50, steps=steps))
        js.append(dict(n_elements=NOMINAL_MESH, imperfection=NOMINAL_IMPERFECTION,
                       shape="one_wave", k=kb, height=0.50, steps=steps))
    # solver tolerance, one decade either side of the published 1e-8, and the
    # imperfection's sign: the bare reference and both braced shapes
    nominal = [("one_wave", 0.0), ("one_wave", kb), ("two_waves", kb)]
    for tol in TOLERANCES:
        for shape, k in nominal:
            js.append(dict(n_elements=NOMINAL_MESH,
                           imperfection=NOMINAL_IMPERFECTION, shape=shape,
                           k=k, height=0.50, tol=tol))
    for shape, k in nominal:
        js.append(dict(n_elements=NOMINAL_MESH,
                       imperfection=-NOMINAL_IMPERFECTION, shape=shape, k=k,
                       height=0.50))
    return js


def run_study(processes=None, verbose=True):
    from multiprocessing import Pool
    processes = processes or min(20, os.cpu_count() or 1)
    # one BLAS thread per worker: twenty processes each spawning a full thread
    # pool for the dense eigen-solve would fight over the same cores
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    js = jobs()
    if verbose:
        print("  %d solves on %d processes" % (len(js), processes), flush=True)
    t = time.time()
    rows = []
    with Pool(processes) as pool:
        for n, r in enumerate(pool.imap_unordered(_job, js)):
            rows.append(r)
            if verbose and (n + 1) % 50 == 0:
                print("  %d / %d  %.0f s" % (n + 1, len(js), time.time() - t),
                      flush=True)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    ident = identities()
    print("identities:", json.dumps(ident))
    springs = spring_checks()
    monotone = eigen_monotone()
    print("spring checks ok:", all(x["ok"] for x in springs),
          " eigen monotone ok:", monotone["ok"])
    rows = run_study()
    E = material().E
    doc = dict(frozen=dict(length=LENGTH, area=AREA, outer=OUTER,
                           imperfections=IMPERFECTIONS, shapes=SHAPES,
                           brace_factors=BRACE_FACTORS,
                           brace_heights=BRACE_HEIGHTS, meshes=MESHES,
                           load_steps=LOAD_STEPS,
                           load_max_factor=LOAD_MAX_FACTOR, strut=strut(),
                           strut_stiffness=strut_stiffness(E),
                           strut_resistance=strut_resistance(
                               E, material().failure_stress)),
               identities=ident, spring_checks=springs,
               eigen_monotone=monotone, rows=rows,
               runtime_s=time.time() - t)
    json.dump(doc, io.open(os.path.join(OUT, "column_brace.json"), "w",
                           encoding="utf-8"), indent=1, default=float)
    print("wrote", os.path.join(OUT, "column_brace.json"),
          "%.0f s" % (time.time() - t))


if __name__ == "__main__":
    main()
