"""The inverse question: fix the load, and ask each shape what it costs.

Every study on this rig so far has fixed the STEEL and asked which shape is
strongest. This module answers the other direction, which is the direction real
design runs in: make every shape reach the same load, then compare what each
one cost -- both the steel and the finished size.

(The question came from a comment on a published video. It is paraphrased
here and nowhere quoted: `COMMENTS_LEDGER.md` rule 5 bans names, handles and
verbatim quotation from anything the audience sees, and this file ships.)

WHAT "RESIZE THE SHAPE" CAN MEAN, AND WHY IT MATTERS

There is more than one way to make a shape carry more, and they are not the
same question:

  `scale` -- GEOMETRIC. Every dimension times lambda, so area goes as
      lambda^2 and second moment as lambda^4. This is what a catalogue does:
      the next size up is roughly the same shape, larger. Both the section
      size and the steel change, which is what the question asked about.
      **b/t is invariant**, so a scaled section keeps its parent's slenderness
      classification and local buckling stays exactly where it was.

  `thicken` -- WALL ONLY. Hold the outer dimension at the parent's value and
      grow the wall. Closer to what the parent did, but it answers a narrower
      question ("how thick must this particular size be") and it cannot
      compare SIZES because they never change. It also drives b/t down, so a
      thickened section is not the parent's section in any local-buckling
      sense either.

Both are provided. A caller has to choose and say which on screen; the two give
different numbers and neither is the "real" answer on its own.

WHAT IS HELD FIXED IN BOTH

The length, the end conditions, the material, and the out-of-straightness. The
imperfection is the member's, not the section's -- a 3 m column that is 3 mm
out of straight stays 3 mm out of straight when you give it a fatter section.
Its EFFECT falls, because the bending stress it causes goes as `c/I`, and that
is real rather than an artefact.

WHAT THIS CANNOT SAY

It cannot say which shape is cheapest. Steel is not priced by the kilogram
alone -- fabrication, connections and availability dominate, and a shape that
needs less steel can easily cost more to build. The quantity here is mass, and
mass is not price.
"""
from __future__ import annotations

import math

import section_properties as sp
from column_fem import run_nonlinear_column
from column_geometry import build_column

# The parent's proportions, read from its config by `shapes_from_config` so
# nothing here is a transcribed constant.
KINDS = ("box", "hollow_tube", "i_section", "solid_circle")


def shapes_from_config(cfg):
    """The parent's four shapes, as functions of a scale factor.

    Returns {kind: builder(lambda) -> section dict} for geometric scaling, and
    the base dimensions the thickening variant needs.
    """
    g = cfg["geometry"]
    A0 = g["target_area_m2"]
    B0 = g["box_width_ratio"] * g["envelope_width_m"]
    D0 = g["tube_diameter_ratio"] * g["envelope_width_m"]
    h0 = g["i_section_height_ratio"] * g["envelope_height_m"]
    bf_ratio = g["i_section_flange_width_ratio"]
    tf_tw = g["i_section_tf_tw_ratio"]

    def scale(kind, lam):
        A = A0 * lam * lam
        if kind == "solid_circle":
            return sp.solid_circle_properties(A)
        if kind == "hollow_tube":
            return sp.hollow_tube_properties(A, D0 * lam)
        if kind == "box":
            return sp.box_properties(A, B0 * lam)
        if kind == "i_section":
            return sp.i_section_properties(A, h0 * lam, bf_ratio * h0 * lam,
                                           tf_tw)
        raise KeyError(kind)

    def thicken(kind, lam):
        """Outer dimension held at the parent's; only the area grows."""
        A = A0 * lam * lam
        if kind == "solid_circle":
            # a solid rod has no wall to thicken -- it can only grow
            return sp.solid_circle_properties(A)
        if kind == "hollow_tube":
            return sp.hollow_tube_properties(A, D0)
        if kind == "box":
            return sp.box_properties(A, B0)
        if kind == "i_section":
            return sp.i_section_properties(A, h0, bf_ratio * h0, tf_tw)
        raise KeyError(kind)

    return dict(scale=scale, thicken=thicken, base_area=A0)


def first_yield_load(result, fy):
    """The load at which the combined stress reaches yield, INTERPOLATED.

    `run_nonlinear_column` reports the first load rung whose stress has already
    passed yield, so its answer is high by up to one rung. That is invisible on
    a fine ramp and enormous on a coarse one -- on a stocky section the rung can
    be 136 kN, and the reported "capacity" then exceeds the squash load, which
    no column can reach.

    Interpolating linearly in stress between the last rung below yield and the
    first above removes the quantisation entirely and makes the answer nearly
    independent of the step count, which is what an inverse problem needs:
    every bisection step here would otherwise carry its own rung error.
    """
    P = list(result.load_history)
    S = list(result.max_combined_stress_history)
    for i in range(1, len(S)):
        if S[i] >= fy:
            if S[i] == S[i - 1]:
                return float(P[i])
            f = (fy - S[i - 1]) / (S[i] - S[i - 1])
            return float(P[i - 1] + f * (P[i] - P[i - 1]))
    return float(P[-1])


def capacity(kind, section, cfg, material, n_steps):
    """The first-yield load under this model, interpolated.

    NOT "the capacity" and NOT "failure": the solver stops at first yield of
    the combined axial-plus-bending stress. A real column carries past first
    yield, and a real column is also reduced by residual stress, local
    buckling and imperfections this model has no term for.
    """
    g, s = cfg["geometry"], cfg["solver"]
    col = build_column(kind, section, g["length_m"], int(g["n_elements"]),
                       float(g["imperfection_ratio"]))
    r = run_nonlinear_column(col, material, int(n_steps),
                             float(s["load_max_factor"]),
                             int(s["newton_max_iter"]), float(s["newton_tol"]))
    return first_yield_load(r, material.failure_stress), col


def press_curve(kind, section, cfg, material, n_steps, samples=40):
    """(P, mid-span deflection) as this column is loaded to first yield.

    THE DEFLECTION IS THE ONE LARGE HONEST MOTION ON THIS RIG, and it was
    nearly thrown away. A note in an earlier study said the bow was "3.00 mm
    for all four, identical, so it shows no contest" -- but 3.00 mm is the
    INITIAL out-of-straightness, not the deflected shape. Loaded to its own
    first-yield load, the box reaches 3.93 mm and the solid rod 27.02 mm: a
    6.9x spread, because the box is at 0.24 of its Euler load and the rod at
    0.89 of its own. **A slender column bends visibly and a stocky one does
    not, and that is this video's subject rather than its decoration.**

    Sampled from the solver's own displacement history, not from the
    amplification formula, so what is drawn is what was solved.
    """
    g, s = cfg["geometry"], cfg["solver"]
    col = build_column(kind, section, g["length_m"], int(g["n_elements"]),
                       float(g["imperfection_ratio"]))
    r = run_nonlinear_column(col, material, int(n_steps),
                             float(s["load_max_factor"]),
                             int(s["newton_max_iter"]), float(s["newton_tol"]))
    fy = first_yield_load(r, material.failure_stress)
    P = list(r.load_history)

    # NOT `midspan_deflection_history`. That field is
    # `u_x - nodes[mid, 0]` -- the displacement MINUS the initial
    # out-of-straightness -- so it starts at -3.00 mm and stays negative for a
    # stocky column, and a gate written against it reported the box as bending
    # -2.08 mm. The name says deflection; the quantity is not one.
    #
    # `column_fem.py` is canonical and ships inside published packages, so it
    # is not touched. The honest quantity is assembled here instead: the true
    # lateral POSITION of the midspan node, initial bow plus displacement.
    # It reproduces the closed form delta0/(1 - P/Pcr) to two decimals.
    mid = len(col.nodes) // 2
    x0 = float(col.nodes[mid, 0])
    D = [x0 + float(u[3 * mid]) for u in r.nodal_disp_history]
    keep = [i for i in range(len(P)) if P[i] <= fy] or [0]
    idx = [keep[round(k * (len(keep) - 1) / (samples - 1))]
           for k in range(samples)]
    S = list(r.max_combined_stress_history)
    fyv = material.failure_stress
    # the utilisation is already in the solver's history; colouring by it costs
    # nothing and is the only thing on screen that changes fast at the start
    out = [{"load_n": float(P[i]), "deflection_m": float(D[i]),
            "utilisation": float(S[i] / fyv)} for i in idx]
    # end exactly on first yield, interpolated, so the last frame of the ramp
    # is the number the rest of the video quotes
    if len(keep) > 1 and P[keep[-1]] < fy:
        j = keep[-1]
        f = ((fy - P[j]) / (P[j + 1] - P[j])) if j + 1 < len(P) else 0.0
        k = min(j + 1, len(D) - 1)
        out[-1] = {"load_n": float(fy),
                   "deflection_m": float(D[j] + f * (D[k] - D[j])),
                   "utilisation": 1.0}
    return out, float(fy), float(r.analytical_p_cr)


def steel_for_load(kind, builder, target, cfg, material, n_steps,
                   lo=0.20, hi=6.0, iters=44):
    """The scale factor at which this shape just carries `target`.

    Bisection rather than a formula: the capacity comes out of a nonlinear
    P-Delta solve with a first-yield stop, and there is no closed form to
    invert. The bracket is checked rather than assumed, so a shape that cannot
    reach the target inside it says so instead of returning the bracket end.
    """
    c_lo, _ = capacity(kind, builder(kind, lo), cfg, material, n_steps)
    c_hi, _ = capacity(kind, builder(kind, hi), cfg, material, n_steps)
    if not (c_lo < target < c_hi):
        raise ValueError("%s: target %.1f N is not bracketed by [%g, %g] "
                         "(%.1f .. %.1f N)" % (kind, target, lo, hi, c_lo, c_hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        c, _ = capacity(kind, builder(kind, mid), cfg, material, n_steps)
        if c < target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)
    sec = builder(kind, lam)
    got, col = capacity(kind, sec, cfg, material, n_steps)
    r_gyr = math.sqrt(col.inertia / col.area)
    return dict(
        kind=kind, lam=lam, section=sec, achieved=got,
        error=(got - target) / target,
        area=float(sec["area"]), outer=float(sec["outer_dim"]),
        mass_per_m=float(sec["area"]) * float(cfg["material"]["density_kg_m3"]),
        radius_of_gyration=r_gyr,
        slenderness=cfg["geometry"]["length_m"] / r_gyr,
        fibre_over_r=float(col.fibre_c) / r_gyr,
        r_over_outer=r_gyr / float(sec["outer_dim"]))
