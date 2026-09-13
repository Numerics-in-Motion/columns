# One braced level

Two **12 m** steel columns of the same section — a 170 × 170 × 6.10 mm square
hollow section, A = 0.004 m², pinned at both ends — are pushed from above until
their steel first yields. The first is bare, crooked by L/1000 in a single
bow. The second is held at mid-height by **one** 3.0 m 100 × 100 × 5 mm steel
strut to a rigid neighbour, and carries the less favourable of its own two
crookedness shapes.

| | axial load at first yield |
|---|---|
| bare reference, single half-wave | **229.2 kN** |
| braced at mid-height, the less favourable of two braced crookedness shapes (two half-waves) | **619.3 kN** (× 2.70) |
| braced at mid-height, single half-wave | 931.7 kN (× 4.07) |

```
python reproduce.py            # all 199 solves, a few minutes
python reproduce.py --quick    # the two headline solves, seconds
```

## What is claimed

In this principal-plane model of a 12 m square steel tube, the bare one-wave
reference first yielded at 229 kN; with one mid-height lateral brace, the less
favourable of two braced crookedness shapes first yielded at 619 kN. Every
registered crookedness-size, shape, and brace-height case remained above twice
its corresponding one-wave bare reference.

**Why, in the elastic picture.** A perfect pinned column buckles at
π²EI/L². A mid-height restraint splits that buckling wave into two halves of
length L/2, which buckle at 4π²EI/L². A crooked column's bow grows faster as
the load nears its buckling load, so raising the buckling load holds the bow
down for longer. This explains the direction of the result; it does not mean
four times the first-yield load, and the computed ratio is 2.70.

**The registered cases** (27 cells, both crookedness shapes in each, the less
favourable kept):

| | |
|---|---|
| crookedness size | L/2000, L/1000, L/500 |
| crookedness shape | a single half-wave, or two half-waves |
| brace height | 0.45, 0.50, 0.55 of the length |
| brace stiffness | 0.5 ×, 1 ×, 2 × the strut — a control band, see below |

The smallest ratio anywhere is **× 2.33** (L/500, brace at 0.45 or 0.55 of the
length, two half-waves). The bare reference itself first yields at 237.0 /
229.2 / 215.1 kN at L/2000 / L/1000 / L/500, against an Euler load of 245.8 kN
(L/r = 179).

**The brace stiffness.** The tested strut is about 390 times the closed-form
ideal-brace threshold for this elastic model. Therefore 0.5×–2× is a saturated
stiff-brace control band, not a stiffness-sensitivity study; much softer braces
were not computed. The threshold (2P/L_b with L_b = L/2 and P = 4P_E) is
0.328 MN/m against the strut's 126.7 MN/m, and the solver's own buckling load
rises with the spring and reaches 4.000 P_E exactly at that threshold.

**With two half-waves and the brace at exactly mid-height** the braced point
sits on the shape's node and does not move, so that ratio is the same for
every stiffness. There, the brace only stops the single half-wave.

## What is not

**First yield of the extreme fibre only.** Not a collapse load, not an
ultimate or code design value: the post-yield response was not computed.

**The result applies to one principal bending plane.** No out-of-plane response
or biaxial stress interaction was computed. A real column would require
adequate restraint in the other direction; this study does not quantify that
system.

A straight push from above. No local buckling of the tube wall, no residual
stress, no twisting. The strut enters as a spring on the column's sideways
movement at the brace node: its own shortening is modelled, its connection is
assumed not to slip, and the neighbour it is attached to is assumed rigid.

## How it is solved

The published `column_fem.run_nonlinear_column` (P-delta, element geometric
stiffness, Newton at every load step) is called **unchanged**. The brace is
added to the stiffness matrix that function assembles, through a temporary
wrapper around `assemble_global`; with the spring at zero the call is the
published call, bit for bit. First yield is interpolated between load steps
by the published `inverse_design.first_yield_load`.

**The bare reference always carries a single half-wave.** Given two
half-waves, the bare column goes unstable before it yields, at 245.2 kN
(239.7 kN at 80 elements) against an Euler load of 245.8 kN: that shape is
orthogonal to its first buckling mode, so it does not bow in that mode until
the column is at the Euler load itself, and it is not a column to compare
against. The less-favourable-shape rule is applied to the braced column. The
two columns being compared therefore do not carry the same crookedness.

**Meshes 20 / 40 / 80 elements**, 40 in production. At 160 elements the
published solver's tangent check (smallest eigenvalue at or below 1e-9 of the
largest) trips on conditioning and reports instability at 160 kN on a column
that yields at 229 kN at both 40 and 80: that is the check's threshold, not
the column.

## Verification, not validation

| check | result |
|---|---|
| the solver's own tangent against π²EI/L² (bare) and 4π²EI/L² (middle held rigidly), at every mesh | within 1.3e-5 |
| first yield reached | 190 / 190 eligible solves, plus exactly the 9 bare two-half-wave controls, which go unstable at the Euler load |
| every first-yield load lies between the two rungs that bracket yield, in load and in stress | 190 / 190 |
| equilibrium at both bracketing rungs of every eligible solve: free-DOF residual, brace-force closure, sideways and vertical reaction closure | within the Newton stopping bound at all 380 rungs; worst 4.1e-7 of it |
| the brace wrapper changes one diagonal entry by k and nothing else; its energy is ½ k u_b²; the published assembly is restored | at every mesh and brace height |
| the solver's buckling load as the spring stiffens | never falls |
| crookedness sign flipped | identical first-yield loads |
| the braced/bare ratio: 40 → 80 elements (20 → 40: 6.1e-4), 600 → 1200 load steps, Newton tolerance 1e-8 → 1e-7 and 1e-9 | u = 1.5e-4 |
| the strut at the column's first yield, against its own squash and Euler loads | at most 1.9 % |
| ship line fixed before any braced solve: every ratio ≥ 2 + 3u, the nominal ≥ 2.5 | × 2.33, × 2.70 |

Every one of those checks the solve against itself or against theory for a
perfect column. No test of a real braced column is reproduced here.

## Files

```
solver/column_brace.py         the study: the columns, the brace, the sweep, the checks
solver/column_brace_gates.py   the gates and every shipped string
config/column_config.json      material and solver settings (the published ones)
data/column_brace.json         all 199 solves and the checks
data/gates.json                the gates as they were passed
reference/                     the frozen answer and every module's SHA-256
labels/worker.json             the scale figure beside the column: measured,
                               not data
```

The 1.70 m worker in the video is a generated illustration. Its height is
measured from helmet to soles and drawn on the column's own metre scale; it
carries no number.
