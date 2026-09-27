# columns

The column solver behind Numerics in Motion's column studies, once, at the
root, and one folder per study that reuses it.

```
column_fem.py  column_geometry.py  inverse_design.py
fem.py  geometry.py  failure.py  materials.py  section_properties.py  utils.py
```

`column_fem.run_nonlinear_column` pushes a crooked two-dimensional column from
above in load steps: Euler-Bernoulli frame elements, an element geometric
stiffness for the P-delta effect, and Newton iterations at every step. It
stops when the tangent stiffness stops being positive definite or when the
extreme fibre reaches yield, and `inverse_design.first_yield_load`
interpolates the load at which that happened.

Every module here is byte-identical to the same file in the solver folder of
the public
[three-thresholds](https://github.com/Numerics-in-Motion/three-thresholds)
package.

**Limits shared by every study here.** First yield, not collapse: nothing here
computes how far a column carries past it. A planar model: no twisting, no
out-of-plane behaviour, no local buckling of a tube wall, no residual stress.
The published tangent check flags instability when the smallest eigenvalue
falls below 1e-9 of the largest; very fine meshes can trip it on conditioning
alone, so each study states the meshes it used and why.

## Studies

| folder | question |
|---|---|
| [`one-braced-level/`](one-braced-level/) | A 12 m steel column is pushed until it first yields. Hold its middle with one steel strut: how much more load does it take? |
| [`which-column-gains-most-from-one-brace/`](which-column-gains-most-from-one-brace/) | Four equal-area 3 m columns get the same brace. The rod gains the most *relative* first-yield load; the original four-way extreme-rank gate failed and remains public. |

Each study folder carries a `reproduce.py` that checks every module against a
SHA-256 frozen with its reference, re-runs the study, and fails if the answer
has moved.

`requirements.txt` is numpy and scipy.
