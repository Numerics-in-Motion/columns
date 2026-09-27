# Which column gains most from one brace?

Four pinned steel columns, each 3.00 m long with the same 0.004 m² cross-sectional area, receive the same lateral brace at mid-height. In this ideal one-plane first-yield calculation, the solid round rod gets the largest **relative** improvement. It does **not** become the strongest column in absolute first-yield load.

| Shape | Bare one-wave first yield | Braced one-wave first yield | Matched gain R | Less-favourable braced wave / bare one-wave Rw |
|---|---:|---:|---:|---:|
| Solid rod | 248.24 kN | 909.31 kN | **3.663×** | 2.360× |
| I-section | 355.27 kN | 932.15 kN | 2.624× | 1.912× |
| Round tube | 898.99 kN | 986.58 kN | 1.097× | 1.026× |
| Box tube | 930.66 kN | 991.11 kN | 1.065× | 1.013× |

R compares **the same one-wave crookedness** before and after bracing. Rw is an *adverse-shape benchmark*: the lesser of the two braced crookedness-shape loads is divided by the **bare one-wave** reference. It is **not** a matched two-wave treatment effect. For rod and I-section, the bare two-wave control exited without a bracketed first-yield event, so no matched two-wave gain is reported. The tube/box order in Rw is unresolved under the study's precommitted >3% point-estimate separation rule; neither is claimed as a unique least beneficiary.

The brace is a modeled 3 m 100×100×5 mm steel strut to a rigid neighbor, with k₀=126.667 MN/m and zero initial preload. It adds **47.5%** to each column's own modeled steel volume; connection and anchor steel are excluded. Stiffness variations are a saturated stiff-restraint control, not brace-size optimization. Column deformation in the video is taken from solver nodal histories; lateral movement and shortening are shown ×10, while the displayed in-plane column widths remain true relative scale. The connection close-up is a generated, labelled **illustration**, not computed evidence.

## What happened to the original question

The original, frozen v3 question required both the strongest **and** weakest relative beneficiaries to be uniquely resolved under two crookedness shapes and all 27 sensitivity conditions. That gate **FAILED**. Round tube versus box is too close under the stipulated 3% separation rule. `reference/original_registration_v3.md` and `reference/v3_failed_results.json` preserve the failure; no threshold was edited to make it pass.

The narrower v4 question asks only whether the rod is the unambiguous leader. It was framed *after* v3's result was known, so the original 27-condition grid is descriptive/discovery data. Two additional, previously uncomputed parameter points were fixed and reviewed before execution: (amp/L, brace height/L, k/k₀) = (0.00075, 0.475, 0.75) and (0.00125, 0.525, 1.50). The rod leads all three alternatives in both R and Rw at each point, with interval separation and >3% point separation. This is a **within-model parameter check**, not experimental confirmation. Both holdout numerical uncertainty envelopes are below 0.31%; the envelopes are operational sums of mesh/step/tolerance/rung differences, not certified errors or model-form uncertainty.

The public registration copies replace internal episode identifiers with descriptive references; physical conditions, numerical thresholds, dates of freezing relative to data collection, and pass/fail conclusions are unchanged.

## Reproduce

From this directory, with Python, NumPy and SciPy installed:

```
python reproduce.py             # source hashes and 12 nominal solves
python reproduce.py --full      # original grid, numerical controls, mechanics, and new points
```

`--full` runs 1,320 registered study/control solves plus separate mechanics checks and can take many minutes. It must reproduce **both** statuses: v3 failed, v4 passed. The runner imports the byte-identical solver at the root of this public `columns` repository and the frozen brace and study modules under `solver/`; it does not import the private production project. `reference/source_sha256.json` protects those modules, configuration, result references and registrations. `data/` holds fresh results when `--full` is run and is not the authority for the frozen answer.

## Limits

These are first-yield loads, **not** collapse, ultimate capacity or a design/safety recommendation. The nonlinear frame model uses the extreme fibre of a single principal bending plane, pinned ends, the stated initial bow, and an ideal rigid neighbor. No local wall buckling, out-of-plane response, residual stress, connection slip, anchor flexibility or post-yield behavior is computed. The independent mechanics checks cover Euler/eigen values, spring matrix and energy, yield brackets, equilibrium, mirrored crookedness, and a square-box diagonal-fibre scope check. Agreement with them does not validate a real built column.
