# The strut's 44.7 kg: brace the column, or build it wider or thicker?

The 12 m pinned column from [`one-braced-level`](../one-braced-level/) — 170 × 170 × 6.10 mm SHS, A = 0.004 m², f_y 250 MPa,
L/1000 crookedness — first yields at 229.2 kN bare and at 619.3 kN with one 3.0 m strut at mid-height to an existing rigid
anchor. That strut is 44.7 kg of steel (modelled member mass). Spend the same steel on the column instead:

| option | section | first yield | over bare |
|---|---|---|---|
| one strut to a rigid anchor | 170 × 6.10 mm | **619.3 kN** | ×2.70 |
| a wider tube, same wall | 189.5 × 6.10 mm | **316.5 kN** | ×1.38 |
| a thicker wall, same outside | 170 × 6.86 mm | **254.1 kN** | ×1.11 |

```
python reproduce.py            # full study
python reproduce.py --quick    # nominal solves only
```

## Method
`solver/column_brace.py` is the published 037 solver, unchanged (P-delta frame FE, Newton, the published first-yield
interpolation); `solver/strut_vs_section.py` only swaps the section for the two section options. Uncertainty u = relative
half-ranges in quadrature over crookedness L/2000–L/500, f_y 225–275 MPa and (strut only) strut stiffness ×0.5–×2 with the
anchor rigid; every adjacent gap must exceed max(10 %, 3u): strut → wider 95.7 % vs 51 %, wider → thicker 24.6 % vs 17 %.
Checks: 037's published numbers reproduced, Euler identities, mesh 80, yield brackets and equilibrium, wall class 1, equal
added steel.

## Scope
First yield of the extreme fibre in one principal plane — not collapse, not a design resistance. The strut ties to an
existing rigid anchor; anchor and connection steel are not counted and installed cost is not compared. The wider and thicker
tubes are alternative column designs, not retrofits. No local buckling, residual stress or out-of-plane response.
