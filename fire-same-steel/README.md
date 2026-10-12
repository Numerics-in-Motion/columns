# Same steel, four columns, one standard fire

Four columns from this repository's earlier equal-steel race — solid rod Ø71.4 mm, round tube Ø170 mm, square box
170 mm, I-section 180 mm — each holding exactly 0.004 m² of steel (E = 200 GPa, fy = 250 MPa), 3.0 m long, pinned,
unprotected, heated by the ISO 834 standard fire. How long until each column's modelled axial resistance falls below
the load it carries?

| | rod | tube | box | I-section |
|---|---|---|---|---|
| exposed surface per volume A_m/V | 56 /m | 134 /m | 170 /m | 176 /m (k_sh 0.69) |
| **Rule B** — each carries half of its own cold resistance | **20.1 min** | 13.1 | 11.6 | 12.9 |
| **Rule A** — the same 99 kN on all four | 20.1 | 26.0 | 25.1 | 14.4 |

Two claims, each tested across load levels 0.3–0.7, emissivity 0.7/0.8, the I-section shadow factor on/off and three
time steps, against a bar of max(10 %, 3u):

* **Rule B: the rod lasts longest**, by at least 42 % over the next column. The other three are *not* ranked.
* **Rule A: the rod outlasts the I-section**, by at least 24 %. The tube and box last longest and are not ranked
  against each other.

Cold, the same four rank box > tube > I-section > rod. The same steel packed solid has about a third of the exposed
surface per volume, so it heats about three times slower — which decides the race when each column carries the same
share of its own strength, and not when they all carry the same load.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

## The model (EN 1993-1-2:2005, nothing fitted)

ISO 834 gas temperature; unprotected steel heating, eq. 4.25 (α_c = 25 W/m²K, ε_m = 0.7, ε_f = 1), the temperature-
dependent specific heat; k_y,θ and k_E,θ from Table 3.1; flexural buckling in fire, eq. 4.5–4.7, minor axis,
γ_M,fi = 1.0. Hollow sections are heated on their outside perimeter. All four sections are class 1 in fire.

Checks that run every time: the heating integrator against an exact exponential solution; the stocky-limit failure
temperature against eq. 4.22; an independent 1-D radial conduction model of the solid rod (λ_a(θ) from 3.4.1.3)
against the uniform-temperature model at the rod's failure time (0.7 %); and a deliberately wrong A_m/V for the rod,
which must change the Rule-B winner.

**Limits.** Standard fire only; unprotected members; pinned ends with no restraint or thermal-expansion forces; no
local buckling, residual stress or creep beyond what the Eurocode formulas imply. "Failure" means the modelled
resistance crossing the load, not a computed collapse. This is not a fire rating of any real column or building.
