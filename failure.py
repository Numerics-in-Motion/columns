"""Member stress recovery, buckling and failure/utilisation calculations."""
from __future__ import annotations

import numpy as np


def euler_buckling_load(E: float, I: float, L: float, K: float = 1.0) -> float:
    """P_cr = pi^2 E I / (K L)^2  [N] (positive magnitude)."""
    return np.pi ** 2 * E * I / (K * L) ** 2


def member_utilisation(axial, Mi, Mj, area, inertia, fibre_c,
                       failure_stress, E, L,
                       consider_buckling=True, K_eff=1.0):
    """Return (utilisation, detail dict) for one frame member.

    utilisation = max( combined_stress / failure_stress ,
                       |axial_compression| / P_cr )      (>=1 => failed)

    axial: tension positive [N];  Mi, Mj: end moments [N m].
    """
    axial_stress = axial / area
    max_moment = max(abs(Mi), abs(Mj))
    bending_stress = max_moment * fibre_c / inertia
    combined_stress = abs(axial_stress) + bending_stress

    util_stress = combined_stress / failure_stress

    util_buckling = 0.0
    p_cr = np.inf
    if consider_buckling and axial < 0.0:
        p_cr = euler_buckling_load(E, inertia, L, K_eff)
        util_buckling = abs(axial) / p_cr

    utilisation = max(util_stress, util_buckling)
    detail = dict(
        axial_force=float(axial),
        axial_stress=float(axial_stress),
        max_moment=float(max_moment),
        bending_stress=float(bending_stress),
        combined_stress=float(combined_stress),
        util_stress=float(util_stress),
        util_buckling=float(util_buckling),
        p_cr=float(p_cr),
        utilisation=float(utilisation),
    )
    return utilisation, detail
