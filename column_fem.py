"""Geometrically nonlinear (P-Delta / geometric-stiffness) beam-column
analysis for the imperfect-column buckling comparison.

Method (spec section 7's "preferred implementation" -- a second-order
P-Delta formulation, not full corotational large-rotation FE): the column
is meshed into several 2-node frame elements with the initial imperfection
baked into the node coordinates (`column_geometry.py`). The elastic
stiffness `K0` (computed once, from the initial imperfect geometry, via
`fem.assemble_global` -- reused unchanged) never changes; at each load
increment, the axial force in every element is used to form a linear
geometric ("stress") stiffness `Kg(N, L)`, and the equilibrium
displacement is found by *iterating to self-consistency* between the
current displacement and the axial-force-dependent tangent stiffness
`K0 + sum(Kg)` -- since the material stays linear-elastic throughout, this
fixed-point iteration converges to the same solution a full incremental
Newton-Raphson would, without needing residual-vector bookkeeping. This is
the numerically solved part of the model; the closed-form Perry-Robertson
formula below is the analytical part used only to validate it (spec
section 7's required distinction between the two).

Failure/critical load for a load step is whichever happens first:
  (a) max combined (axial + bending) stress reaches the material's yield
      stress -- checked via `failure.member_utilisation` per element, with
      `consider_buckling=False` (each mesh element's own short-length
      Euler load is not physically meaningful here; GLOBAL column
      buckling is exactly what the geometric-stiffness amplification of
      bending stress already captures as P approaches the true critical
      load), or
  (b) the fixed-point iteration fails to converge within the iteration
      budget (the tangent stiffness has gone singular/ill-conditioned --
      a limit point).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import LinAlgError

from failure import euler_buckling_load, member_utilisation
from fem import assemble_global, element_global_stiffness, local_stiffness


def geometric_stiffness_local(N: float, L: float) -> np.ndarray:
    """6x6 local linear geometric ("stress") stiffness for dof order
    [u_i,v_i,th_i, u_j,v_j,th_j] -- the standard consistent formulation
    (e.g. Przemieniecki). `N` is the element's own current axial force,
    tension positive (matching `fem.local_stiffness`'s convention and
    `fl[3]` from a local end-force recovery) -- compression (N<0) softens
    the lateral/rotational stiffness, which is exactly the buckling
    mechanism."""
    Kg = np.zeros((6, 6))
    L2 = L * L
    Kg[1, 1] = 6.0 / 5.0
    Kg[1, 2] = L / 10.0
    Kg[1, 4] = -6.0 / 5.0
    Kg[1, 5] = L / 10.0
    Kg[2, 1] = L / 10.0
    Kg[2, 2] = 2.0 * L2 / 15.0
    Kg[2, 4] = -L / 10.0
    Kg[2, 5] = -L2 / 30.0
    Kg[4, 1] = -6.0 / 5.0
    Kg[4, 2] = -L / 10.0
    Kg[4, 4] = 6.0 / 5.0
    Kg[4, 5] = -L / 10.0
    Kg[5, 1] = L / 10.0
    Kg[5, 2] = -L2 / 30.0
    Kg[5, 4] = -L / 10.0
    Kg[5, 5] = 2.0 * L2 / 15.0
    return (N / L) * Kg


def column_constrained_dofs(column) -> list:
    """Pinned-pinned column boundary conditions: bottom node fixed in both
    translations (rotation free); top node fixed in the LATERAL
    translation only (rotation free, axial translation free so the
    compressive load can be applied there). This is specific enough to
    columns that it is kept local to this module rather than further
    generalizing `fem.constrained_dofs`."""
    b, t = column.bottom_node, column.top_node
    return sorted({3 * b, 3 * b + 1, 3 * t})


@dataclass
class ColumnResult:
    column: object
    load_history: list = field(default_factory=list)          # [N]
    midspan_deflection_history: list = field(default_factory=list)  # [m]
    max_combined_stress_history: list = field(default_factory=list)  # [Pa]
    failure_load: float = None
    failure_mode: str = ""       # "yield" | "instability"
    failure_step: int = None
    analytical_p_cr: float = 0.0
    analytical_imperfect_capacity: float = 0.0
    n_load_steps_completed: int = 0
    computation_time: float = 0.0
    nodal_disp_history: list = field(default_factory=list)   # (ndof,) per step
    member_util_history: list = field(default_factory=list)  # {member_idx: util} per step

    def to_dict(self):
        return dict(
            name=self.column.name,
            load_history=self.load_history,
            midspan_deflection_history=self.midspan_deflection_history,
            max_combined_stress_history=self.max_combined_stress_history,
            failure_load=self.failure_load, failure_mode=self.failure_mode,
            failure_step=self.failure_step,
            analytical_p_cr=self.analytical_p_cr,
            analytical_imperfect_capacity=self.analytical_imperfect_capacity,
            n_load_steps_completed=self.n_load_steps_completed,
            computation_time=self.computation_time,
        )

    def state_at_load(self, P):
        """Return (disp reshaped (Nn,2), util_map, failed_bool) at the
        nearest completed load step to `P` (clamped to the last completed
        step if P exceeds it -- i.e. the column has already failed by
        then)."""
        n = len(self.load_history)
        if n == 0:
            ndof = 3 * self.column.n_nodes
            return np.zeros((self.column.n_nodes, 2)), {}, False
        idx = int(np.searchsorted(self.load_history, P))
        idx = max(0, min(idx, n - 1))
        disp = self.nodal_disp_history[idx].reshape(-1, 3)[:, :2]
        util = self.member_util_history[idx]
        failed = self.failure_load is not None and P >= self.failure_load
        return disp, util, failed


def analytical_imperfect_capacity(E, I, A, c, L, delta0, sigma_yield,
                                  K_eff=1.0):
    """Closed-form Perry-Robertson / secant-formula capacity of an
    imperfect pin-ended column: the load P at which

        sigma_max(P) = P/A + P*delta0*c / (I*(1 - P/P_cr))  ==  sigma_yield

    `P_cr = pi^2 E I / (K_eff L)^2` (Euler, reused from `failure.
    euler_buckling_load`). Solved by bisection (monotonic, unique root in
    (0, P_cr)) -- this is the ANALYTICAL validation target for the
    numerical P-Delta solver, not itself part of the simulated result.
    """
    P_cr = euler_buckling_load(E, I, L, K_eff)

    def f(P):
        return P / A + P * delta0 * c / (I * (1.0 - P / P_cr)) - sigma_yield

    lo, hi = 1e-6 * P_cr, P_cr * (1.0 - 1e-9)
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def run_nonlinear_column(column, material, n_load_steps, load_max_factor,
                         newton_max_iter=40, newton_tol=1e-8,
                         n_elements_override=None) -> ColumnResult:
    """Load-controlled P-Delta analysis of one imperfect column up to
    first failure (yield or instability), whichever comes first."""
    import time as _time
    t_start = _time.perf_counter()

    E, A, I, c = material.E, column.area, column.inertia, column.fibre_c
    L = column.height
    fail_stress = material.failure_stress

    res = ColumnResult(column)
    res.analytical_p_cr = euler_buckling_load(E, I, L, 1.0)
    res.analytical_imperfect_capacity = analytical_imperfect_capacity(
        E, I, A, c, L, column.imperfection_amplitude, fail_stress, 1.0)

    ndof = 3 * column.n_nodes
    fixed = set(column_constrained_dofs(column))
    free = [d for d in range(ndof) if d not in fixed]

    K0, cache = assemble_global(column.nodes, column.members, E, A, I)

    top = column.top_node
    mid_node = column.n_nodes // 2
    P_max = load_max_factor * res.analytical_p_cr
    dP = P_max / n_load_steps

    F_ext = np.zeros(ndof)
    u = np.zeros(ndof)

    for step in range(1, n_load_steps + 1):
        P = step * dP
        F_ext[:] = 0.0
        F_ext[3 * top + 1] = -P  # compressive load, -y at the top node

        u_iter = u.copy()
        converged = False
        for _ in range(newton_max_iter):
            Kt = K0.copy()
            N_elem = []
            for k, (i, j) in enumerate(column.members):
                T, kl, Le = cache[k]
                ue = u_iter[[3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]]
                ul = T @ ue
                fl = kl @ ul
                N = fl[3]
                N_elem.append(N)
                Kg_local = geometric_stiffness_local(N, Le)
                Kg_global = T.T @ Kg_local @ T
                dofs = [3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]
                for a in range(6):
                    for b in range(6):
                        Kt[dofs[a], dofs[b]] += Kg_global[a, b]

            Kt_free = Kt[np.ix_(free, free)]
            try:
                evals = np.linalg.eigvalsh(Kt_free)
                if evals[0] <= 1e-9 * max(evals[-1], 1.0):
                    raise LinAlgError("tangent stiffness non-positive-definite")
                u_new_free = np.linalg.solve(Kt_free, F_ext[free])
            except LinAlgError:
                converged = False
                break

            u_new = np.zeros(ndof)
            u_new[free] = u_new_free
            delta = np.max(np.abs(u_new - u_iter))
            scale = max(np.max(np.abs(u_new)), 1e-12)
            u_iter = u_new
            if delta < newton_tol * scale:
                converged = True
                break

        if not converged:
            res.failure_load = P
            res.failure_mode = "instability"
            res.failure_step = step
            break

        max_stress = 0.0
        util_map = {}
        for k, (i, j) in enumerate(column.members):
            T, kl, Le = cache[k]
            ue = u_iter[[3*i, 3*i+1, 3*i+2, 3*j, 3*j+1, 3*j+2]]
            ul = T @ ue
            fl = kl @ ul
            axial, Mi, Mj = fl[3], fl[2], fl[5]
            util, detail = member_utilisation(axial, Mi, Mj, A, I, c,
                                              fail_stress, E, Le,
                                              consider_buckling=False)
            util_map[k] = util
            max_stress = max(max_stress, detail["combined_stress"])

        u = u_iter
        res.load_history.append(P)
        res.midspan_deflection_history.append(
            float(u[3 * mid_node] - column.nodes[mid_node, 0]))
        res.max_combined_stress_history.append(max_stress)
        res.nodal_disp_history.append(u.copy())
        res.member_util_history.append(util_map)
        res.n_load_steps_completed = step

        if max_stress >= fail_stress:
            res.failure_load = P
            res.failure_mode = "yield"
            res.failure_step = step
            break

    res.computation_time = _time.perf_counter() - t_start
    return res
