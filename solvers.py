# solvers.py
from __future__ import annotations

from typing import Any, Dict

from scipy.optimize import fsolve


def solve_branch_mdot(branch, N_rpm: float, Pc: float, mdot_guess: float) -> float:
    """
    Solve for the branch mass flow rate that zeros the hydraulic residual:

        branch.residual(mdot, N_rpm, Pc) = 0

    We wrap `fsolve` so the rest of the code doesn't need to know SciPy's API.
    """

    def residual_scalar(mdot_scalar: float) -> float:
        return branch.residual(float(mdot_scalar), N_rpm, Pc)

    # `fsolve` operates on vectors; we unwrap to a scalar.
    mdot_solution = fsolve(
        lambda m: residual_scalar(m[0]),
        x0=[float(mdot_guess)],
    )[0]

    return float(mdot_solution)


def solve_coupled_fixed_point(
    ctx: Dict[str, Any],
    Pc0: float | None = None,
    mdot_f0: float = 1.0,
    mdot_o0: float = 2.0,
    omega: float = 0.6,
    tol_rel: float = 1e-5,
    tol_abs: float = 1e-3,
    max_iter: int = 50,
) -> Dict[str, float]:
    """
    Simple fixed-point outer loop that couples:

        - Fuel branch
        - LOX branch
        - Nozzle / TCA

    Algorithm:
      1) Start from Pc_0 (either provided or from ctx['chamber'].Pc)
      2) At each iteration k:
           a) Solve each branch for mdot_f, mdot_ox at Pc_k.
           b) Ask the TCA (nozzle) for a consistent Pc_noz and thrust.
           c) Relax Pc_k toward Pc_noz:

                  Pc_{k+1} = (1 - ω) * Pc_k + ω * Pc_noz

      3) Stop when Pc and both mass flows have converged.

    Returns:
      dict with keys:
         'Pc', 'mdot_f', 'mdot_ox', 'MR', 'p_exit', 'thrust_N'
      (and possibly 'warning' if the loop didn't converge in max_iter)
    """
    fuel_branch = ctx["branches"]["fuel"]
    lox_branch = ctx["branches"]["lox"]
    tca = ctx["tca"]

    Nf = ctx["steady"]["N_fuel_rpm"]
    No = ctx["steady"]["N_lox_rpm"]

    # Initial Pc guess
    Pc = float(Pc0 if Pc0 is not None else ctx["chamber"].Pc)

    mdot_f = float(mdot_f0)
    mdot_o = float(mdot_o0)

    for it in range(1, max_iter + 1):
        # ------------------------------------------------------------------
        # 1) Solve branches at current Pc
        # ------------------------------------------------------------------
        mdot_f_new = solve_branch_mdot(
            branch=fuel_branch,
            N_rpm=Nf,
            Pc=Pc,
            mdot_guess=mdot_f if it > 1 else mdot_f0,
        )
        mdot_o_new = solve_branch_mdot(
            branch=lox_branch,
            N_rpm=No,
            Pc=Pc,
            mdot_guess=mdot_o if it > 1 else mdot_o0,
        )

        # ------------------------------------------------------------------
        # 2) Nozzle / TCA: back-solve Pc and thrust from mass flows
        # ------------------------------------------------------------------
        Pc_noz, MR, p_exit, thrust_N = tca.backsolve_pc_and_thrust(mdot_f_new, mdot_o_new)

        # ------------------------------------------------------------------
        # 3) Relax Pc toward nozzle prediction
        # ------------------------------------------------------------------
        Pc_next = (1.0 - omega) * Pc + omega * Pc_noz

        # Convergence checks
        dPc_abs = abs(Pc_next - Pc)
        dPc_rel = dPc_abs / max(abs(Pc), 1.0)
        d_mdot_f = abs(mdot_f_new - mdot_f)
        d_mdot_o = abs(mdot_o_new - mdot_o)

        # Update iterate
        Pc = Pc_next
        mdot_f = mdot_f_new
        mdot_o = mdot_o_new

        if (dPc_abs < tol_abs or dPc_rel < tol_rel) and d_mdot_f < 1e-6 and d_mdot_o < 1e-6:
            Pc_final, MR, p_exit, thrust_N = tca.backsolve_pc_and_thrust(mdot_f, mdot_o)
            return {
                "Pc": Pc_final,
                "mdot_f": mdot_f,
                "mdot_ox": mdot_o,
                "MR": MR,
                "p_exit": p_exit,
                "thrust_N": thrust_N,
            }

    # If we reach here, we didn't converge within max_iter; still return last iterate.
    Pc_final, MR, p_exit, thrust_N = tca.backsolve_pc_and_thrust(mdot_f, mdot_o)
    return {
        "Pc": Pc_final,
        "mdot_f": mdot_f,
        "mdot_ox": mdot_o,
        "MR": MR,
        "p_exit": p_exit,
        "thrust_N": thrust_N,
        "warning": "fixed-point did not converge within max_iter",
    }
