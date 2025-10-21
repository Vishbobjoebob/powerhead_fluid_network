# network.py
from __future__ import annotations
import numpy as np
from scipy.optimize import fsolve
from components import g, area

class Tank:
    def __init__(self, p):
        self.p = float(p)  # Pa

class Chamber:
    def __init__(self, Pc):
        self.Pc = float(Pc)  # Pa

class Branch:
    """Full fluid branch (fuel or LOX)."""
    def __init__(self, rho, mu, tank,
                 suction_lines, suction_bends,
                 pump,
                 discharge_lines, discharge_bends,
                 injector):
        self.rho = float(rho)
        self.mu  = float(mu)
        self.tank = tank
        self.suction_lines = list(suction_lines or [])
        self.suction_bends = list(suction_bends or [])
        self.pump = pump
        self.discharge_lines = list(discharge_lines or [])
        self.discharge_bends = list(discharge_bends or [])
        self.injector = injector

    # helpers
    def _dp_lines(self, lines, mdot):
        return sum(Li.dp(mdot, self.rho, self.mu) for Li in lines)

    def _dp_bends(self, bends, mdot):
        return sum(Bi.dp(mdot, self.rho) for Bi in bends)

    def suction_dp(self, mdot):
        return self._dp_lines(self.suction_lines, mdot) + self._dp_bends(self.suction_bends, mdot)

    def discharge_dp(self, mdot):
        return self._dp_lines(self.discharge_lines, mdot) + self._dp_bends(self.discharge_bends, mdot)

    def suction_inertance(self):
        return sum(Li.inertance(self.rho) for Li in self.suction_lines)

    def discharge_inertance(self):
        return sum(Li.inertance(self.rho) for Li in self.discharge_lines)

    # mdot residual
    def residual(self, mdot, N_rpm, Pc):
        """
        p_tank - Δp_suct + ρ g H(Q,N) = Pc + Δp_dis + Δp_inj
        Returns LHS - RHS (Pa).
        """
        Q = mdot / self.rho
        dps = self.suction_dp(mdot)
        dPp = self.rho * g * self.pump.H(Q, N_rpm, self.rho, self.mu)
        dpd = self.discharge_dp(mdot)
        dpi = self.injector.dp(mdot, self.rho)
        lhs = self.tank.p - dps + dPp
        rhs = Pc + dpd + dpi
        return lhs - rhs

    def npsh_available(self, mdot, p_vap):
        return (self.tank.p - p_vap - self.suction_dp(mdot)) / (self.rho * g)

# steady state (explicit Pc) retained for transient/legacy usage
def solve_given_Pc(fuel, lox, Nf, Nox, Pc):
    def F(x):
        mf, mo = x
        return [fuel.residual(mf, Nf, Pc), lox.residual(mo, Nox, Pc)]
    mf, mo = fsolve(F, x0=[1.0, 2.0])
    return {"Pc": Pc, "mdot_f": mf, "mdot_ox": mo, "MR": mo / mf}

# transient RHS
def rhs_branch_inertive(t, Q, br, N_of_t, Pc_of_t, Ls, Ld):
    """
    (Ls + Ld) dQ/dt = p_tank - Δp_suct + ρ g H(Q,N) - (Pc + Δp_dis + Δp_inj)
    Returns dQ/dt.
    """
    mdot = Q * br.rho
    dps = br.suction_dp(mdot)
    dpd = br.discharge_dp(mdot)
    N = N_of_t(t)
    Pc = Pc_of_t(t)
    dPp = br.rho * g * br.pump.H(Q, N, br.rho, br.mu)
    dpi = br.injector.dp(mdot, br.rho)
    num = br.tank.p - dps + dPp - (Pc + dpd + dpi)
    den = Ls + Ld
    return num / den

def line_inertance_sum(lines, rho):
    return sum(Li.inertance(rho) for Li in lines)
