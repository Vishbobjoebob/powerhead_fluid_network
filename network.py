# network.py
from __future__ import annotations

from components import g


class Tank:
    """Simple tank boundary condition with a fixed absolute pressure [Pa]."""

    def __init__(self, p: float):
        self.p = float(p)


class Chamber:
    """Combustion chamber boundary with a fixed nominal Pc [Pa]."""

    def __init__(self, Pc: float):
        self.Pc = float(Pc)


class Branch:
    """
    Full fluid branch: tank -> suction plumbing -> pump -> discharge plumbing -> injector.

    The branch knows:
      * fluid properties (ρ, μ)
      * tank pressure boundary
      * pump object
      * suction / discharge line + bend components
      * injector component

    It exposes:
      - suction_dp(mdot)  : total suction-side pressure drop [Pa]
      - discharge_dp(mdot): total discharge-side pressure drop [Pa]
      - residual(mdot, N_rpm, Pc):
            p_tank - Δp_suct + ρ g H_pump(Q,N)
                - (Pc + Δp_dis + Δp_inj)
        used as a scalar root for steady-state mass flow.
      - npsh_available(mdot, p_vap): available NPSH at the pump eye [m]
    """

    def __init__(
        self,
        rho: float,
        mu: float,
        tank: Tank,
        suction_lines,
        suction_bends,
        pump,
        discharge_lines,
        discharge_bends,
        injector,
    ):
        self.rho = float(rho)
        self.mu = float(mu)
        self.tank = tank

        # Store component lists as plain Python lists
        self.suction_lines = list(suction_lines or [])
        self.suction_bends = list(suction_bends or [])
        self.pump = pump
        self.discharge_lines = list(discharge_lines or [])
        self.discharge_bends = list(discharge_bends or [])
        self.injector = injector

    # ------------------------------------------------------------------
    # Internal helpers: sum losses across components
    # ------------------------------------------------------------------
    def _dp_lines(self, lines, mdot: float) -> float:
        """Sum Darcy–Weisbach losses over a list of Line components."""
        return sum(Li.dp(mdot, self.rho, self.mu) for Li in lines)

    def _dp_bends(self, bends, mdot: float) -> float:
        """Sum minor losses over a list of Bend components."""
        return sum(Bi.dp(mdot, self.rho) for Bi in bends)

    # ------------------------------------------------------------------
    # Public hydraulic utilities
    # ------------------------------------------------------------------
    def suction_dp(self, mdot: float) -> float:
        """Total suction-side pressure drop [Pa]."""
        return self._dp_lines(self.suction_lines, mdot) + self._dp_bends(self.suction_bends, mdot)

    def discharge_dp(self, mdot: float) -> float:
        """Total discharge-side pressure drop [Pa]."""
        return self._dp_lines(self.discharge_lines, mdot) + self._dp_bends(self.discharge_bends, mdot)

    # ------------------------------------------------------------------
    # Scalar residual used by the steady solver
    # ------------------------------------------------------------------
    def residual(self, mdot: float, N_rpm: float, Pc: float) -> float:
        """
        Hydraulic balance for this branch.

        In head/pressure form:

            p_tank - Δp_suction + ρ g H_pump(Q, N)
                = Pc + Δp_discharge + Δp_injector

        We return LHS - RHS in Pascals; the root of this function in mdot
        is the steady-state operating point for the given Pc and pump speed.
        """
        Q = mdot / self.rho

        dps = self.suction_dp(mdot)
        dpd = self.discharge_dp(mdot)
        dpi = self.injector.dp(mdot, self.rho)

        dP_pump = self.rho * g * self.pump.H(Q, N_rpm)

        lhs = self.tank.p - dps + dP_pump
        rhs = Pc + dpd + dpi

        return lhs - rhs

    # ------------------------------------------------------------------
    # NPSH helper
    # ------------------------------------------------------------------
    def npsh_available(self, mdot: float, p_vap: float) -> float:
        """
        Available NPSH at the pump eye [m]:

            NPSH_avail = (p_tank - p_vap - Δp_suction) / (ρ g)
        """
        return (self.tank.p - p_vap - self.suction_dp(mdot)) / (self.rho * g)
    
    def pressure_profile(self, mdot: float, N_rpm: float, Pc: float):
        """
        Build a 1D pressure map along the branch for debugging.

        Returns an ordered list of (label, p) pairs, starting at the tank
        and stepping through:

            Tank -> suction lines -> suction bends -> pump ->
            discharge lines -> discharge bends -> injector -> chamber

        Pressures are absolute [Pa].
        """
        nodes = []
        rho = self.rho
        mu = self.mu
        Q = mdot / rho

        # ----------------- suction side -----------------------------------
        p = self.tank.p
        nodes.append(("tank_outlet", p))

        # suction lines in order
        for i, L in enumerate(self.suction_lines):
            dp = L.dp(mdot, rho, mu)
            p -= dp
            label = f"after_suction_line[{i}] ({L.name})"
            nodes.append((label, p))

        # suction bends in order
        for i, B in enumerate(self.suction_bends):
            dp = B.dp(mdot, rho)
            p -= dp
            label = f"after_suction_bend[{i}] ({B.name})"
            nodes.append((label, p))

        # pump inlet (same as last suction node)
        nodes.append(("pump_inlet", p))

        # ----------------- pump -------------------------------------------
        H_pump = self.pump.H(Q, N_rpm)      # [m]
        dP_pump = rho * g * H_pump          # [Pa]
        p += dP_pump
        nodes.append(("pump_outlet", p))

        # ----------------- discharge side ---------------------------------
        for i, L in enumerate(self.discharge_lines):
            dp = L.dp(mdot, rho, mu)
            p -= dp
            label = f"after_discharge_line[{i}] ({L.name})"
            nodes.append((label, p))

        for i, B in enumerate(self.discharge_bends):
            dp = B.dp(mdot, rho)
            p -= dp
            label = f"after_discharge_bend[{i}] ({B.name})"
            nodes.append((label, p))

        # injector
        p_inj_in = p
        nodes.append(("injector_inlet", p_inj_in))

        dp_inj = self.injector.dp(mdot, rho)
        p -= dp_inj
        nodes.append(("injector_outlet", p))   # should be ~ Pc

        # You can compare this last value with the input Pc if you like.
        return nodes

