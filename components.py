# components.py
from __future__ import annotations

import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Generic hydraulic utilities
# ---------------------------------------------------------------------------
class HydroUtils:
    """
    Collection of static helpers for hydraulic / turbomachinery calculations.
    """
    PA_PER_PSI = 6894.757293168361
    g = 9.80665  # [m/s^2]

    # --- unit conversions --------------------------------------------------
    @staticmethod
    def pa2psi(x: float) -> float:
        """Convert pressure from Pa to psi."""
        return float(x) / HydroUtils.PA_PER_PSI

    @staticmethod
    def psi2pa(x: float) -> float:
        """Convert pressure from psi to Pa."""
        return float(x) * HydroUtils.PA_PER_PSI

    # --- simple geometry ---------------------------------------------------
    @staticmethod
    def area(D: float) -> float:
        """Cross-sectional area of a circular duct with diameter D [m]."""
        return math.pi * (D**2) / 4.0

    # --- basic dimensionless groups ---------------------------------------
    @staticmethod
    def reynolds(rho: float, V: float, D: float, mu: float) -> float:
        """Reynolds number for internal flow."""
        return rho * V * D / mu

    @staticmethod
    def tip_speed(D: float, N_rps: float) -> float:
        """Blade tip speed for diameter D [m] and rotational speed N [rev/s]."""
        return math.pi * D * N_rps

    # --- friction factor / Colebrook-White -------------------------------
    @staticmethod
    def colebrook_white(
        Re: float,
        eps: float,
        D: float,
        f0: float = 0.02,
        iters: int = 30,
        tol: float = 1e-10,
    ) -> float:
        """
        Colebrook-White implicit equation for turbulent friction factor.

        Inputs
        ------
        Re   : Reynolds number
        eps  : roughness height [m]
        D    : hydraulic diameter [m]
        f0   : initial guess for f
        iters: max Newton iterations
        tol  : stopping tolerance on f
        """
        f = f0
        for _ in range(iters):
            inv_sqrt_f = 1.0 / math.sqrt(f)
            rhs = (eps / D) / 3.7 + 2.51 / (Re * math.sqrt(f))

            # Colebrook-White in form: 1/sqrt(f) = -2 log10( rhs )
            gF = inv_sqrt_f + 2.0 * math.log10(rhs)

            # Derivative wrt f (simple Newton step)
            d_inv = -0.5 * f ** (-1.5)
            d_rhs = 2.51 * (-0.5) * f ** (-1.5) / Re
            dg = d_inv + 2.0 / (math.log(10.0) * rhs) * d_rhs

            f_new = f - gF / dg
            if abs(f_new - f) < tol:
                f = f_new
                break
            f = f_new
        return float(f)

    # --- orifice / injector relations -------------------------------------
    @staticmethod
    def orifice_dp_from_mdot(mdot: float, rho: float, CdA: float) -> float:
        """
        Pressure drop across an orifice from mass flow:

            Δp = ( m_dot / (CdA) )^2 / (2 ρ)

        """
        return (mdot / CdA) ** 2 / (2.0 * rho)

    @staticmethod
    def orifice_mdot_from_dp(dp: float, rho: float, CdA: float) -> float:
        """
        Mass flow through an orifice for a given Δp:

            m_dot = CdA * sqrt( 2 ρ Δp )
        """
        return CdA * math.sqrt(2.0 * rho * dp)


# Handy exports so other modules can do:
#    from components import g, pa2psi, psi2pa, area
g = HydroUtils.g
pa2psi = HydroUtils.pa2psi
psi2pa = HydroUtils.psi2pa
area = HydroUtils.area


# ---------------------------------------------------------------------------
# Base component class
# ---------------------------------------------------------------------------
class Component:
    """
    Lightweight base class for all hydraulic components.

    Child classes should implement:
        - dp(...) -> float : pressure drop [Pa]
        - optional inertance(...) -> float : inertance [kg/m^4] (for transients; unused here)
    """

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    def dp(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def inertance(self, *args, **kwargs) -> float:
        # Transient model has been removed, so default inertance is zero.
        return 0.0


# ---------------------------------------------------------------------------
# Line (Darcy-Weisbach) and bends (minor losses)
# ---------------------------------------------------------------------------
class Line(Component):
    """Straight pipe with Darcy–Weisbach friction."""
    def __init__(self, L: float, D: float, eps: float, name: str | None = None):
        super().__init__(name)
        self.L = float(L)      # length [m]
        self.D = float(D)      # diameter [m]
        self.eps = float(eps)  # roughness height [m]

    def dp(self, mdot: float, rho: float, mu: float) -> float:
        """
        Pressure drop using Darcy–Weisbach:
            Δp = f * (L/D) * (ρ V^2 / 2)
        """
        A = HydroUtils.area(self.D)
        V = mdot / (rho * A)
        Re = HydroUtils.reynolds(rho, V, self.D, mu)

        if Re < 2300.0:
            # laminar
            f = 64.0 / Re
        else:
            # turbulent: Colebrook-White
            f = HydroUtils.colebrook_white(Re, self.eps, self.D)

        return f * (self.L / self.D) * 0.5 * rho * V * V

    def inertance(self, rho: float) -> float:
        """
        Hydraulic inertance L_h = ρ L / A [kg/m^4]
        (Not used in steady-state, but kept for completeness.)
        """
        return rho * self.L / HydroUtils.area(self.D)


class Bend(Component):
    """Minor loss with a fixed loss coefficient K."""
    def __init__(self, K: float, D: float, name: str | None = None):
        super().__init__(name)
        self.K = float(K)  # dimensionless loss coefficient
        self.D = float(D)  # diameter [m]

    def dp(self, mdot: float, rho: float) -> float:
        """
        Pressure drop across the bend:

            Δp = K * ρ V^2 / 2
        """
        V = mdot / (rho * HydroUtils.area(self.D))
        return self.K * 0.5 * rho * V * V


# ---------------------------------------------------------------------------
# Injector (simple orifice)
# ---------------------------------------------------------------------------
class Injector(Component):
    """Simple orifice injector modeled with an effective CdA."""
    def __init__(self, CdA: float, name: str | None = None):
        super().__init__(name)
        self.CdA = float(CdA)  # [m^2]

    def dp(self, mdot: float, rho: float) -> float:
        """Pressure drop across the injector for a given mass flow."""
        return HydroUtils.orifice_dp_from_mdot(mdot, rho, self.CdA)

    def mdot_from_dp(self, dp: float, rho: float) -> float:
        """Mass flow for a given pressure drop across the injector."""
        return HydroUtils.orifice_mdot_from_dp(dp, rho, self.CdA)


# ---------------------------------------------------------------------------
# Gulich radial pump model
# ---------------------------------------------------------------------------
class GulichPump(Component):
    """
    Radial centrifugal pump per Gülich, with two head models:

      1) head_model = "meanline" (default)
         Use Gülich-style meanline correlations (geometry-based).

      2) head_model = "curve"
         Use a multi-RPM Q–H pump map.  The map is stored as:

             curve_map[rpm] = {"Q": [Q_i], "H": [H_i]}

         and we bilinearly interpolate in Q and N_rpm.
    """
    def __init__(
        self,
        # geometry
        d1, dn, d2, b2, zLa, beta2B_deg, n_q,
        alpha1_deg=90.0, e1=0.0, e=0.0, lambdaLa_deg=90.0,
        # efficiencies / leakage
        QS3=0.0, QSp=0.0,
        # sheet parameters
        Q_ref=1.0,
        # head model options
        head_model: str = "meanline",
        curve_map: dict[float, dict[str, list[float]]] | None = None,
        name: str | None = None,
    ):
        super().__init__(name)

        # --- impeller geometry (meanline) ---------------------------------
        self.d1 = float(d1)
        self.dn = float(dn)
        self.d2 = float(d2)
        self.b2 = float(b2)
        self.zLa = int(zLa)
        self.beta2B_deg = float(beta2B_deg)
        self.alpha1_deg = float(alpha1_deg)
        self.e1 = float(e1)
        self.e = float(e)
        self.lambdaLa_deg = float(lambdaLa_deg)
        self.d1m_star = (d1 + dn) / (2.0 * d2)

        # --- leakage flows -------------------------------------------------
        self.QS3 = float(QS3)
        self.QSp = float(QSp)

        # --- sheet parameters ----------------------------------------------
        self.Q_ref = float(Q_ref)
        self.n_q = float(n_q)

        # --- head model selection ------------------------------------------
        self.head_model = (head_model or "meanline").lower()
        if self.head_model not in ("meanline", "curve"):
            raise ValueError(f"Unknown head_model '{head_model}'. Use 'meanline' or 'curve'.")

        # Multi-RPM curve data
        self.curve_map: dict[float, dict[str, list[float]]] = curve_map or {}

        # Efficiency used in power_W(). In meanline mode, updated in H_meanline().
        self.eta_worst: float = 0.45
    # ----------------- internal helpers for meanline model -----------------
    def _u(self, d: float, n_rpm: float) -> float:
        """Blade speed at diameter d for rotational speed n_rpm."""
        return math.pi * d * (n_rpm / 60.0)

    def _A1(self) -> float:
        """Inlet flow area between shroud and hub."""
        return (math.pi / 4.0) * (self.d1**2 - self.dn**2)

    def _tau_inlet(self, beta1: float) -> float:
        """Inlet blockage factor τ₁ due to finite blade thickness."""
        s1 = math.sin(beta1)
        sL = math.sin(math.radians(self.lambdaLa_deg))
        return 1.0 / (1.0 - (self.zLa * self.e1) / (math.pi * self.d1 * s1 * sL))

    def _tau_outlet(self) -> float:
        """Outlet blockage factor τ₂."""
        s2B = math.sin(math.radians(self.beta2B_deg))
        sL = math.sin(math.radians(self.lambdaLa_deg))
        return 1.0 / (1.0 - (self.e * self.zLa) / (math.pi * self.d2 * s2B * sL))

    def _kw(self) -> float:
        """Wiesner slip correction factor k_w."""
        s2B = math.sin(math.radians(self.beta2B_deg))
        eps_lim = math.e ** (-8.16 * s2B / self.zLa)
        return 1.0 - ((self.d1m_star - eps_lim) / (1.0 - eps_lim))**3

    def _slip_factor_gamma_radial(self) -> float:
        """Slip factor γ for radial impellers."""
        f1 = 0.98
        return (
            f1
            * (1.0 - (math.sin(math.radians(self.beta2B_deg)))**0.5 / (self.zLa**0.7))
            * self._kw()
        )

    def _eta_opt(self, Q: float, m_eta: float) -> float:
        """Optimum hydraulic efficiency correlation."""
        R = self.Q_ref / Q
        return (
            1.0
            - 0.095 * (R ** m_eta)
            - 0.3 * (0.35 - math.log10(self.n_q / 23.0))**2 * (R ** 0.05)
        )

    # ----------------- head calculation: meanline model --------------------
    def H_meanline(self, Q: float, N_rpm: float) -> float:
        """
        Delivered head H [m] from Gülich-style meanline correlations.
        Also updates self.eta_worst (used later in power_W()).
        """
        # 1) leakage-adjusted flow for triangles
        QLa = Q + self.QS3 + self.QSp

        # Rough scaling exponent for efficiency correlation
        if Q <= 1.0:
            a = 1.0
        else:
            a = 0.5

        m_eta = 0.1 * a * (self.Q_ref / Q)**0.15 * (45.0 / self.n_q)**0.06

        # 2) inlet triangle
        u1 = self._u(self.d1, N_rpm)
        A1 = self._A1()
        c1m = QLa / A1
        tan_alpha1 = math.tan(math.radians(self.alpha1_deg))
        c1u = c1m / tan_alpha1

        beta1 = math.atan(c1m / (u1 - c1u))
        tau1 = self._tau_inlet(beta1)
        _beta1_p = math.atan((c1m * tau1) / (u1 - c1u))  # not used further but left for clarity

        # 3) outlet triangle
        u2 = self._u(self.d2, N_rpm)
        c2m = QLa / (math.pi * self.d2 * self.b2)
        tau2 = self._tau_outlet()
        gamma = self._slip_factor_gamma_radial()
        tan_beta2B = math.tan(math.radians(self.beta2B_deg))

        c2u = u2 * (gamma - (c2m * tau2) / (u2 * tan_beta2B))

        # 4) theoretical Euler head and hydraulic efficiency
        H_th = (u2 * c2u - u1 * c1u) / g
        eta_opt = self._eta_opt(Q, m_eta)
        eta_delta = 0.2 * (1.0 - eta_opt)
        self.eta_worst = eta_opt - eta_delta

        return H_th

        # ----------------- internal helpers for curve model -------------------
    @staticmethod
    def _interp_1d(Q: float, Qs: list[float], Hs: list[float]) -> float:
        """
        Piecewise-linear interpolation H(Q) on a single RPM curve.
        Clamp to endpoints outside the data range.
        """
        if not Qs:
            return 0.0

        # Below minimum Q -> clamp
        if Q <= Qs[0]:
            return Hs[0]
        # Above maximum Q -> clamp
        if Q >= Qs[-1]:
            return Hs[-1]

        # Find segment [Q_i, Q_{i+1}] containing Q
        for i in range(len(Qs) - 1):
            q0, q1 = Qs[i], Qs[i + 1]
            if q0 <= Q <= q1:
                h0, h1 = Hs[i], Hs[i + 1]
                t = (Q - q0) / (q1 - q0)
                return h0 + t * (h1 - h0)

        # Fallback (should not happen)
        return Hs[-1]

    def H_curve(self, Q: float, N_rpm: float) -> float:
        """
        Head H [m] using a multi-RPM Q–H map with bilinear interpolation:

          1. Find two RPM curves bracketing N_rpm: N_lo <= N_rpm <= N_hi.
          2. Interpolate in Q on each curve to get H_lo(Q), H_hi(Q).
          3. Interpolate in N between H_lo and H_hi.

        If N_rpm is outside the tabulated range, we clamp to the nearest
        available RPM curve.
        """
        if not self.curve_map:
            raise ValueError("head_model 'curve' selected, but curve_map is empty.")

        # Sorted list of available pump speeds
        speeds = sorted(self.curve_map.keys())

        # Clamp N_rpm to the available range
        if N_rpm <= speeds[0]:
            N_lo = N_hi = speeds[0]
        elif N_rpm >= speeds[-1]:
            N_lo = N_hi = speeds[-1]
        else:
            # Find bracketing speeds
            N_lo = speeds[0]
            N_hi = speeds[-1]
            for i in range(len(speeds) - 1):
                if speeds[i] <= N_rpm <= speeds[i + 1]:
                    N_lo = speeds[i]
                    N_hi = speeds[i + 1]
                    break

        # Interpolate in Q on the lower-speed curve
        Q_lo = self.curve_map[N_lo]["Q"]
        H_lo_curve = self.curve_map[N_lo]["H"]
        H_lo = self._interp_1d(Q, Q_lo, H_lo_curve)

        # If N_lo == N_hi, we're clamped; no interpolation in N.
        if N_lo == N_hi:
            return H_lo

        # Interpolate in Q on the upper-speed curve
        Q_hi = self.curve_map[N_hi]["Q"]
        H_hi_curve = self.curve_map[N_hi]["H"]
        H_hi = self._interp_1d(Q, Q_hi, H_hi_curve)

        # Linear interpolation in N between the two heads
        tN = (N_rpm - N_lo) / (N_hi - N_lo)
        return H_lo + tN * (H_hi - H_lo)


        # ----------------- unified head + power API ----------------------------
    def H(self, Q: float, N_rpm: float) -> float:
        """
        Unified head method:

          - If head_model == "meanline": use the Gülich meanline script.
          - If head_model == "curve":   use the multi-RPM pump map.
        """
        if self.head_model == "curve":
            return self.H_curve(Q, N_rpm)
        else:
            return self.H_meanline(Q, N_rpm)

    def power_W(self, Q: float, N_rpm: float, rho: float) -> float:
        """
        Shaft power [W] required to deliver flow Q at head H.

        In meanline mode, η_worst is computed from correlations.
        In curve mode, η_worst defaults to 0.8 (you can override this after
        construction if you have a better efficiency estimate).
        """
        H = self.H(Q, N_rpm)
        P_hyd = rho * g * Q * H
        return P_hyd / self.eta_worst



# ---------------------------------------------------------------------------
# TCA liner loss component (optional)
# ---------------------------------------------------------------------------
class TCALinerLoss(Component):
    """
    Optional component representing additional pressure loss between the
    chamber bulk and the nozzle throat (e.g. through TCA liner cooling
    passages or manifolds).

    Model used:
        Δp = K * (ρ V^2 / 2)

    where V is based on an equivalent hydraulic diameter D_hyd_m.
    """
    def __init__(
        self,
        K: float,
        D_hyd_m: float,
        active: bool = True,
        name: str | None = None,
    ):
        super().__init__(name)
        self.K = float(K)
        self.D_hyd_m = float(D_hyd_m)
        self.active = bool(active)

    def dp(self, mdot: float, rho: float) -> float:
        """
        Compute pressure loss [Pa] given a total mass flow mdot [kg/s] and
        mean density rho [kg/m^3] in the liner region.
        """
        if not self.active:
            return 0.0

        A = HydroUtils.area(self.D_hyd_m)
        V = mdot / (rho * A)
        return self.K * 0.5 * rho * V * V


# ---------------------------------------------------------------------------
# Nozzle + TCA (moved from nozzle.py into this file)
# ---------------------------------------------------------------------------
def mdot_choked_perfect_gas(
    p0: float,
    T0: float,
    gamma: float,
    R: float,
    A_t: float,
    Cd: float,
) -> float:
    """
    Choked mass flow for a perfect gas:

        m_dot = Cd * A_t * p0 * sqrt( γ / (R T0) ) *
                (2 / (γ + 1))^{(γ + 1) / [2 (γ - 1)]}
    """
    term = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return Cd * A_t * p0 * math.sqrt(gamma / (R * T0)) * term


class NozzleSpec:
    """
    Specification for a simple axisymmetric nozzle.
    """
    def __init__(
        self,
        oxName: str,
        fuelName: str,
        throat_diameter_m: float,
        area_ratio: float,
        Cd_throat: float,
        c_star_eff: float,
        p_amb_Pa: float,
    ):
        self.oxName = oxName
        self.fuelName = fuelName
        self.throat_diameter_m = float(throat_diameter_m)
        self.area_ratio = float(area_ratio)
        self.Cd_throat = float(Cd_throat)
        self.c_star_eff = float(c_star_eff)
        self.p_amb_Pa = float(p_amb_Pa)

    @property
    def A_t(self) -> float:
        """Throat area [m^2]."""
        r = 0.5 * self.throat_diameter_m
        return math.pi * r * r

    @property
    def A_e(self) -> float:
        """Exit area [m^2]."""
        return self.area_ratio * self.A_t


# CEA import is only needed for the TCA; keep it localized.
try:
    from rocketcea.cea_obj import CEA_Obj
except ImportError as exc:
    CEA_Obj = None  # type: ignore


class TCA(Component):
    """
    Thrust Chamber Assembly (TCA).

    Responsibilities:
      * Given mdot_f and mdot_o, back-solve a consistent chamber pressure Pc
        using RocketCEA + a choked-flow model.
      * Compute exit pressure and thrust for the ambient pressure in NozzleSpec.
      * Optionally include additional liner loss via a TCALinerLoss component.

    Note: TCA is *not* used directly as a network component; it just provides
    a clean nozzle API for the outer coupling.
    """
    def __init__(
        self,
        spec: NozzleSpec,
        liner_loss: Optional[TCALinerLoss] = None,
        name: str | None = None,
    ):
        super().__init__(name)
        if CEA_Obj is None:
            raise ImportError(
                "rocketcea is required for TCA. "
                "Install it or stub out TCA if you only want hydraulics."
            )

        self.spec = spec
        self.liner_loss = liner_loss
        self._cea = CEA_Obj(oxName=spec.oxName, fuelName=spec.fuelName)

    # ---------------------------------------------------------------------
    # Internal helper: consistency function for Pc
    # ---------------------------------------------------------------------
    def _pc_consistency(self, mdot_f: float, mdot_o: float, Pc_pa: float) -> float:
        """
        Consistency function g(Pc) = m_dot_choked(Pc) - (m_dot_f + m_dot_o).

        A root of g(Pc) gives a Pc such that the choked mass flow equals the
        total incoming mass flow.
        """
        mdot_tot = mdot_f + mdot_o
        MR = mdot_o / mdot_f
        pc_psi = pa2psi(Pc_pa)

        # Thermodynamic state from CEA at the throat
        MW, gam = self._cea.get_Throat_MolWt_gamma(Pc=pc_psi, MR=MR, eps=self.spec.area_ratio)
        R = 8.314 / (MW / 1000.0)   # [J/kg-K]
        Tc_R = self._cea.get_Tcomb(Pc=pc_psi, MR=MR)  # combustion temp in °R
        Tc_K = Tc_R * 5.0 / 9.0                      # convert to K

        # Effective stagnation pressure used for the choke condition
        # (optionally reduced by liner loss).
        Pc_choke = Pc_pa
        if self.liner_loss is not None:
            rho_chamber = Pc_pa / (R * Tc_K)  # ideal gas approximation
            dp_liner = self.liner_loss.dp(mdot_tot, rho_chamber)
            # Ensure we don't drive Pc_choke non-physical
            Pc_choke = max(Pc_pa - dp_liner, 1.0)

        mdot_hat = mdot_choked_perfect_gas(
            p0=Pc_choke,
            T0=Tc_K,
            gamma=gam,
            R=R,
            A_t=self.spec.A_t,
            Cd=self.spec.Cd_throat,
        )

        return mdot_hat - mdot_tot

    # ---------------------------------------------------------------------
    # Public API: solve for Pc and compute thrust
    # ---------------------------------------------------------------------
    def backsolve_pc_and_thrust(
        self,
        mdot_f: float,
        mdot_o: float,
    ) -> Tuple[float, float, float, float]:
        """
        Given fuel and oxidizer mass flows, solve for:

            Pc      : chamber pressure [Pa]
            MR      : mixture ratio (oxidizer / fuel)
            p_exit  : exit static pressure [Pa]
            thrust  : thrust [N]
        """
        mdot_tot = mdot_f + mdot_o
        MR = mdot_o / mdot_f

        # --- 1) bracket Pc and find root of g(Pc) -------------------------
        a, b = 5.0e5, 2.0e7
        ga = self._pc_consistency(mdot_f, mdot_o, a)
        gb = self._pc_consistency(mdot_f, mdot_o, b)

        if ga * gb > 0.0:
            # If default bracket failed, build an estimate from c*.
            cstar_m = self._cea.get_Cstar(Pc=pa2psi(2.0e6), MR=MR) * 0.3048
            pc_guess = (self.spec.c_star_eff * cstar_m) * mdot_tot / self.spec.A_t
            a, b = 0.5 * pc_guess, 2.0 * pc_guess

        for _ in range(80):
            mid = 0.5 * (a + b)
            gm = self._pc_consistency(mdot_f, mdot_o, mid)
            if abs(gm) < 1e-5:
                Pc_pa = mid
                break

            ga = self._pc_consistency(mdot_f, mdot_o, a)
            if ga * gm < 0.0:
                b = mid
            else:
                a = mid
        else:
            # fallback if we somehow didn't converge
            Pc_pa = mid

        # --- 2) Exit conditions and thrust from CEA -----------------------
        pc_psi = pa2psi(Pc_pa) * self.spec.c_star_eff

        # Exit Mach number
        M_exit = self._cea.get_MachNumber(
            Pc=pc_psi,
            MR=MR,
            eps=self.spec.area_ratio,
            frozen=1,
            frozenAtThroat=1,
        )

        # Use throat gamma to estimate p_exit via isentropic relation
        MW_th, gam_th = self._cea.get_Throat_MolWt_gamma(
            Pc=pc_psi,
            MR=MR,
            eps=self.spec.area_ratio,
        )
        p_exit_pa = Pc_pa * (1.0 + (gam_th - 1.0) / 2.0 * M_exit**2) ** (-gam_th / (gam_th - 1.0))

        # Exit velocity: CeA returns sonic velocity in ft/s; last entry is exit
        a_exit_fps = self._cea.get_SonicVelocities(
            Pc=pc_psi,
            MR=MR,
            eps=self.spec.area_ratio,
            frozen=1,
            frozenAtThroat=1,
        )[-1]
        v_exit = M_exit * (a_exit_fps * 0.3048)  # [m/s]

        thrust_N = mdot_tot * v_exit + (p_exit_pa - self.spec.p_amb_Pa) * self.spec.A_e

        return Pc_pa, MR, p_exit_pa, thrust_N
