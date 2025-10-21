# components.py
from __future__ import annotations
import math
import numpy as np

# ------------------------------
# Shared utilities / interface
# ------------------------------
class HydroUtils:
    PA_PER_PSI = 6894.757293168361
    g = 9.80665

    # Units
    @staticmethod
    def pa2psi(x: float) -> float:
        return float(x) / HydroUtils.PA_PER_PSI

    @staticmethod
    def psi2pa(x: float) -> float:
        return float(x) * HydroUtils.PA_PER_PSI

    # Geometry
    @staticmethod
    def area(D: float) -> float:
        return math.pi * (D**2) / 4.0

    # Kinematics / dimensionless
    @staticmethod
    def reynolds(rho: float, V: float, D: float, mu: float) -> float:
        return rho * V * D / mu

    @staticmethod
    def tip_speed(D: float, N_rps: float) -> float:
        return math.pi * D * N_rps

    # Friction
    @staticmethod
    def colebrook_white(Re: float, eps: float, D: float, f0: float = 0.02, iters: int = 30, tol: float = 1e-10) -> float:
        f = f0
        for _ in range(iters):
            inv_sqrt_f = 1.0 / math.sqrt(f)
            rhs = (eps / D) / 3.7 + 2.51 / (Re * math.sqrt(f))
            gF = inv_sqrt_f + 2.0 * math.log10(rhs)
            d_inv = -0.5 * f**(-1.5)
            d_rhs = 2.51 * (-0.5) * f**(-1.5) / Re
            dg = d_inv + 2.0 / (math.log(10.0) * rhs) * d_rhs
            f_new = f - gF / dg
            if abs(f_new - f) < tol:
                f = f_new
                break
            f = f_new
        return float(f)

    # Orifice / injector relations
    @staticmethod
    def orifice_dp_from_mdot(mdot: float, rho: float, CdA: float) -> float:
        return (mdot / CdA)**2 / (2.0 * rho)

    @staticmethod
    def orifice_mdot_from_dp(dp: float, rho: float, CdA: float) -> float:
        return CdA * math.sqrt(2.0 * rho * dp)

# Handy exports
g = HydroUtils.g
pa2psi = HydroUtils.pa2psi
psi2pa = HydroUtils.psi2pa
area = HydroUtils.area

# Base component
class Component:
    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    def dp(self, *args, **kwargs) -> float:
        raise NotImplementedError

    def inertance(self, *args, **kwargs) -> float:
        return 0.0

# Child components
class Line(Component):
    """Straight pipe (Darcy–Weisbach)."""
    def __init__(self, L: float, D: float, eps: float, name: str | None = None):
        super().__init__(name)
        self.L = float(L)
        self.D = float(D)
        self.eps = float(eps)

    def dp(self, mdot: float, rho: float, mu: float) -> float:
        A = HydroUtils.area(self.D)
        V = mdot / (rho * A)
        Re = HydroUtils.reynolds(rho, V, self.D, mu)
        if Re < 2300.0:
            f = 64.0 / Re
        else:
            f = HydroUtils.colebrook_white(Re, self.eps, self.D)
        return f * (self.L / self.D) * 0.5 * rho * V * V

    def inertance(self, rho: float) -> float:
        return rho * self.L / HydroUtils.area(self.D)

class Bend(Component):
    """Minor loss with fixed K (excess head coefficient)."""
    def __init__(self, K: float, D: float, name: str | None = None):
        super().__init__(name)
        self.K = float(K)
        self.D = float(D)

    def dp(self, mdot: float, rho: float) -> float:
        V = mdot / (rho * HydroUtils.area(self.D))
        return self.K * 0.5 * rho * V * V

class Injector(Component):
    """Simple orifice injector using CdA."""
    def __init__(self, CdA: float, name: str | None = None):
        super().__init__(name)
        self.CdA = float(CdA)

    def dp(self, mdot: float, rho: float) -> float:
        return HydroUtils.orifice_dp_from_mdot(mdot, rho, self.CdA)

    def mdot_from_dp(self, dp: float, rho: float) -> float:
        return HydroUtils.orifice_mdot_from_dp(dp, rho, self.CdA)

# Radial-only Gulich pump
class GulichPump(Component):
    """
    Radial centrifugal pump per Gülich.
    """
    def __init__(self,
                 # geometry
                 d1, dn, d2, b2, zLa, beta2B_deg, n_q,
                 alpha1_deg=90.0, e1=0.0, e=0.0, lambdaLa_deg=90.0,
                 # efficiencies / leakage
                 QS3=0.0, QSp=0.0,
                 # sheet parametersF
                 Q_ref=1.0,
                 name: str | None = None):
        super().__init__(name)
        # geometry
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
        # efficiencies / leakage
        self.QS3 = float(QS3)
        self.QSp = float(QSp)
        # sheet parameters
        self.Q_ref = float(Q_ref)
        self.n_q = float(n_q)

    # ---- helpers from tables 3.1 / 3.2 (no guardrails) ----
    def _u(self, d, n_rpm):
        return math.pi * d * (n_rpm / 60.0)

    def _A1(self):
        return (math.pi / 4.0) * (self.d1**2 - self.dn**2)

    def _tau_inlet(self, beta1):
        s1 = math.sin(beta1)
        sL = math.sin(math.radians(self.lambdaLa_deg))
        return 1.0 / (1.0 - (self.zLa * self.e1) / (math.pi * self.d1 * s1 * sL))

    def _tau_outlet(self):
        s2B = math.sin(math.radians(self.beta2B_deg))
        sL  = math.sin(math.radians(self.lambdaLa_deg))
        return 1.0 / (1.0 - (self.e * self.zLa) / (math.pi * self.d2 * s2B * sL))

    def _kw(self):
        s2B = math.sin(math.radians(self.beta2B_deg))
        eps_lim = math.e ** ( - 8.16 * s2B / self.zLa )
        return 1.0 - ((self.d1m_star - eps_lim) / (1.0 - eps_lim))**3

    def _slip_factor_gamma_radial(self):
        f1 = 0.98
        return f1 * (1.0 - (math.sin(math.radians(self.beta2B_deg)))**0.5 / (self.zLa**0.7)) * self._kw()

    def _eta_opt(self, Q, m_eta):
        R = self.Q_ref / Q
        return 1.0 - 0.095 * (R ** m_eta) - 0.3 * (0.35 - math.log10(self.n_q / 23.0))**2 * (R ** 0.05)

    # ---- public API aligned with your network (H(Q,N,rho,mu)) ----
    def H(self, Q: float, N_rpm: float, rho: float, mu: float) -> float:
        """
        Returns delivered head H [m] using triangles + η_h sheet formula.
        Q here is delivered flow (m^3/s) = mdot/rho.
        n_q used is the reference n_q you supplied at construction.
        """
        # leakage-adjusted flow for triangles
        QLa = Q + self.QS3 + self.QSp

        if Q <= 1:
            a = 1
        else:
            a = 0.5

        m_eta = 0.1 * a * (self.Q_ref / Q)**0.15 * (45/self.n_q)**0.06

        # Inlet triangle
        u1 = self._u(self.d1, N_rpm)
        A1 = (math.pi / 4.0) * (self.d1**2 - self.dn**2)
        c1m = QLa / A1
        tan_alpha1 = math.tan(math.radians(self.alpha1_deg))
        c1u = c1m / tan_alpha1
        beta1 = math.atan( c1m / (u1 - c1u) )
        tau1 = self._tau_inlet(beta1)
        _beta1_p = math.atan( (c1m * tau1) / (u1 - c1u) )

        # Outlet triangle
        u2 = self._u(self.d2, N_rpm)
        c2m = QLa / (math.pi * self.d2 * self.b2)
        tau2 = self._tau_outlet()
        gamma = self._slip_factor_gamma_radial()
        tan_beta2B = math.tan(math.radians(self.beta2B_deg))
        c2u = u2 * ( gamma - (c2m * tau2) / (u2 * tan_beta2B) )

        # Euler head and hydraulic efficiency
        H_th = (u2 * c2u - u1 * c1u) / g
        eta_opt = self._eta_opt(Q, m_eta)
        eta_delta = 0.2 * (1-eta_opt)
        self.eta_worst = eta_opt - eta_delta

        return H_th

    def power_W(self, Q: float, N_rpm: float, rho: float, mu: float) -> float:
        """Shaft power using η_h (sheet), η_v from leakage, and η_mech."""
        H = self.H(Q, N_rpm, rho, mu)
        P_hyd = rho * g * Q * H
        return P_hyd / self.eta_worst
