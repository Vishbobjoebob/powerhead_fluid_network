import numpy as np

# constants
g = 9.80665
PA_PER_PSI = 6894.757293168361

# helpers
def pa2psi(x):
    return x / PA_PER_PSI

def psi2pa(x):
    return x * PA_PER_PSI

def area(D):
    return 0.25 * np.pi * D**2

# Colebrook-White
def colebrook_white(Re, eps, D, f0 = 0.02, iters = 30, tol = 1e-10):
    """
    Returns Darcy-Weisbach friction factor f
    """
    f = f0
    for _ in range(iters):
        inv_sqrt_f = 1.0 / np.sqrt(f)
        rhs = (eps / D) / 3.7 + 2.51 / (Re * np.sqrt(f))
        gF = inv_sqrt_f + 2.0 * np.log10(rhs)
        d_inv = -0.5 * f**(-1.5)
        d_rhs = 2.51 * (-0.5) * f**(-1.5) / Re
        dg = d_inv + 2.0 / (np.log(10.0) * rhs) * d_rhs
        f_new = f - gF / dg
        if abs(f_new - f) < tol:
            f = f_new
            break
        f = f_new
    return float(f)

# hydraulic components
class Line:
    """
    Straight pipe segment modeled with Darcy–Weisbach
    """
    def __init__(self, L, D, eps):
        self.L = float(L)     # m
        self.D = float(D)     # m
        self.eps = float(eps)

    def dp(self, mdot, rho, mu):
        """
        Pressure drop across the line.
        """
        A = area(self.D)
        V = mdot / (rho * A)
        Re = rho * abs(V) * self.D / mu
        if Re < 2300.0:
            f = 64.0 / Re
        else:
            f = colebrook_white(Re, self.eps, self.D)
        return f * (self.L / self.D) * 0.5 * rho * V * V

    def inertance(self, rho):
        """
        Hydraulic inertance L_h = rho * L / A.
        """
        return rho * self.L / area(self.D)

class Bend:
    """
    Minor loss element with fixed K (excess head coefficient)
    """
    def __init__(self, K, D):
        self.K = float(K)   # dimensionless
        self.D = float(D)   # m

    def dp(self, mdot, rho):
        """
        dp = K * 0.5 * rho * V^2
        """
        V = mdot / (rho * area(self.D))
        return self.K * 0.5 * rho * V * V

class Injector:
    """
    Calculates dp with CdA, mdot, and rho.
    """
    def __init__(self, CdA):
        self.CdA = float(CdA)  # m^2

    def dp(self, mdot, rho):
        """
        Δp = (1/(2 rho)) * (mdot/CdA)^2
        """
        return (mdot / self.CdA) ** 2 / (2.0 * rho)

class MeanlineGeom:
    def __init__(self, D2, b2, Z, beta2_deg):
        self.D2 = float(D2)           # m
        self.b2 = float(b2)           # m
        self.Z = int(Z)               # number of blades
        self.beta2_deg = float(beta2_deg)  # deg

class MeanlineLoss:
    '''
    Estimates of pump loss coefficients
    '''
    def __init__(self, k_inc = 0.0, k_prof = 0.04, k_disc = 0.01, k_clear = 0.01, k_diff = 0.02,
                 eta_mech = 0.985, eta_vol = 0.995):
        self.k_inc = float(k_inc)
        self.k_prof = float(k_prof)
        self.k_disc = float(k_disc)
        self.k_clear = float(k_clear)
        self.k_diff = float(k_diff)
        self.eta_mech = float(eta_mech)
        self.eta_vol = float(eta_vol)

class MeanlinePump:
    """
    Compact impeller meanline with RPM input
    """
    def __init__(self, geom, loss):
        self.geom = geom
        self.loss = loss

    def _stage(self, Q, N_rpm, rho, mu):
        D2, b2, Z = self.geom.D2, self.geom.b2, self.geom.Z
        beta2 = np.deg2rad(self.geom.beta2_deg)
        r2 = D2 / 2.0
        U2 = np.pi * D2 * (N_rpm / 60.0)                 # m/s
        Vr2 = Q / (2.0 * np.pi * r2 * b2)                # m/s
        Vth2_ideal = U2 - Vr2 / np.tan(beta2)            # m/s
        phi2 = Vr2/U2
        sigma = 1.0 - (np.pi / Z) * np.sin(beta2) / (1 - phi2*(1/np.tan(beta2)))      # Stodola's Equation
        Vth2 = sigma * Vth2_ideal
        Hideal = U2 * Vth2 / g                           # Euler head (m)
        V2_sq = Vr2**2 + Vth2**2
        Hloss = (self.loss.k_prof * V2_sq + self.loss.k_disc * U2**2 + self.loss.k_clear * U2**2) / (2.0 * g) \
                + self.loss.k_diff * Vr2**2 / (2.0 * g)
        H = Hideal - Hloss
        eta_h = H / Hideal
        eta_tot = float(np.clip(eta_h * self.loss.eta_mech * self.loss.eta_vol, 0.0, 0.99))
        Pshaft = rho * g * Q * H / max(eta_tot, 1e-6)    # W
        return H, eta_tot, Pshaft

    def H(self, Q, N_rpm, rho, mu):
        return self._stage(Q, N_rpm, rho, mu)[0]

    def eta(self, Q, N_rpm, rho, mu):
        return self._stage(Q, N_rpm, rho, mu)[1]
