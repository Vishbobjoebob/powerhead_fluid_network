# nozzle.py
from __future__ import annotations
import math
from typing import Tuple

from components import HydroUtils as HU

from rocketcea.cea_obj import CEA_Obj  # ensure rocketcea is installed

class NozzleSpec:
    def __init__(self, oxName, fuelName, throat_diameter_m, area_ratio, Cd_throat, c_star_eff, p_amb_Pa):
        self.oxName = oxName
        self.fuelName = fuelName
        self.throat_diameter_m = float(throat_diameter_m)
        self.area_ratio = float(area_ratio)
        self.Cd_throat = float(Cd_throat)
        self.c_star_eff = float(c_star_eff)
        self.p_amb_Pa = float(p_amb_Pa)

    @property
    def A_t(self) -> float:
        r = 0.5 * self.throat_diameter_m
        return math.pi * r * r

    @property
    def A_e(self) -> float:
        return self.area_ratio * self.A_t

def mdot_choked_perfect_gas(p0: float, T0: float, gamma: float, R: float, A_t: float, Cd: float) -> float:
    term = (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    return Cd * A_t * p0 * math.sqrt(gamma / (R * T0)) * term

class TCA:
    """
    Thrust Chamber Assembly: solves Pc from mdots with RocketCEA and computes thrust (ambient as final boundary).
    """
    def __init__(self, spec: NozzleSpec):
        self.spec = spec
        self._cea = CEA_Obj(oxName=spec.oxName, fuelName=spec.fuelName)

    def _pc_consistency(self, mdot_f: float, mdot_o: float, Pc_pa: float) -> float:
        mdot_tot = mdot_f + mdot_o
        pc_psi = HU.pa2psi(Pc_pa)
        MR = mdot_o / mdot_f
        MW, gam = self._cea.get_Throat_MolWt_gamma(Pc=pc_psi, MR=MR, eps=self.spec.area_ratio)
        R = 8.314 / (MW / 1000.0)
        Tc_R = self._cea.get_Tcomb(Pc=pc_psi, MR=MR)  # °R
        Tc_K = Tc_R * 5.0 / 9.0
        mdot_hat = mdot_choked_perfect_gas(Pc_pa, Tc_K, gam, R, self.spec.A_t, self.spec.Cd_throat)
        return mdot_hat - mdot_tot

    def backsolve_pc_and_thrust(self, mdot_f: float, mdot_o: float) -> Tuple[float, float, float, float]:
        mdot_tot = mdot_f + mdot_o
        MR = mdot_o / mdot_f

        a, b = 5.0e5, 2.0e7
        ga = self._pc_consistency(mdot_f, mdot_o, a)
        gb = self._pc_consistency(mdot_f, mdot_o, b)
        if ga * gb > 0.0:
            cstar_m = self._cea.get_Cstar(Pc=HU.pa2psi(2.0e6), MR=MR) * 0.3048
            pc_guess = (self.spec.c_star_eff * cstar_m) * mdot_tot / self.spec.A_t
            a, b = 0.5 * pc_guess, 2.0 * pc_guess

        for _ in range(80):
            mid = 0.5 * (a + b)
            gm = self._pc_consistency(mdot_f, mdot_o, mid)
            if abs(gm) < 1e-5:
                Pc_pa = mid
                break
            if self._pc_consistency(mdot_f, mdot_o, a) * gm < 0.0:
                b = mid
            else:
                a = mid
        else:
            Pc_pa = mid

        pc_psi = HU.pa2psi(Pc_pa) * self.spec.c_star_eff
        M_exit = self._cea.get_MachNumber(Pc=pc_psi, MR=MR, eps=self.spec.area_ratio, frozen=1, frozenAtThroat=1)
        MW_th, gam_th = self._cea.get_Throat_MolWt_gamma(Pc=pc_psi, MR=MR, eps=self.spec.area_ratio)
        p_exit_pa = Pc_pa * (1.0 + (gam_th - 1.0) / 2.0 * M_exit**2) ** (-gam_th / (gam_th - 1.0))
        a_exit_fps = self._cea.get_SonicVelocities(Pc=pc_psi, MR=MR, eps=self.spec.area_ratio, frozen=1, frozenAtThroat=1)[-1]
        v_exit = M_exit * (a_exit_fps * 0.3048)

        thrust_N = mdot_tot * v_exit + (p_exit_pa - self.spec.p_amb_Pa) * self.spec.A_e
        return Pc_pa, MR, p_exit_pa, thrust_N
