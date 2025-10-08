import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from components import MeanlineGeom, MeanlineLoss, MeanlinePump, Line, Bend, Injector, pa2psi, psi2pa
from network import Tank, Chamber, Branch, solve_with_MR, solve_given_Pc, rhs_branch_inertive, line_inertance_sum

# Tanks
tank_f = Tank(p = psi2pa(140.0))  # Pa
tank_o = Tank(p = psi2pa(140.0))  # Pa

# Geometry (add more lines and bends to the list)

# Suction sides
suction_lines_f = [Line(L = 0.026 * 5, D = 0.026, eps = 5e-5)]
suction_bends_f = [Bend(K = 0.50, D = 0.026)]
suction_lines_o = [Line(L = 0.026 * 5, D = 0.026, eps = 5e-5)]
suction_bends_o = [Bend(K = 0.50, D = 0.026)]

# Discharge sides (L [m], D [m])
discharge_lines_f = [Line(L = 0.026 * 5, D = 0.026, eps = 5e-5)]
discharge_bends_f = [Bend(K = 0.50, D = 0.026)]
discharge_lines_o = [Line(L = 0.026 * 5, D = 0.026, eps = 5e-5)]
discharge_bends_o = [Bend(K = 0.50, D = 0.026)]

# Pumps (D2 [m], b2 [m], Z [int], beta2_deg [deg])
mlF = MeanlinePump(MeanlineGeom(D2 = 0.07, b2 = 0.006, Z = 6, beta2_deg = 22), MeanlineLoss())
mlO = MeanlinePump(MeanlineGeom(D2 = 0.07, b2 = 0.006, Z = 6, beta2_deg = 22), MeanlineLoss())

# Injectors
inj_f = Injector(CdA = 2e-5)  # m^2
inj_o = Injector(CdA = 2e-5)  # m^2

# Chamber (steady state)
chamber = Chamber(Pc = psi2pa(545.0))  # Pa (not used directly in transient function)

# Fluids
rho_f, mu_f = 800.0, 2.0e-3    # kg/m^3, Pa·s
rho_o, mu_o = 1140.0, 2.0e-3   # kg/m^3, Pa·s

# Branches
fuel = Branch(
    rho = rho_f, 
    mu = mu_f,
    tank = tank_f,
    suction_lines = suction_lines_f, suction_bends = suction_bends_f,
    pump = mlF,
    discharge_lines = discharge_lines_f, discharge_bends = discharge_bends_f,
    injector = inj_f
)
lox = Branch(
    rho = rho_o, 
    mu = mu_o,
    tank = tank_o,
    suction_lines = suction_lines_o, suction_bends = suction_bends_o,
    pump = mlO,
    discharge_lines = discharge_lines_o, discharge_bends = discharge_bends_o,
    injector = inj_o
)

# RPM and Pc Ramps (transient)
Nf0, Nox0 = 10000.0, 10000.0     # RPM
Nf1, Nox1 = 20000.0, 20000.0     # RPM
Pc0, Pc1 = psi2pa(0.0), psi2pa(400.0)  # Pa

t0, t1 = 0.0, 0.01   # speed ramp window [s]
t0a, t1a = 0.0, 0.10 # Pc ramp window [s]

def ramp(a0, a1, t0, t1, t):
    if t <= t0:
        return a0
    if t >= t1:
        return a1
    return a0 + (a1 - a0) * (t - t0) / (t1 - t0)

def Nf_of_t(t): return ramp(Nf0, Nf1, t0,  t1,  t)
def Nox_of_t(t): return ramp(Nox0, Nox1, t0,  t1,  t)
def Pc_of_t(t): return ramp(Pc0, Pc1, t0a, t1a, t)

# --- steady checks -----------------------------------------------------------
res_mr = solve_with_MR(fuel, lox, Nf1, Nox1, MR = 2.23)
print(f"MR solve: Pc = {pa2psi(res_mr['Pc']):.1f} psi  mdot_f = {res_mr['mdot_f']:.4f} kg/s  "
      f"mdot_ox = {res_mr['mdot_ox']:.4f} kg/s  MR = {res_mr['MR']:.3f}")

res_pc = solve_given_Pc(fuel, lox, Nf1, Nox1, Pc = Pc0)
print(f"Pc solve: Pc = {pa2psi(res_pc['Pc']):.1f} psi  mdot_f = {res_pc['mdot_f']:.4f} kg/s  "
      f"mdot_ox = {res_pc['mdot_ox']:.4f} kg/s  MR = {res_pc['MR']:.3f}")

# --- inertances (sum over arbitrary line lists) ------------------------------
Ls_f = line_inertance_sum(fuel.suction_lines, rho_f)
Ld_f = line_inertance_sum(fuel.discharge_lines, rho_f)
Ls_o = line_inertance_sum(lox.suction_lines,  rho_o)
Ld_o = line_inertance_sum(lox.discharge_lines, rho_o)



def rhs_coupled_Q(t, x):
    Qf, Qo = x
    dQf = rhs_branch_inertive(t, Qf, fuel, Nf_of_t, Pc_of_t, Ls_f, Ld_f)
    dQo = rhs_branch_inertive(t, Qo,  lox,  Nox_of_t, Pc_of_t, Ls_o, Ld_o)
    return [dQf, dQo]

x0 = [1e-5, 1e-5] # initial mass flows 
sol = solve_ivp(rhs_coupled_Q, (0.0, 0.25), x0, method = "Radau",
                rtol = 1e-6, atol = 1e-8, max_step = 1e-3)

# --- reporting & plots -------------------------------------------------------
t = sol.t
mdot_f_t = sol.y[0] * rho_f   # kg/s
mdot_ox_t = sol.y[1] * rho_o  # kg/s
Pc_t = np.array([Pc_of_t(tt) for tt in t])

idx = np.linspace(0, t.size - 1, 12, dtype = int)
print("\nTransient (sampled):")
print(f"{'t[s]':>8}  {'mdot_f[kg/s]':>14}  {'mdot_ox[kg/s]':>16}  {'Pc[psi]':>10}")
for i in idx:
    print(f"{t[i]:8.5f}  {mdot_f_t[i]:14.6f}  {mdot_ox_t[i]:16.6f}  {pa2psi(Pc_t[i]):10.1f}")

plt.figure()
plt.plot(t, mdot_f_t, label = "Fuel mdot [kg/s]")
plt.plot(t, mdot_ox_t, label = "LOX mdot [kg/s]")
plt.xlabel("Time [s]")
plt.ylabel("Mass flow [kg/s]")
plt.title("Transient mass flows (linear ramps)")
plt.legend()
plt.grid(True, which = "both", linestyle = "--", alpha = 0.5)
plt.tight_layout()

plt.figure()
plt.plot(t, pa2psi(Pc_t))
plt.xlabel("Time [s]")
plt.ylabel("Pc [psi]")
plt.title("Chamber pressure ramp")
plt.grid(True, which = "both", linestyle = "--", alpha = 0.5)
plt.tight_layout()

plt.show()
