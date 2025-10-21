# main.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from yaml_io import build_from_yaml
from components import pa2psi
from network import rhs_branch_inertive, line_inertance_sum
import yaml
from power_optimize import run_power_optimizer


# Point to your YAML here (no CLI needed)
CONFIG_PATH = Path("config.yaml")


def run_steady_with_tca(ctx):
    """Coupled steady solve: solves (Pc, mdot_f, mdot_ox) with TCA/nozzle and prints pump powers."""
    tca = ctx["tca"]
    if tca is None:
        print("[Steady] No TCA/nozzle in YAML; skipping Pc/thrust back-solve.")
        return None

    fuel = ctx["branches"]["fuel"]
    lox  = ctx["branches"]["lox"]
    Nf   = ctx["steady"]["N_fuel_rpm"]
    No   = ctx["steady"]["N_lox_rpm"]

    # 3×unknowns: Pc, mf, mo
    def F(x):
        Pc, mf, mo = x
        r1 = fuel.residual(mf, Nf, Pc)
        r2 = lox.residual(mo,  No, Pc)
        Pc_hat, _, _, _ = tca.backsolve_pc_and_thrust(mf, mo)
        return [r1, r2, Pc_hat - Pc]

    Pc0 = float(ctx["chamber"].Pc)
    mf0, mo0 = 1.0, 2.0
    Pc, mf, mo = fsolve(F, x0=[Pc0, mf0, mo0])
    Pc_final, MR, p_exit, thrust_N = tca.backsolve_pc_and_thrust(mf, mo)

    # Pump power
    Qf = mf / fuel.rho
    Qo = mo / lox.rho
    Pf = fuel.pump.power_W(Qf, Nf, fuel.rho, fuel.mu)
    Po = lox.pump.power_W(Qo, No, lox.rho,  lox.mu)
    Ptot = Pf + Po

    print("\n[Steady State Nozzle-coupled via TCA]")
    print(f"Pc = {pa2psi(Pc_final):.1f} psi   mdot_f = {mf:.4f} kg/s   mdot_ox = {mo:.4f} kg/s   MR = {MR:.3f}")
    print(f"p_exit = {pa2psi(p_exit):.2f} psi   Thrust = {thrust_N:.1f} N ({thrust_N/4.448:.1f} lbf)")
    print(f"\nPump power (steady): Fuel {Pf/1000:.2f} kW | LOX {Po/1000:.2f} kW | Total {Ptot/1000:.2f} kW")

    return {"Pc": Pc_final, "mdot_f": mf, "mdot_ox": mo, "MR": MR, "p_exit": p_exit, "thrust_N": thrust_N}


# def run_transient(ctx):
#     """Transient ramp IVP if 'transient' exists in YAML."""
#     if ctx["transient"] is None:
#         print("\n[Transient] section missing in YAML; skipping.")
#         return None

#     fuel = ctx["branches"]["fuel"]; lox = ctx["branches"]["lox"]
#     rho_f = ctx["fluids"]["fuel"]["rho"]; rho_o = ctx["fluids"]["lox"]["rho"]

#     ramps = ctx["transient"]["ramps"]
#     Nf_of_t = ramps["Nf_of_t"]; Nox_of_t = ramps["Nox_of_t"]; Pc_of_t = ramps["Pc_of_t"]

#     ivp = ctx["transient"]["ivp"]
#     t0, tf     = ivp.get("t_span", [0.0, 0.25])
#     method     = ivp.get("method", "Radau")
#     rtol       = ivp.get("rtol", 1e-6)
#     atol       = ivp.get("atol", 1e-8)
#     max_step   = ivp.get("max_step", 1e-3)
#     Q0_fuel    = ivp.get("Q0_fuel", 1e-5)
#     Q0_lox     = ivp.get("Q0_lox",  1e-5)

#     Ls_f = line_inertance_sum(fuel.suction_lines, rho_f)
#     Ld_f = line_inertance_sum(fuel.discharge_lines, rho_f)
#     Ls_o = line_inertance_sum(lox.suction_lines,  rho_o)
#     Ld_o = line_inertance_sum(lox.discharge_lines, rho_o)

#     def rhs_coupled_Q(t, x):
#         Qf, Qo = x
#         dQf = rhs_branch_inertive(t, Qf, fuel, Nf_of_t, Pc_of_t, Ls_f, Ld_f)
#         dQo = rhs_branch_inertive(t, Qo,  lox,  Nox_of_t, Pc_of_t, Ls_o, Ld_o)
#         return [dQf, dQo]

#     sol = solve_ivp(rhs_coupled_Q, (t0, tf), [Q0_fuel, Q0_lox],
#                     method=method, rtol=rtol, atol=atol, max_step=max_step)

#     t = sol.t
#     mdot_f_t = sol.y[0] * rho_f
#     mdot_ox_t = sol.y[1] * rho_o
#     Pc_t = np.array([Pc_of_t(tt) for tt in t])

#     nprint = int(ctx["transient"].get("print_samples", 12))
#     idx = np.linspace(0, t.size - 1, nprint, dtype=int)
#     print("\n[Transient] (sampled)")
#     print(f"{'t[s]':>8}  {'mdot_f[kg/s]':>14}  {'mdot_ox[kg/s]':>16}  {'Pc[psi]':>10}")
#     for i in idx:
#         print(f"{t[i]:8.5f}  {mdot_f_t[i]:14.6f}  {mdot_ox_t[i]:16.6f}  {pa2psi(Pc_t[i]):10.1f}")

#     if bool(ctx["transient"].get("plots", True)):
#         plt.figure()
#         plt.plot(t, mdot_f_t, label="Fuel mdot [kg/s]")
#         plt.plot(t, mdot_ox_t, label="LOX mdot [kg/s]")
#         plt.xlabel("Time [s]"); plt.ylabel("Mass flow [kg/s]")
#         plt.title("Transient mass flows"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
#         plt.tight_layout()

#         plt.figure()
#         plt.plot(t, Pc_t / 6895)
#         plt.xlabel("Time [s]"); plt.ylabel("Pc [psi]")
#         plt.title("Chamber pressure ramp"); plt.grid(True, linestyle="--", alpha=0.5)
#         plt.tight_layout()
#         plt.show()

#     return {"t": t, "mdot_f_t": mdot_f_t, "mdot_ox_t": mdot_ox_t, "Pc_t": Pc_t}


def main():
    ctx = build_from_yaml(CONFIG_PATH)
    if ctx["steady"].get("use_nozzle_solve", True):
        run_steady_with_tca(ctx)
    # run_transient(ctx)

    
    # --- Optimizer toggle via YAML ---
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    opt_cfg = cfg.get("optimize") or {}
    if opt_cfg.get("enabled", False):
        print("\n[Optimization] Minimize total pump power with thrust constraint...")
        rep = run_power_optimizer(ctx, opt_cfg)

        print("  Status:", "SUCCESS" if rep["success"] else "FAILED")
        print("  Message:", rep["message"])
        P = rep["power_W"]
        print(f"  Power: Fuel={P['fuel']/1000:.2f} kW  LOX={P['lox']/1000:.2f} kW  Total={P['total']/1000:.2f} kW")
        sol = rep["solution"]
        print(f"  Pc = {sol['Pc_psi']:.1f} psi   mdot_f = {sol['mdot_f']:.4f} kg/s   "
            f"mdot_ox = {sol['mdot_ox']:.4f} kg/s   MR = {sol['MR']:.3f}")
        print(f"  Thrust = {sol['thrust_N']:.1f} N ({sol['thrust_N']/4.448:.1f} lbf)")
        print("  Design variables:")
        for k, v in rep["design_vars"].items():
            print(f"    {k}: {v}")
    else:
        print("\n[Optimization] Skipped (optimize.enabled is false or missing).")

    


if __name__ == "__main__":
    main()
