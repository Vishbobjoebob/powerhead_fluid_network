# main.py
from __future__ import annotations

from pathlib import Path
import yaml

from components import pa2psi
from yaml_io import build_from_yaml
from solvers import solve_coupled_fixed_point

# Path to the master configuration file
CONFIG_PATH = Path("config.yaml")

def print_branch_pressures(branch, branch_name: str, mdot: float, N_rpm: float, Pc: float) -> None:
    """
    Pretty-print the pressure at every node in a branch.

    Shows absolute pressure in Pa and psi, before/after each component.
    """
    from components import pa2psi  # local import to avoid cycles

    nodes = branch.pressure_profile(mdot, N_rpm, Pc)

    print(f"\n[{branch_name} branch pressure profile]  (Pa / psi)")
    for label, p in nodes:
        print(f"  {label:35s}: {p:12.1f} Pa  ({pa2psi(p):9.3f} psi)")

    # Also print the chamber pressure we used to solve
    print(f"  {'chamber_Pc (solve input)':35s}: {Pc:12.1f} Pa  ({pa2psi(Pc):9.3f} psi)")



def main() -> None:
    """
    Entry point for the steady-state powerhead solve + optional optimization.

    Steps:
      1) Build the full hydraulic + pump + nozzle model from YAML.
      2) Solve a coupled steady state for Pc, mdot_f, mdot_ox, thrust.
      3) Optionally run a power optimizer (local or global) controlled by YAML.
    """
    # ------------------------------------------------------------
    # 1) Build model from YAML
    # ------------------------------------------------------------
    ctx = build_from_yaml(CONFIG_PATH)

    # ------------------------------------------------------------
    # 2) Coupled steady-state solve (no transients)
    # ------------------------------------------------------------
    res = solve_coupled_fixed_point(
        ctx,
        omega=0.6,      # under-relaxation on Pc update (0 < ω ≤ 1)
        tol_rel=1e-5,   # relative convergence tolerance on Pc
        tol_abs=500.0,  # absolute convergence tolerance on Pc [Pa]
        max_iter=50,
    )

    # Convenience names
    fuel_branch = ctx["branches"]["fuel"]
    lox_branch = ctx["branches"]["lox"]

    # Convert mass flows to volume flows for pump power
    Qf = res["mdot_f"] / fuel_branch.rho  # [m^3/s]
    Qo = res["mdot_ox"] / lox_branch.rho  # [m^3/s]

    Pf = fuel_branch.pump.power_W(Qf, ctx["steady"]["N_fuel_rpm"], fuel_branch.rho)
    Po = lox_branch.pump.power_W(Qo, ctx["steady"]["N_lox_rpm"], lox_branch.rho)

    print(
        f"\n[Steady Coupled] "
        f"Pc = {pa2psi(res['Pc']):.2f} psi, "
        f"mdot_f = {res['mdot_f']:.5f} kg/s, "
        f"mdot_ox = {res['mdot_ox']:.5f} kg/s, "
        f"MR = {res['MR']:.3f}, "
        f"Thrust = {res['thrust_N']:.1f} N;  "
        f"Fuel Pump Power = {Pf/1000:.2f} kW; "
        f"Ox Pump Power = {Po/1000:.2f} kW"
    )

    Pc_solved = res["Pc"]

    print_branch_pressures(
        branch=fuel_branch,
        branch_name="Fuel",
        mdot=res["mdot_f"],
        N_rpm=ctx["steady"]["N_fuel_rpm"],
        Pc=Pc_solved,
    )

    print_branch_pressures(
        branch=lox_branch,
        branch_name="LOX",
        mdot=res["mdot_ox"],
        N_rpm=ctx["steady"]["N_lox_rpm"],
        Pc=Pc_solved,
    )


if __name__ == "__main__":
    main()
