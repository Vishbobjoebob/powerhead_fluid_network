# optimizer.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import copy
import numpy as np
from scipy.optimize import fsolve, minimize

from components import pa2psi
from network import Branch
from nozzle import TCA

from yaml_io import _eval_if_expr


# ----------------- core steady coupled solve -----------------
def _steady_coupled_solve(ctx: Dict[str, Any]) -> Dict[str, float]:
    fuel: Branch = ctx["branches"]["fuel"]
    lox:  Branch = ctx["branches"]["lox"]
    tca:  TCA    = ctx["tca"]
    if tca is None:
        raise ValueError("No nozzle/TCA in ctx (add a 'nozzle:' block in YAML).")

    Nf = ctx["steady"]["N_fuel_rpm"]
    No = ctx["steady"]["N_lox_rpm"]

    def F(x):
        Pc, mf, mo = x
        r1 = fuel.residual(mf, Nf, Pc)
        r2 = lox .residual(mo,  No, Pc)
        Pc_hat, _, _, _ = tca.backsolve_pc_and_thrust(mf, mo)
        return [r1, r2, Pc_hat - Pc]

    Pc0 = float(ctx["chamber"].Pc)
    mf0, mo0 = 1.0, 2.0
    Pc, mf, mo = fsolve(F, x0=[Pc0, mf0, mo0])

    Pc_final, MR, p_exit, thrust_N = tca.backsolve_pc_and_thrust(mf, mo)
    return {"Pc": Pc_final, "mdot_f": mf, "mdot_ox": mo, "MR": MR, "p_exit": p_exit, "thrust_N": thrust_N}


# ----------------- super simple var access -----------------
# Supported variable names:
#   fuel_tank_p, lox_tank_p
#   throat_diameter_m, area_ratio
#   N_fuel_rpm, N_lox_rpm
#   fuel_d1, fuel_dn, fuel_d2, lox_d1, lox_dn, lox_d2
def _get_value(ctx: Dict[str, Any], name: str) -> float:
    if   name == "fuel_tank_p":         return ctx["tanks"]["fuel"].p
    elif name == "lox_tank_p":          return ctx["tanks"]["lox"].p
    elif name == "throat_diameter_m":   return ctx["tca"].spec.throat_diameter_m
    elif name == "area_ratio":          return ctx["tca"].spec.area_ratio
    elif name == "N_fuel_rpm":          return ctx["steady"]["N_fuel_rpm"]
    elif name == "N_lox_rpm":           return ctx["steady"]["N_lox_rpm"]
    elif name == "fuel_d1":             return ctx["branches"]["fuel"].pump.d1
    elif name == "fuel_dn":             return ctx["branches"]["fuel"].pump.dn
    elif name == "fuel_d2":             return ctx["branches"]["fuel"].pump.d2
    elif name == "lox_d1":              return ctx["branches"]["lox"].pump.d1
    elif name == "lox_dn":              return ctx["branches"]["lox"].pump.dn
    elif name == "lox_d2":              return ctx["branches"]["lox"].pump.d2
    else:
        raise KeyError(f"Unknown variable '{name}'")

def _set_value(ctx: Dict[str, Any], name: str, val: float) -> None:
    if   name == "fuel_tank_p":         ctx["tanks"]["fuel"].p = float(val)
    elif name == "lox_tank_p":          ctx["tanks"]["lox"].p  = float(val)
    elif name == "throat_diameter_m":   setattr(ctx["tca"].spec, "throat_diameter_m", float(val))
    elif name == "area_ratio":          setattr(ctx["tca"].spec, "area_ratio", float(val))
    elif name == "N_fuel_rpm":          ctx["steady"]["N_fuel_rpm"] = float(val)
    elif name == "N_lox_rpm":           ctx["steady"]["N_lox_rpm"]  = float(val)
    elif name == "fuel_d1":             setattr(ctx["branches"]["fuel"].pump, "d1", float(val))
    elif name == "fuel_dn":             setattr(ctx["branches"]["fuel"].pump, "dn", float(val))
    elif name == "fuel_d2":             setattr(ctx["branches"]["fuel"].pump, "d2", float(val))
    elif name == "lox_d1":              setattr(ctx["branches"]["lox"].pump, "d1", float(val))
    elif name == "lox_dn":              setattr(ctx["branches"]["lox"].pump, "dn", float(val))
    elif name == "lox_d2":              setattr(ctx["branches"]["lox"].pump, "d2", float(val))
    else:
        raise KeyError(f"Unknown variable '{name}'")

def _pack_x(ctx: Dict[str, Any], names: List[str]) -> np.ndarray:
    return np.array([_get_value(ctx, nm) for nm in names], dtype=float)

def _apply_x(ctx: Dict[str, Any], names: List[str], x: np.ndarray) -> None:
    for nm, v in zip(names, x):
        _set_value(ctx, nm, float(v))


# ----------------- objective & constraints (with printing) -----------------
def _objective_total_power(ctx0: Dict[str, Any], names: List[str], verbose: bool):
    eval_counter = {"n": 0}

    def f(x: np.ndarray) -> float:
        eval_counter["n"] += 1
        ctx = copy.deepcopy(ctx0)
        _apply_x(ctx, names, x)
        try:
            sol = _steady_coupled_solve(ctx)
        except Exception:
            if verbose and eval_counter["n"] % 10 == 0:
                print(f"[obj] eval {eval_counter['n']:4d}: NON-CONVERGED → penalty")
            return 1e20

        fuel = ctx["branches"]["fuel"]; lox = ctx["branches"]["lox"]
        Nf = ctx["steady"]["N_fuel_rpm"]; No = ctx["steady"]["N_lox_rpm"]
        Qf = sol["mdot_f"] / fuel.rho
        Qo = sol["mdot_ox"] / lox.rho
        Pf = fuel.pump.power_W(Qf, Nf, fuel.rho, fuel.mu)
        Po = lox.pump.power_W(Qo, No, lox.rho,  lox.mu)
        Ptot = Pf + Po

        if verbose and eval_counter["n"] % 10 == 0:
            print(f"[obj] eval {eval_counter['n']:4d}: Ptot={Ptot/1000:9.3f} kW, Thrust={sol['thrust_N']:9.1f} N, Pc={pa2psi(sol['Pc']):8.2f} psi")
        return Ptot
    return f

def _thrust_constraint(ctx0: Dict[str, Any], names: List[str], target_N: float, mode: str = "ge", tol_frac: float = 0.005, verbose: bool = True):
    def thrust_of_x(x):
        ctx = copy.deepcopy(ctx0)
        _apply_x(ctx, names, x)
        return _steady_coupled_solve(ctx)["thrust_N"]

    def g_ge(x):   # >= 0 OK
        try:
            return thrust_of_x(x) - target_N
        except Exception:
            return -1.0

    cons = []
    if mode == "ge":
        cons.append({"type": "ineq", "fun": g_ge})
    elif mode == "le":
        cons.append({"type": "ineq", "fun": (lambda x: -g_ge(x))})
    else:
        # "eq": two-sided band
        def g_low(x):   # thrust >= (1 - tol)*target
            return g_ge(x) + target_N * tol_frac
        def g_high(x):  # thrust <= (1 + tol)*target
            return (target_N * (1.0 + tol_frac)) - (target_N + g_ge(x))
        cons.append({"type": "ineq", "fun": g_low})
        cons.append({"type": "ineq", "fun": g_high})

    # Optional iteration callback: print every iteration’s status
    def callback(xk):
        if not verbose:
            return
        try:
            ctx = copy.deepcopy(ctx0)
            _apply_x(ctx, names, xk)
            sol = _steady_coupled_solve(ctx)
            fuel = ctx["branches"]["fuel"]; lox = ctx["branches"]["lox"]
            Nf = ctx["steady"]["N_fuel_rpm"]; No = ctx["steady"]["N_lox_rpm"]
            Qf = sol["mdot_f"] / fuel.rho
            Qo = sol["mdot_ox"] / lox.rho
            Pf = fuel.pump.power_W(Qf, Nf, fuel.rho, fuel.mu)
            Po = lox.pump.power_W(Qo, No, lox.rho,  lox.mu)
            Ptot = Pf + Po
            print(f"[iter] Ptot={Ptot/1000:9.3f} kW | Thrust={sol['thrust_N']:9.1f} N | Pc={pa2psi(sol['Pc']):8.2f} psi")
        except Exception:
            print("[iter] non-converged point")

    return cons, callback


# ----------------- public API -----------------
def run_power_optimizer(ctx: Dict[str, Any], opt_cfg: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """
    YAML (under `optimize:`):
      enabled: true
      thrust_target_N: 22000
      thrust_mode: "ge"                  # "ge", "le", "eq"
      variables:                         # list of names to vary (order matters)
        - fuel_tank_p
        - lox_tank_p
        - throat_diameter_m
        - area_ratio
        - N_fuel_rpm
        - N_lox_rpm
        - fuel_d1
        - fuel_dn
        - fuel_d2
        - lox_d1
        - lox_dn
        - lox_d2
      bounds:                            # dict: name -> [lb, ub]
        fuel_tank_p: [8.0e5, 2.5e6]
        lox_tank_p:  [8.0e5, 2.5e6]
        throat_diameter_m: [0.040, 0.120]
        area_ratio: [3.0, 12.0]
        N_fuel_rpm: [8000, 35000]
        N_lox_rpm:  [8000, 35000]
        fuel_d1: [0.03, 0.07]
        fuel_dn: [0.01, 0.05]
        fuel_d2: [0.06, 0.12]
        lox_d1:  [0.03, 0.07]
        lox_dn:  [0.01, 0.05]
        lox_d2:  [0.06, 0.12]
    """
    names: List[str] = list(opt_cfg.get("variables", []))
    if not names:
        raise ValueError("optimize.variables must be a non-empty list of variable names.")

    bdict: Dict[str, List[float]] = opt_cfg.get("bounds", {})
    bounds: List[Tuple[float, float]] = []
    for nm in names:
        if nm not in bdict:
            raise KeyError(f"Missing bounds for variable '{nm}'")
        lo, hi = bdict[nm]
        bounds.append((float(lo), float(hi)))

    # pre-run banner
    if verbose:
        print("\n[Optimization] Variables & bounds")
        for nm, (lo, hi) in zip(names, bounds):
            print(f"  - {nm:18s}: [{lo:.6g}, {hi:.6g}]")
        print(f"  Target thrust: {_eval_if_expr(opt_cfg['thrust_target_N']):.2f} N   mode={opt_cfg.get('thrust_mode','ge')}")

    x0 = _pack_x(ctx, names)

    obj = _objective_total_power(ctx, names, verbose=verbose)
    cons, callback = _thrust_constraint(
        ctx, names,
        target_N=_eval_if_expr(opt_cfg["thrust_target_N"]),
        mode=str(opt_cfg.get("thrust_mode", "ge")),
        tol_frac=0.005,
        verbose=verbose
    )

    res = minimize(
        obj, x0, method="SLSQP", bounds=bounds, constraints=cons,
        options={"maxiter": 250, "ftol": 1e-9, "disp": verbose},
        callback=callback
    )

    # Build final design and summarize
    ctx_star = copy.deepcopy(ctx)
    _apply_x(ctx_star, names, res.x)
    sol = _steady_coupled_solve(ctx_star)

    fuel = ctx_star["branches"]["fuel"]; lox = ctx_star["branches"]["lox"]
    Nf = ctx_star["steady"]["N_fuel_rpm"]; No = ctx_star["steady"]["N_lox_rpm"]
    Qf = sol["mdot_f"] / fuel.rho
    Qo = sol["mdot_ox"] / lox.rho
    Pf = fuel.pump.power_W(Qf, Nf, fuel.rho, fuel.mu)
    Po = lox.pump.power_W(Qo, No, lox.rho,  lox.mu)
    Ptot = Pf + Po

    if verbose:
        print("\n[Optimization Result]")
        print("  Status :", "SUCCESS" if res.success else "FAILED")
        print("  Message:", res.message)
        print(f"  Power  : Fuel={Pf/1000:.2f} kW  LOX={Po/1000:.2f} kW  Total={Ptot/1000:.2f} kW")
        print(f"  Pc     : {pa2psi(sol['Pc']):.2f} psi")
        print(f"  mdots  : fuel={sol['mdot_f']:.5f} kg/s  lox={sol['mdot_ox']:.5f} kg/s  MR={sol['MR']:.3f}")
        print(f"  Thrust : {sol['thrust_N']:.2f} N ({sol['thrust_N']/4.448:.2f} lbf)")
        print("  Design variables:")
        for nm, val in zip(names, res.x):
            print(f"    - {nm:18s} = {val:.8g}")

    return {
        "success": bool(res.success),
        "message": res.message,
        "design_vars": {nm: float(val) for nm, val in zip(names, res.x)},
        "power_W": {"fuel": Pf, "lox": Po, "total": Ptot},
        "solution": {
            "Pc_Pa": sol["Pc"], "Pc_psi": pa2psi(sol["Pc"]),
            "mdot_f": sol["mdot_f"], "mdot_ox": sol["mdot_ox"], "MR": sol["MR"],
            "thrust_N": sol["thrust_N"]
        },
        "raw_result": res
    }
