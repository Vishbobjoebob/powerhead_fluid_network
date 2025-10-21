# yaml_io.py
from __future__ import annotations
import ast
from typing import Any, Dict, List
from pathlib import Path
import yaml

from components import psi2pa, Line, Bend, Injector, GulichPump
from network import Tank, Chamber, Branch
from nozzle import NozzleSpec, TCA

def _eval_if_expr(x):
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        try:
            node = ast.parse(x, mode="eval")
            return float(eval(compile(node, "<yaml>", "eval"), {"__builtins__": {}}, {}))
        except Exception:
            return float(x)
    return float(x)

def _build_lines(lst: List[Dict[str, Any]]) -> List[Line]:
    return [Line(L=_eval_if_expr(d["L"]), D=float(d["D"]), eps=float(d["eps"])) for d in (lst or [])]

def _build_bends(lst: List[Dict[str, Any]]) -> List[Bend]:
    return [Bend(K=float(d["K"]), D=float(d["D"])) for d in (lst or [])]

def _pump_from_yaml(p: Dict[str, Any]) -> GulichPump:
    geom = p.get("geom", {})
    eff  = p.get("efficiency", {})
    leak = p.get("leakage", {})
    # GulichPump ctor:
    return GulichPump(
        d1=float(geom["d1"]), dn=float(geom["dn"]), d2=float(geom["d2"]),
        b2=float(geom["b2"]), zLa=int(geom["zLa"]), beta2B_deg=float(geom["beta2B_deg"]), n_q=float(eff["n_q"]),
        alpha1_deg=float(geom.get("alpha1_deg", 90.0)),
        e1=float(geom.get("e1", 0.0)),
        e=float(geom.get("e", 0.0)),
        lambdaLa_deg=float(geom.get("lambdaLa_deg", 90.0)),
        QS3=float(leak.get("QS3", 0.0)),
        QSp=float(leak.get("QSp", 0.0)),
        Q_ref=float(eff.get("Q_ref", 1.0)),
    )

def build_from_yaml(path: str | Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    # Fluids
    fluids = cfg.get("fluids", {})
    rho_f = float(fluids["fuel"]["rho"]); mu_f = float(fluids["fuel"]["mu"])
    rho_o = float(fluids["lox"]["rho"]);  mu_o = float(fluids["lox"]["mu"])

    # Tanks & Chamber
    bt = cfg["boundaries"]["tanks"]
    tank_f = Tank(p=float(bt["fuel_tank"]["p"]))
    tank_o = Tank(p=float(bt["lox_tank"]["p"]))
    chamber = Chamber(Pc=float(cfg["boundaries"]["chamber"]["Pc"]))

    # Geometry + pumps
    gf = cfg["geometry"]["fuel"]
    go = cfg["geometry"]["lox"]
    pumps = cfg["pumps"]
    pump_f = _pump_from_yaml(pumps["fuel"])
    pump_o = _pump_from_yaml(pumps["lox"])

    fuel = Branch(
        rho=rho_f, mu=mu_f, tank=tank_f,
        suction_lines=_build_lines(gf.get("suction_lines", [])),
        suction_bends=_build_bends(gf.get("suction_bends", [])),
        pump=pump_f,
        discharge_lines=_build_lines(gf.get("discharge_lines", [])),
        discharge_bends=_build_bends(gf.get("discharge_bends", [])),
        injector=Injector(CdA=float(gf["injector"]["CdA"]))
    )
    lox = Branch(
        rho=rho_o, mu=mu_o, tank=tank_o,
        suction_lines=_build_lines(go.get("suction_lines", [])),
        suction_bends=_build_bends(go.get("suction_bends", [])),
        pump=pump_o,
        discharge_lines=_build_lines(go.get("discharge_lines", [])),
        discharge_bends=_build_bends(go.get("discharge_bends", [])),
        injector=Injector(CdA=float(go["injector"]["CdA"]))
    )

    # Steady & Nozzle/TCA
    steady_cfg = cfg.get("steady", {})
    steady = {
        "N_fuel_rpm": float(steady_cfg.get("N_fuel_rpm", 20000.0)),
        "N_lox_rpm":  float(steady_cfg.get("N_lox_rpm",  20000.0)),
        "use_nozzle_solve": bool(steady_cfg.get("use_nozzle_solve", True)),
    }

    tca = None
    noz_cfg = cfg.get("nozzle", None)
    if noz_cfg:
        spec = NozzleSpec(
            oxName=noz_cfg["oxName"],
            fuelName=noz_cfg["fuelName"],
            throat_diameter_m=_eval_if_expr(noz_cfg["throat_diameter_m"]),
            area_ratio=float(noz_cfg["area_ratio"]),
            Cd_throat=float(noz_cfg["Cd_throat"]),
            c_star_eff=float(noz_cfg["c_star_eff"]),
            p_amb_Pa=_eval_if_expr(noz_cfg["p_amb"]),
        )
        tca = TCA(spec)

    # # Transient
    # transient_cfg = cfg.get("transient", None)
    # transient = None
    # if transient_cfg:
    #     ramps = transient_cfg.get("ramps")
    #     Nf = ramps.get("N_fuel_rpm")
    #     No = ramps.get("N_lox_rpm")
    #     Pc_r = ramps.get("Pc", {"start": 0.0, "end": 400.0, "t0": 0.0, "t1": 0.10})

    #     def _ramp(a0, a1, t0, t1, t):
    #         if t <= t0: return a0
    #         if t >= t1: return a1
    #         return a0 + (a1 - a0) * (t - t0) / (t1 - t0)

    #     Nf0, Nf1, t0, t1 = float(Nf["start"]), float(Nf["end"]), float(Nf["t0"]), float(Nf["t1"])
    #     No0, No1, u0, u1 = float(No["start"]), float(No["end"]), float(No["t0"]), float(No["t1"])
    #     Pc0 = Pc_r
    #     Pc1 = _maybe_field_units(Pc_r, "end")
    #     v0, v1 = float(Pc_r["t0"]), float(Pc_r["t1"])

    #     transient = {
    #         "ramps": {
    #             "Nf_of_t": lambda t: _ramp(Nf0, Nf1, t0, t1, t),
    #             "Nox_of_t": lambda t: _ramp(No0, No1, u0, u1, t),
    #             "Pc_of_t":  lambda t: _ramp(Pc0, Pc1, v0, v1, t),
    #         },
    #         "ivp": transient_cfg.get("ivp", {}),
    #         "print_samples": int(transient_cfg.get("print_samples", 12)),
    #         "plots": bool(transient_cfg.get("plots", True)),
    #     }

    return {
        "fluids": {"fuel": {"rho": rho_f, "mu": mu_f}, "lox": {"rho": rho_o, "mu": mu_o}},
        "tanks": {"fuel": tank_f, "lox": tank_o},
        "chamber": chamber,
        "branches": {"fuel": fuel, "lox": lox},
        "steady": steady,
        "tca": tca,
        # "transient": transient,
    }
