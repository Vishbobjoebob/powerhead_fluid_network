# yaml_io.py
from __future__ import annotations

import ast
from typing import Any, Dict, List
from pathlib import Path

import yaml
import csv

from components import Line, Bend, Injector, GulichPump, NozzleSpec, TCA, TCALinerLoss
from network import Tank, Chamber, Branch


def _eval_if_expr(x) -> float:
    """
    Allow simple expressions in YAML (e.g. '2*0.0254').

    If x is:
      * number -> return as float
      * string -> try to parse/eval as a Python expression in a safe namespace
    """
    if isinstance(x, (int, float)):
        return float(x)

    if isinstance(x, str):
        try:
            node = ast.parse(x, mode="eval")
            return float(
                eval(
                    compile(node, "<yaml>", "eval"),
                    {"__builtins__": {}},
                    {},
                )
            )
        except Exception:
            # Fall back to plain float conversion
            return float(x)

    return float(x)


def _build_lines(lst: List[Dict[str, Any]]) -> List[Line]:
    """Build a list of Line objects from YAML list entries."""
    return [
        Line(L=_eval_if_expr(d["L"]), D=float(d["D"]), eps=float(d["eps"]))
        for d in (lst or [])
    ]


def _build_bends(lst: List[Dict[str, Any]]) -> List[Bend]:
    """Build a list of Bend objects from YAML list entries."""
    return [
        Bend(K=float(d["K"]), D=float(d["D"]))
        for d in (lst or [])
    ]


def _pump_from_yaml(p: Dict[str, Any]) -> GulichPump:
    """
    Construct a GulichPump from YAML.

    Supports:
      head_model: "meanline"  -> use Gülich meanline correlations
      head_model: "curve"     -> use a multi-RPM Q–H map from CSV or inline

    For curve mode with CSV:

      curve:
        file: "fuel_pump_curve.csv"

    CSV must have at least:
      N_rpm, Q_m3h and/or Q_m3s, H_m

    Example row:
      20000, 10.0, 450.0
    """
    geom = p.get("geom", {})
    eff = p.get("efficiency", {})
    leak = p.get("leakage", {})

    head_model = (p.get("head_model") or "meanline").lower()

    # curve_map: rpm -> {"Q": [Q_i], "H": [H_i]}
    curve_map: Dict[float, Dict[str, list[float]]] = {}

    if head_model == "curve":
        curve_cfg = p.get("curve", {}) or {}

        # --------- Option A: CSV file (recommended) -----------------------
        curve_file = curve_cfg.get("file")
        if curve_file:
            path = Path(curve_file)
            with path.open(newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Pump speed [rpm]
                    N_rpm = float(row["N_rpm"])

                    # Flow: Q_m3h or Q_m3s or Q
                    if "Q_m3h" in row and row["Q_m3h"]:
                        Q_m3s = float(row["Q_m3h"]) / 3600.0
                    elif "Q_m3s" in row and row["Q_m3s"]:
                        Q_m3s = float(row["Q_m3s"])
                    else:
                        Q_m3s = float(row.get("Q", 0.0))

                    # Head: H_m or H
                    if "H_m" in row and row["H_m"]:
                        H_m = float(row["H_m"])
                    else:
                        H_m = float(row.get("H", 0.0))

                    if N_rpm not in curve_map:
                        curve_map[N_rpm] = {"Q": [], "H": []}
                    curve_map[N_rpm]["Q"].append(Q_m3s)
                    curve_map[N_rpm]["H"].append(H_m)

        # --------- Option B: inline arrays (single RPM) -------------------
        else:
            # Single-speed curve, still stored in the same map structure.
            N_rpm = float(curve_cfg.get("N_rpm", 1.0))
            Q_list = [float(q) for q in curve_cfg.get("Q_m3s", [])]
            H_list = [float(h) for h in curve_cfg.get("H_m", [])]

            curve_map[N_rpm] = {"Q": Q_list, "H": H_list}

        # Sort Q–H lists for each RPM by Q
        for N_rpm, data in curve_map.items():
            if not data["Q"]:
                continue
            pairs = sorted(zip(data["Q"], data["H"]), key=lambda p: p[0])
            qs, hs = zip(*pairs)
            data["Q"] = list(qs)
            data["H"] = list(hs)

    return GulichPump(
        d1=float(geom["d1"]),
        dn=float(geom["dn"]),
        d2=float(geom["d2"]),
        b2=float(geom["b2"]),
        zLa=int(geom["zLa"]),
        beta2B_deg=float(geom["beta2B_deg"]),
        n_q=float(eff["n_q"]),
        alpha1_deg=float(geom.get("alpha1_deg", 90.0)),
        e1=float(geom.get("e1", 0.0)),
        e=float(geom.get("e", 0.0)),
        lambdaLa_deg=float(geom.get("lambdaLa_deg", 90.0)),
        QS3=float(leak.get("QS3", 0.0)),
        QSp=float(leak.get("QSp", 0.0)),
        Q_ref=float(eff.get("Q_ref", 1.0)),
        head_model=head_model,
        curve_map=curve_map,
    )



def build_from_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Build the full model context from a YAML configuration.

    The returned dict has the structure:

      {
        "fluids": { ... },
        "tanks":  { "fuel": Tank, "lox": Tank },
        "chamber": Chamber,
        "branches": { "fuel": Branch, "lox": Branch },
        "steady": { ... },
        "tca": TCA | None,
      }
    """
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    # ----------------------------------------------------------------------
    # Fluids
    # ----------------------------------------------------------------------
    fluids_cfg = cfg.get("fluids", {})
    rho_f = float(fluids_cfg["fuel"]["rho"])
    mu_f = float(fluids_cfg["fuel"]["mu"])
    rho_o = float(fluids_cfg["lox"]["rho"])
    mu_o = float(fluids_cfg["lox"]["mu"])

    # ----------------------------------------------------------------------
    # Tanks & chamber
    # ----------------------------------------------------------------------
    bt = cfg["boundaries"]["tanks"]
    tank_f = Tank(p=float(bt["fuel_tank"]["p"]))
    tank_o = Tank(p=float(bt["lox_tank"]["p"]))

    chamber = Chamber(Pc=float(cfg["boundaries"]["chamber"]["Pc"]))

    # ----------------------------------------------------------------------
    # Geometry + pumps
    # ----------------------------------------------------------------------
    gf = cfg["geometry"]["fuel"]
    go = cfg["geometry"]["lox"]
    pumps_cfg = cfg["pumps"]

    pump_f = _pump_from_yaml(pumps_cfg["fuel"])
    pump_o = _pump_from_yaml(pumps_cfg["lox"])

    # Fuel branch
    fuel_branch = Branch(
        rho=rho_f,
        mu=mu_f,
        tank=tank_f,
        suction_lines=_build_lines(gf.get("suction_lines", [])),
        suction_bends=_build_bends(gf.get("suction_bends", [])),
        pump=pump_f,
        discharge_lines=_build_lines(gf.get("discharge_lines", [])),
        discharge_bends=_build_bends(gf.get("discharge_bends", [])),
        injector=Injector(CdA=float(gf["injector"]["CdA"])),
    )

    # LOX branch
    lox_branch = Branch(
        rho=rho_o,
        mu=mu_o,
        tank=tank_o,
        suction_lines=_build_lines(go.get("suction_lines", [])),
        suction_bends=_build_bends(go.get("suction_bends", [])),
        pump=pump_o,
        discharge_lines=_build_lines(go.get("discharge_lines", [])),
        discharge_bends=_build_bends(go.get("discharge_bends", [])),
        injector=Injector(CdA=float(go["injector"]["CdA"])),
    )

    # ----------------------------------------------------------------------
    # Steady settings
    # ----------------------------------------------------------------------
    steady_cfg = cfg.get("steady", {})
    steady = {
        "N_fuel_rpm": float(steady_cfg.get("N_fuel_rpm", 20000.0)),
        "N_lox_rpm": float(steady_cfg.get("N_lox_rpm", 20000.0)),
        "use_nozzle_solve": bool(steady_cfg.get("use_nozzle_solve", True)),
    }

    # ----------------------------------------------------------------------
    # Nozzle / TCA (optional)
    # ----------------------------------------------------------------------
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

        # Optional TCA liner loss block
        liner_loss_cfg = noz_cfg.get("liner_loss")
        liner_loss = None
        if liner_loss_cfg is not None:
            liner_loss = TCALinerLoss(
                K=float(liner_loss_cfg.get("K", 0.0)),
                D_hyd_m=_eval_if_expr(
                    liner_loss_cfg.get("D_hyd_m", noz_cfg["throat_diameter_m"])
                ),
                active=bool(liner_loss_cfg.get("active", True)),
            )

        tca = TCA(spec, liner_loss=liner_loss)

    # ----------------------------------------------------------------------
    # Return context
    # ----------------------------------------------------------------------
    return {
        "fluids": {
            "fuel": {"rho": rho_f, "mu": mu_f},
            "lox": {"rho": rho_o, "mu": mu_o},
        },
        "tanks": {"fuel": tank_f, "lox": tank_o},
        "chamber": chamber,
        "branches": {"fuel": fuel_branch, "lox": lox_branch},
        "steady": steady,
        "tca": tca,
    }
