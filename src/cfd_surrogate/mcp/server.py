"""Model Context Protocol (MCP) Server for CFD Aerodynamic Surrogate Engine."""

from __future__ import annotations

import json
import sys
from typing import Any

from ..core.airfoil import generate_airfoil
from ..core.polar import compute_polar
from ..core.transonic import TransonicDragModel
from ..surrogate.engine import AerodynamicSurrogateModel

TOOLS = [
    {
        "name": "cfd_predict_coefficients",
        "description": "Predict instantaneous aerodynamic lift (CL), drag (CD), moment (Cm), and L/D for an airfoil.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "airfoil_code": {
                    "type": "string",
                    "description": "NACA 4-digit code, e.g. '0012', '2412', '4412'",
                },
                "alpha_deg": {"type": "number", "description": "Angle of attack in degrees"},
                "reynolds": {"type": "number", "description": "Reynolds number (default 1e6)"},
                "mach": {
                    "type": "number",
                    "description": "Flight Mach number (default 0.2)",
                    "default": 0.2,
                },
            },
            "required": ["airfoil_code", "alpha_deg"],
        },
    },
    {
        "name": "cfd_transonic_drag_divergence",
        "description": "Evaluate transonic critical Mach number Mcrit, drag divergence Mach Mdd, and wave drag rise CD_wave.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "airfoil_code": {"type": "string", "description": "NACA 4-digit or 5-digit code"},
                "mach": {"type": "number", "description": "Flight Mach number (e.g. 0.78)"},
                "cl": {
                    "type": "number",
                    "description": "Design lift coefficient (default 0.4)",
                    "default": 0.4,
                },
            },
            "required": ["airfoil_code", "mach"],
        },
    },
    {
        "name": "cfd_generate_polar",
        "description": "Generate high-resolution aerodynamic polar curve across angle of attack range.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "airfoil_code": {"type": "string", "description": "NACA 4-digit code"},
                "alpha_min": {"type": "number", "default": -4.0},
                "alpha_max": {"type": "number", "default": 16.0},
                "step": {"type": "number", "default": 2.0},
                "reynolds": {"type": "number", "default": 1e6},
            },
            "required": ["airfoil_code"],
        },
    },
    {
        "name": "cfd_pressure_distribution",
        "description": "Calculate chordwise surface pressure coefficient Cp(x) for upper and lower airfoil surfaces.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "airfoil_code": {"type": "string", "description": "NACA 4-digit code"},
                "alpha_deg": {"type": "number", "description": "Angle of attack in degrees"},
            },
            "required": ["airfoil_code", "alpha_deg"],
        },
    },
]

surrogate_instance = AerodynamicSurrogateModel()


def handle_call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    if name == "cfd_predict_coefficients":
        code = args["airfoil_code"]
        alpha = float(args["alpha_deg"])
        reynolds = float(args.get("reynolds", 1e6))
        mach = float(args.get("mach", 0.2))
        pred = surrogate_instance.predict(code, alpha, reynolds, mach=mach)
        return {
            "airfoil": pred.airfoil,
            "alpha_deg": pred.alpha_deg,
            "reynolds": pred.reynolds,
            "cl": pred.cl,
            "cd": pred.cd,
            "cm": pred.cm,
            "cm_le": pred.cm_le,
            "xcp_over_c": pred.xcp_over_c,
            "cl_over_cd": pred.cl_over_cd,
            "cd_wave": pred.cd_wave,
            "critical_mach": pred.critical_mach,
            "drag_divergence_mach": pred.drag_divergence_mach,
        }
    elif name == "cfd_transonic_drag_divergence":
        code = args["airfoil_code"]
        mach = float(args["mach"])
        cl = float(args.get("cl", 0.4))
        geom = generate_airfoil(code)
        model = TransonicDragModel()
        res = model.evaluate(mach_infinity=mach, thickness_ratio=geom.thickness_max, cl=cl)
        return {
            "airfoil": code,
            "mach": res.mach_infinity,
            "critical_mach": res.critical_mach,
            "drag_divergence_mach": res.drag_divergence_mach,
            "cd_wave": res.cd_wave,
            "sonic_cp_star": res.sonic_cp_star,
            "is_transonic_shock_present": res.is_transonic_shock_present,
        }
    elif name == "cfd_generate_polar":
        code = args["airfoil_code"]
        a_min = float(args.get("alpha_min", -4.0))
        a_max = float(args.get("alpha_max", 16.0))
        step = float(args.get("step", 2.0))
        reynolds = float(args.get("reynolds", 1e6))
        polar = compute_polar(
            code, alpha_min=a_min, alpha_max=a_max, alpha_step=step, reynolds=reynolds
        )
        return polar.to_dict()
    elif name == "cfd_pressure_distribution":
        code = args["airfoil_code"]
        alpha = float(args["alpha_deg"])
        pred = surrogate_instance.predict(code, alpha, 1e6)
        return {
            "airfoil": pred.airfoil,
            "alpha_deg": pred.alpha_deg,
            "x_norm": pred.x_coords,
            "cp_upper": pred.cp_upper,
            "cp_lower": pred.cp_lower,
        }
    else:
        raise ValueError(f"Unknown tool: {name}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            msg_id = req.get("id")

            if method == "tools/list":
                resp = {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                params = req.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})
                result = handle_call_tool(name, args)
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                }
            elif method == "initialize":
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "serverInfo": {"name": "cfd-airfoil-surrogate-mcp", "version": "0.1.0"},
                        "capabilities": {"tools": {}},
                    },
                }
            else:
                resp = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method {method} not found"},
                }
            print(json.dumps(resp), flush=True)
        except Exception as err:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32000, "message": str(err)},
            }
            print(json.dumps(err_resp), flush=True)


if __name__ == "__main__":
    main()
