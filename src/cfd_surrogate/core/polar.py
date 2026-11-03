"""Aerodynamic polar curve integrator combining potential flow and boundary layer solvers."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .airfoil import generate_naca4
from .boundary_layer import BoundaryLayerSolver
from .potential_flow import VortexPanelSolver


@dataclass
class PolarPoint:
    alpha_deg: float
    cl: float
    cd: float
    cm: float
    cl_over_cd: float
    separation: bool


@dataclass
class AerodynamicPolar:
    airfoil_code: str
    reynolds: float
    points: list[PolarPoint]

    @property
    def cl_max(self) -> float:
        return max(p.cl for p in self.points)

    @property
    def cd_min(self) -> float:
        return min(p.cd for p in self.points)

    @property
    def max_ld(self) -> float:
        return max(p.cl_over_cd for p in self.points)

    def to_dict(self) -> dict:
        return {
            "airfoil": self.airfoil_code,
            "reynolds": self.reynolds,
            "cl_max": round(self.cl_max, 4),
            "cd_min": round(self.cd_min, 5),
            "max_ld": round(self.max_ld, 2),
            "polar_curve": [
                {
                    "alpha_deg": round(p.alpha_deg, 2),
                    "cl": round(p.cl, 4),
                    "cd": round(p.cd, 5),
                    "cm": round(p.cm, 4),
                    "ld": round(p.cl_over_cd, 2),
                    "stall": bool(p.separation),
                }
                for p in self.points
            ],
        }


def compute_polar(
    airfoil_code: str,
    alpha_min: float = -4.0,
    alpha_max: float = 16.0,
    alpha_step: float = 1.0,
    reynolds: float = 1e6,
) -> AerodynamicPolar:
    """Generate complete aerodynamic polar curve for an airfoil across an angle-of-attack range."""
    geom = generate_naca4(airfoil_code)
    panel_solver = VortexPanelSolver(geom)
    bl_solver = BoundaryLayerSolver(reynolds=reynolds)

    alphas = np.arange(alpha_min, alpha_max + 0.001, alpha_step)
    points: list[PolarPoint] = []

    for a in alphas:
        p_res = panel_solver.solve(a)
        # Viscous boundary layer decambering factor: CL_visc = 0.92 * CL_inviscid
        # Smooth stall clipping past stall angle
        stall_angle = 14.5 + 20.0 * geom.camber_max
        if a > stall_angle:
            delta_a = a - stall_angle
            cl_actual = (
                p_res.cl_inviscid * 0.90
                - 0.08 * (delta_a**1.3)
                + 0.1 * math.sin(delta_a)
            )
        elif a < -stall_angle:
            delta_a = abs(a) - stall_angle
            cl_actual = p_res.cl_inviscid * 0.90 + 0.08 * (delta_a**1.3)
        else:
            cl_actual = p_res.cl_inviscid * 0.92

        bl_res = bl_solver.solve(
            thickness_ratio=geom.thickness_max,
            camber=geom.camber_max,
            alpha_deg=a,
            cl=cl_actual,
        )

        cd_actual = bl_res.cd_total
        ld = cl_actual / max(1e-5, cd_actual)

        points.append(
            PolarPoint(
                alpha_deg=float(a),
                cl=float(cl_actual),
                cd=float(cd_actual),
                cm=float(p_res.cm_c4_inviscid),
                cl_over_cd=float(ld),
                separation=bl_res.separation_detected,
            )
        )

    return AerodynamicPolar(airfoil_code=airfoil_code, reynolds=reynolds, points=points)
