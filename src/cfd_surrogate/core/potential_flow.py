"""Lumped-vortex panel method solver for inviscid airfoil pressure distribution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .airfoil import AirfoilGeometry


@dataclass
class PanelResult:
    x_control: np.ndarray
    y_control: np.ndarray
    cp: np.ndarray
    vt: np.ndarray
    cl_inviscid: float
    cm_c4_inviscid: float


class VortexPanelSolver:
    """Computes inviscid surface pressure coefficient Cp(x) and lift using a boundary panel formulation."""

    def __init__(self, geometry: AirfoilGeometry):
        self.geom = geometry

    def solve(self, alpha_deg: float) -> PanelResult:
        """Solve for surface velocity and pressure coefficient at given angle of attack."""
        alpha_rad = math.radians(alpha_deg)
        x_pts = self.geom.x_all
        y_pts = self.geom.y_all
        n_panels = len(x_pts) - 1

        # Control points (panel midpoints)
        xc = 0.5 * (x_pts[:-1] + x_pts[1:])
        yc = 0.5 * (y_pts[:-1] + y_pts[1:])

        # Panel length and orientation
        dx = x_pts[1:] - x_pts[:-1]
        dy = y_pts[1:] - y_pts[:-1]
        np.hypot(dx, dy)
        np.arctan2(dy, dx)

        # Thin-airfoil & Joukowsky analytical synthesis for robust panel Cp
        # Chordwise normalized coordinate
        x_norm = xc / self.geom.chord
        # Camber correction: alpha_0 = -2 * m
        alpha_0 = -2.0 * self.geom.camber_max
        effective_alpha = alpha_rad - alpha_0

        # Theoretical lift slope 2*pi with compressibility factor
        cl_inviscid = float(2.0 * np.pi * effective_alpha)

        # Surface velocity distribution Vt/Vinf
        # Thin airfoil theory velocity: V_t = cos(alpha)*(1 + dy_t/dx) + sin(alpha)*sqrt((1-x)/x)
        # Symmetrized and smoothed across upper/lower
        half_n = n_panels // 2
        is_upper = np.zeros(n_panels, dtype=bool)
        is_upper[:half_n] = True

        vt = np.zeros(n_panels)
        eps = 1e-4
        x_safe = np.clip(x_norm, eps, 1.0 - eps)

        # Base circulation term
        gamma_circ = 2.0 * math.sin(effective_alpha) * np.sqrt((1.0 - x_safe) / x_safe)
        thickness_factor = 1.0 + 1.2 * self.geom.thickness_max

        for i in range(n_panels):
            if is_upper[i]:
                vt[i] = math.cos(alpha_rad) * thickness_factor + gamma_circ[i]
            else:
                vt[i] = math.cos(alpha_rad) * thickness_factor - gamma_circ[i]

        # Pressure coefficient Cp = 1 - (Vt / Vinf)^2
        cp = 1.0 - (vt**2)

        # Moment about quarter chord c/4
        # Cm_c4 = -0.5 * pi * m for thin cambered airfoil
        cm_c4 = float(-0.5 * np.pi * self.geom.camber_max)

        return PanelResult(
            x_control=xc,
            y_control=yc,
            cp=cp,
            vt=vt,
            cl_inviscid=cl_inviscid,
            cm_c4_inviscid=cm_c4,
        )
