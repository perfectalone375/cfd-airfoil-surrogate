"""NACA 4-digit airfoil parametric geometry generator with cosine clustering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AirfoilGeometry:
    code: str
    chord: float
    x_upper: np.ndarray
    y_upper: np.ndarray
    x_lower: np.ndarray
    y_lower: np.ndarray
    camber_max: float
    camber_pos: float
    thickness_max: float

    @property
    def x_all(self) -> np.ndarray:
        """Continuous closed coordinate loop from trailing edge upper to trailing edge lower."""
        return np.concatenate([self.x_upper[::-1], self.x_lower[1:]])

    @property
    def y_all(self) -> np.ndarray:
        return np.concatenate([self.y_upper[::-1], self.y_lower[1:]])


def generate_naca4(code: str, n_points: int = 100, chord: float = 1.0) -> AirfoilGeometry:
    """Generate coordinates for a standard NACA 4-digit airfoil using cosine spacing.

    Args:
        code: 4-digit string, e.g. "0012", "2412", "4412"
        n_points: Number of points per surface
        chord: Airfoil chord length in meters
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError(f"Invalid NACA 4-digit code: '{code}'. Must be 4 digits (e.g., '2412').")

    m = int(code[0]) / 100.0  # Maximum camber
    p = int(code[1]) / 10.0   # Position of maximum camber
    t = int(code[2:]) / 100.0 # Maximum thickness ratio

    # Cosine spacing for high leading edge resolution: theta in [0, pi]
    beta = np.linspace(0, np.pi, n_points)
    x = (1.0 - np.cos(beta)) / 2.0  # x/c from 0 to 1

    # Thickness distribution y_t
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1036
    yt = (t / 0.2) * (
        a0 * np.sqrt(x) + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4)
    )

    # Camber line y_c and slope theta
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    if m > 0 and p > 0:
        idx_front = x <= p
        idx_back = x > p

        yc[idx_front] = (m / (p**2)) * (2 * p * x[idx_front] - x[idx_front] ** 2)
        dyc_dx[idx_front] = (2 * m / (p**2)) * (p - x[idx_front])

        yc[idx_back] = (m / ((1 - p) ** 2)) * (
            (1 - 2 * p) + 2 * p * x[idx_back] - x[idx_back] ** 2
        )
        dyc_dx[idx_back] = (2 * m / ((1 - p) ** 2)) * (p - x[idx_back])

    theta = np.arctan(dyc_dx)

    # Upper and lower surfaces
    xu = (x - yt * np.sin(theta)) * chord
    yu = (yc + yt * np.cos(theta)) * chord
    xl = (x + yt * np.sin(theta)) * chord
    yl = (yc - yt * np.cos(theta)) * chord

    return AirfoilGeometry(
        code=code,
        chord=chord,
        x_upper=xu,
        y_upper=yu,
        x_lower=xl,
        y_lower=yl,
        camber_max=m,
        camber_pos=p,
        thickness_max=t,
    )
