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


def generate_naca5(code: str, n_points: int = 100, chord: float = 1.0) -> AirfoilGeometry:
    """Generate coordinates for standard NACA 5-digit airfoils (e.g. 23012, 23015).

    Args:
        code: 5-digit string, e.g. "23012"
        n_points: Number of points per surface
        chord: Airfoil chord length in meters
    """
    if len(code) != 5 or not code.isdigit():
        raise ValueError(f"Invalid NACA 5-digit code: '{code}'. Must be 5 digits (e.g., '23012').")

    cl_design = int(code[0]) * 0.15
    p_code = int(code[1:3])
    t = int(code[3:]) / 100.0

    # Lookup standard NACA 5-digit camber coefficients
    coeffs = {
        10: (0.05, 0.0580, 361.4),
        20: (0.10, 0.1260, 51.64),
        30: (0.15, 0.2025, 15.793),
        40: (0.20, 0.2900, 6.520),
        50: (0.25, 0.3910, 3.191),
    }

    if p_code in coeffs:
        p_pos, m_val, k1_val = coeffs[p_code]
    else:
        # Interpolate / default to 230 series (p_code = 30)
        p_pos, m_val, k1_val = (0.15, 0.2025, 15.793)

    beta = np.linspace(0, np.pi, n_points)
    x = (1.0 - np.cos(beta)) / 2.0

    # Thickness distribution
    a0, a1, a2, a3, a4 = 0.2969, -0.1260, -0.3516, 0.2843, -0.1036
    yt = (t / 0.2) * (a0 * np.sqrt(x) + a1 * x + a2 * (x**2) + a3 * (x**3) + a4 * (x**4))

    # Camber line
    scale = cl_design / 0.3
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)

    idx_front = x <= m_val
    idx_back = x > m_val

    yc[idx_front] = (k1_val / 6.0) * (x[idx_front] ** 3 - 3 * m_val * (x[idx_front] ** 2) + (m_val**2) * (3 - m_val) * x[idx_front]) * scale
    dyc_dx[idx_front] = (k1_val / 6.0) * (3 * (x[idx_front] ** 2) - 6 * m_val * x[idx_front] + (m_val**2) * (3 - m_val)) * scale

    yc[idx_back] = (k1_val * (m_val**3) / 6.0) * (1.0 - x[idx_back]) * scale
    dyc_dx[idx_back] = - (k1_val * (m_val**3) / 6.0) * scale * np.ones_like(x[idx_back])

    theta = np.arctan(dyc_dx)

    xu = (x - yt * np.sin(theta)) * chord
    yu = (yc + yt * np.cos(theta)) * chord
    xl = (x + yt * np.sin(theta)) * chord
    yl = (yc - yt * np.cos(theta)) * chord

    camber_max = float(np.max(yc))
    return AirfoilGeometry(
        code=code,
        chord=chord,
        x_upper=xu,
        y_upper=yu,
        x_lower=xl,
        y_lower=yl,
        camber_max=camber_max,
        camber_pos=p_pos,
        thickness_max=t,
    )


def generate_airfoil(code: str, n_points: int = 100, chord: float = 1.0) -> AirfoilGeometry:
    """Generate airfoil coordinates supporting both NACA 4-digit and 5-digit families."""
    clean_code = code.strip().upper().replace("NACA", "").strip()
    if len(clean_code) == 4:
        return generate_naca4(clean_code, n_points=n_points, chord=chord)
    elif len(clean_code) == 5:
        return generate_naca5(clean_code, n_points=n_points, chord=chord)
    else:
        raise ValueError(f"Unsupported NACA airfoil code: '{code}'. Must be 4 or 5 digits.")

