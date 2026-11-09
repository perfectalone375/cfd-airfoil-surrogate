"""
Transonic compressibility, critical Mach number solver, and wave drag divergence modeling.
Uses Karman-Tsien compressibility correction and Korn's transonic equation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TransonicDivergenceResult:
    """Aerodynamic transonic regime parameters and wave drag components."""

    mach_infinity: float
    critical_mach: float
    drag_divergence_mach: float
    sonic_cp_star: float
    compressible_cp_min: float
    cd_wave: float
    is_transonic_shock_present: bool


def sonic_pressure_coefficient(mach: float, gamma: float = 1.4) -> float:
    """
    Calculate the sonic pressure coefficient Cp* where local flow reaches Mach 1.0:
    Cp* = (2 / (gamma * M^2)) * [ ((2 + (gamma - 1) * M^2) / (gamma + 1))^(gamma / (gamma - 1)) - 1 ]
    """
    if mach <= 0.0 or math.isclose(mach, 0.0):
        return -float("inf")
    term1 = 2.0 / (gamma * (mach**2))
    base = (2.0 + (gamma - 1.0) * (mach**2)) / (gamma + 1.0)
    exponent = gamma / (gamma - 1.0)
    return term1 * ((base**exponent) - 1.0)


def karman_tsien_cp(cp0: float, mach: float) -> float:
    """
    Karman-Tsien compressibility rule for subsonic/transonic peak suction Cp:
    Cp(M) = Cp0 / [ sqrt(1 - M^2) + (M^2 / (1 + sqrt(1 - M^2))) * (Cp0 / 2) ]
    """
    if mach >= 0.999:
        mach = 0.999
    beta = math.sqrt(1.0 - mach**2)
    denom = beta + ((mach**2) / (1.0 + beta)) * (cp0 / 2.0)
    if abs(denom) < 1e-12:
        return cp0 / beta
    return cp0 / denom


def solve_critical_mach(cp0_min: float, gamma: float = 1.4) -> float:
    """
    Find critical Mach number Mcrit where local suction peak reaches sonic conditions:
    Cp_min(Mcrit) = Cp*(Mcrit).
    Uses bisection root-finding over [0.20, 0.95].
    """
    if cp0_min >= 0.0:
        return 0.95  # Pure flat plate / zero camber-thickness limit

    low, high = 0.20, 0.95
    for _ in range(60):
        mid = (low + high) / 2.0
        cp_mid = karman_tsien_cp(cp0_min, mid)
        cp_star = sonic_pressure_coefficient(mid, gamma)
        # Cp_min is negative (suction). When mid < Mcrit, Cp_min > Cp_star (less suction than sonic)
        if cp_mid > cp_star:
            low = mid
        else:
            high = mid

    return (low + high) / 2.0


class TransonicDragModel:
    """Models transonic drag rise, critical Mach number, and wave drag onset."""

    def __init__(self, gamma: float = 1.4, technology_factor_kappa: float = 0.87):
        self.gamma = gamma
        self.kappa = technology_factor_kappa

    def evaluate(
        self,
        mach_infinity: float,
        thickness_ratio: float,
        cl: float,
        cp0_min: float = -0.65,
    ) -> TransonicDivergenceResult:
        """
        Evaluate critical Mach, drag divergence Mach, and wave drag coefficient CD_wave.
        """
        m_inf = max(0.01, min(1.2, mach_infinity))

        # 1. Exact Critical Mach via Karman-Tsien & Sonic Cp* intersection
        m_crit = solve_critical_mach(cp0_min, self.gamma)

        # 2. Drag Divergence Mach via Mason/Korn empirical correlation:
        # Mdd ~ Mcrit + 0.08 * (1 - t/c)
        m_dd = min(0.98, m_crit + 0.10 * (1.0 - thickness_ratio))

        # 3. Sonic Cp* and Compressible Cp_min at flight Mach
        if m_inf < 1.0:
            cp_star = sonic_pressure_coefficient(m_inf, self.gamma)
            cp_min = karman_tsien_cp(cp0_min, m_inf)
        else:
            cp_star = sonic_pressure_coefficient(0.99, self.gamma)
            cp_min = karman_tsien_cp(cp0_min, 0.99)

        # 4. Wave Drag Rise Calculation
        if m_inf <= m_crit:
            cd_wave = 0.0
            shock_present = False
        elif m_inf <= m_dd:
            # Sub-divergence onset curvature
            progress = (m_inf - m_crit) / max(0.001, (m_dd - m_crit))
            cd_wave = 0.002 * (progress**2)
            shock_present = True
        else:
            # Post-divergence steep power-law wave drag rise
            cd_wave = 0.002 + 20.0 * ((m_inf - m_dd) ** 4)
            shock_present = True

        return TransonicDivergenceResult(
            mach_infinity=round(m_inf, 3),
            critical_mach=round(m_crit, 3),
            drag_divergence_mach=round(m_dd, 3),
            sonic_cp_star=round(cp_star, 4),
            compressible_cp_min=round(cp_min, 4),
            cd_wave=round(cd_wave, 5),
            is_transonic_shock_present=shock_present,
        )
