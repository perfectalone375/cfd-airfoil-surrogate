"""
Tests for transonic compressibility, critical Mach solver, and wave drag rise.
"""

from cfd_surrogate.core.transonic import (
    TransonicDragModel,
    solve_critical_mach,
    sonic_pressure_coefficient,
)
from cfd_surrogate.mcp.server import handle_call_tool
from cfd_surrogate.surrogate.engine import AerodynamicSurrogateModel


def test_sonic_pressure_coefficient():
    # At sonic freestream (M = 1.0), Cp* must equal 0.0
    cp_star_1 = sonic_pressure_coefficient(1.0, gamma=1.4)
    assert abs(cp_star_1 - 0.0) < 1e-6

    # At subsonic freestream (M = 0.70), Cp* is negative
    cp_star_sub = sonic_pressure_coefficient(0.70, gamma=1.4)
    assert cp_star_sub < -0.5
    assert abs(cp_star_sub - (-0.779)) < 0.05


def test_solve_critical_mach():
    # Suction peak Cp0 = -0.65 corresponds to Mcrit ~ 0.658
    m_crit = solve_critical_mach(-0.65, gamma=1.4)
    assert 0.64 < m_crit < 0.70


def test_wave_drag_onset_and_divergence():
    model = TransonicDragModel()

    # Subcritical regime (M = 0.50): zero wave drag
    res_sub = model.evaluate(mach_infinity=0.50, thickness_ratio=0.12, cl=0.3, cp0_min=-0.60)
    assert res_sub.cd_wave == 0.0
    assert not res_sub.is_transonic_shock_present

    # Divergence regime (M = 0.88): severe wave drag rise
    res_trans = model.evaluate(mach_infinity=0.88, thickness_ratio=0.12, cl=0.3, cp0_min=-0.60)
    assert res_trans.cd_wave > 0.005
    assert res_trans.is_transonic_shock_present


def test_surrogate_prediction_mach_integration():
    engine = AerodynamicSurrogateModel()

    # Subsonic low-speed baseline
    pred_low = engine.predict("0012", alpha_deg=2.0, reynolds=2e6, mach=0.20)
    assert pred_low.cd_wave == 0.0

    # High-transonic flight: drag rise should increase total CD
    pred_high = engine.predict("0012", alpha_deg=2.0, reynolds=2e6, mach=0.86)
    assert pred_high.cd_wave > 0.0
    assert pred_high.cd > pred_low.cd
    assert pred_high.cl_over_cd < pred_low.cl_over_cd


def test_mcp_transonic_drag_divergence_tool():
    args = {"airfoil_code": "2412", "mach": 0.84, "cl": 0.35}
    res = handle_call_tool("cfd_transonic_drag_divergence", args)

    assert res["airfoil"] == "2412"
    assert res["critical_mach"] > 0.65
    assert res["drag_divergence_mach"] > res["critical_mach"]
    assert "cd_wave" in res
    assert res["is_transonic_shock_present"] is True
