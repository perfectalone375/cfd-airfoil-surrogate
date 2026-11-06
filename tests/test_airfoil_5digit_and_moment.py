"""Tests for NACA 5-digit airfoil generator and aerodynamic moment / center of pressure."""
import pytest
from cfd_surrogate.core.airfoil import generate_airfoil, generate_naca4, generate_naca5
from cfd_surrogate.surrogate.engine import AerodynamicSurrogateModel


def test_naca23012_geometry():
    geom = generate_naca5("23012", n_points=50)
    assert geom.code == "23012"
    assert geom.thickness_max == 0.12
    assert geom.camber_pos == 0.15
    assert len(geom.x_upper) == 50
    assert len(geom.x_lower) == 50
    # Camber should be positive
    assert geom.camber_max > 0.0


def test_generate_airfoil_dispatcher():
    # 4-digit delegation
    g4 = generate_airfoil("2412", n_points=30)
    assert g4.code == "2412"

    # 5-digit delegation
    g5 = generate_airfoil("23015", n_points=30)
    assert g5.code == "23015"
    assert g5.thickness_max == 0.15

    # With NACA prefix
    g_pref = generate_airfoil("NACA 0012", n_points=30)
    assert g_pref.code == "0012"

    with pytest.raises(ValueError, match="Unsupported NACA airfoil code"):
        generate_airfoil("NACA 64-212")


def test_center_of_pressure_and_moment():
    surrogate = AerodynamicSurrogateModel()
    # At positive angle of attack, CL > 0
    pred = surrogate.predict("2412", alpha_deg=5.0)
    assert pred.cl > 0
    assert pred.xcp_over_c is not None
    # For cambered airfoil at positive lift, center of pressure is typically ahead of 0.50 chord
    assert 0.10 <= pred.xcp_over_c <= 0.60
    assert pred.cm_le is not None


def test_symmetric_airfoil_moment():
    surrogate = AerodynamicSurrogateModel()
    pred = surrogate.predict("0012", alpha_deg=4.0)
    # Symmetric airfoil has Cm_c4 ~ 0
    assert pred.cm == pytest.approx(0.0, abs=1e-3)
    # Center of pressure at quarter chord for symmetric airfoil in thin airfoil theory
    assert pred.xcp_over_c == pytest.approx(0.25, abs=0.01)
