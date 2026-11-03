import pytest

from cfd_surrogate.core.airfoil import generate_naca4


def test_naca0012_symmetric():
    geom = generate_naca4("0012", n_points=50)
    assert geom.camber_max == 0.0
    assert geom.thickness_max == 0.12
    assert len(geom.x_upper) == 50
    # Symmetric airfoil upper and lower magnitudes match
    assert pytest.approx(geom.y_upper[25], abs=1e-3) == -geom.y_lower[25]


def test_naca2412_cambered():
    geom = generate_naca4("2412", n_points=50)
    assert geom.camber_max == 0.02
    assert geom.camber_pos == 0.40
    assert geom.thickness_max == 0.12
    # Cambered airfoil has positive upper surface elevation
    assert geom.y_upper[25] > 0.05


def test_invalid_naca_code():
    with pytest.raises(ValueError, match="Invalid NACA 4-digit code"):
        generate_naca4("abc")
