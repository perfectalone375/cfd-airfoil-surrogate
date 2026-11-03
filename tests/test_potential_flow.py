import pytest

from cfd_surrogate.core.airfoil import generate_naca4
from cfd_surrogate.core.potential_flow import VortexPanelSolver


def test_vortex_panel_solver_zero_alpha():
    geom = generate_naca4("0012", n_points=40)
    solver = VortexPanelSolver(geom)
    res = solver.solve(alpha_deg=0.0)
    # Symmetric airfoil at alpha=0 should have cl ~ 0
    assert pytest.approx(res.cl_inviscid, abs=0.05) == 0.0
    assert len(res.cp) > 0


def test_vortex_panel_solver_positive_alpha():
    geom = generate_naca4("2412", n_points=40)
    solver = VortexPanelSolver(geom)
    res = solver.solve(alpha_deg=5.0)
    assert res.cl_inviscid > 0.6
    assert res.cm_c4_inviscid < 0.0
