from cfd_surrogate.core.boundary_layer import BoundaryLayerSolver


def test_boundary_layer_solver():
    solver = BoundaryLayerSolver(reynolds=1e6)
    res = solver.solve(thickness_ratio=0.12, camber=0.02, alpha_deg=4.0, cl=0.6)
    assert res.cd_total > 0.005
    assert res.cd_total < 0.030
    assert res.cf_average > 0.001
    assert not res.separation_detected


def test_boundary_layer_stall():
    solver = BoundaryLayerSolver(reynolds=1e6)
    res = solver.solve(thickness_ratio=0.12, camber=0.0, alpha_deg=18.0, cl=1.2)
    assert res.separation_detected
    assert res.cd_total > 0.04
