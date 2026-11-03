from cfd_surrogate.core.polar import compute_polar
from cfd_surrogate.surrogate.engine import AerodynamicSurrogateModel


def test_compute_polar():
    polar = compute_polar("2412", alpha_min=-2.0, alpha_max=10.0, alpha_step=2.0, reynolds=1e6)
    assert polar.cl_max > 1.0
    assert polar.cd_min > 0.005
    assert polar.max_ld > 30.0
    assert len(polar.points) == 7


def test_surrogate_model():
    model = AerodynamicSurrogateModel()
    pred = model.predict("2412", alpha_deg=4.0, reynolds=1e6)
    assert 0.5 < pred.cl < 0.9
    assert 0.006 < pred.cd < 0.015
    assert pred.cl_over_cd > 40.0
    assert len(pred.cp_upper) == 50
    assert len(pred.cp_lower) == 50
