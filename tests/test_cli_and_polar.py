from cfd_surrogate.core.polar import compute_polar
from cfd_surrogate.surrogate.engine import AerodynamicSurrogateModel


def test_polar_symmetric_zero_lift():
    polar = compute_polar("0012", alpha_min=0.0, alpha_max=0.0, alpha_step=1.0)
    assert abs(polar.points[0].cl) < 1e-4
    assert polar.points[0].cd > 0.007


def test_surrogate_high_angle():
    model = AerodynamicSurrogateModel()
    pred = model.predict("2412", alpha_deg=12.0)
    assert pred.cl > 1.3
    assert pred.cd > 0.015
