from cfd_surrogate.mcp.server import TOOLS, handle_call_tool


def test_mcp_tools_list():
    assert len(TOOLS) == 4
    names = [t["name"] for t in TOOLS]
    assert "cfd_predict_coefficients" in names
    assert "cfd_generate_polar" in names
    assert "cfd_pressure_distribution" in names
    assert "cfd_transonic_drag_divergence" in names


def test_handle_predict_tool():
    res = handle_call_tool("cfd_predict_coefficients", {"airfoil_code": "0012", "alpha_deg": 6.0})
    assert "cl" in res
    assert "cd" in res
    assert res["cl"] > 0.5


def test_handle_polar_tool():
    res = handle_call_tool(
        "cfd_generate_polar", {"airfoil_code": "2412", "alpha_min": 0, "alpha_max": 8, "step": 4}
    )
    assert "cl_max" in res
    assert len(res["polar_curve"]) == 3
