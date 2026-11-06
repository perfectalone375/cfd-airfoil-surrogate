"""CFD Airfoil Surrogate Model package."""
from .core.airfoil import AirfoilGeometry, generate_airfoil, generate_naca4, generate_naca5
from .core.boundary_layer import BoundaryLayerResult, BoundaryLayerSolver
from .core.polar import AerodynamicPolar, PolarPoint, compute_polar
from .core.potential_flow import PanelResult, VortexPanelSolver
from .surrogate.engine import AerodynamicSurrogateModel, SurrogatePrediction

__version__ = "0.1.0"
__all__ = [
    "AirfoilGeometry",
    "generate_airfoil",
    "generate_naca4",
    "generate_naca5",
    "BoundaryLayerResult",
    "BoundaryLayerSolver",
    "AerodynamicPolar",
    "PolarPoint",
    "compute_polar",
    "PanelResult",
    "VortexPanelSolver",
    "AerodynamicSurrogateModel",
    "SurrogatePrediction",
]
