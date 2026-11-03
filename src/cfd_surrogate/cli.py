"""Command line interface for cfd-airfoil-surrogate (aeroml)."""
from __future__ import annotations

import argparse
import json
import sys

from .core.polar import compute_polar
from .surrogate.engine import AerodynamicSurrogateModel


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        prog="aeroml",
        description="AeroML: Physics-Informed CFD Aerodynamic Surrogate & Polar Generator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate instantaneous airfoil aerodynamics")
    ev.add_argument("--airfoil", type=str, default="2412", help="NACA 4-digit code (e.g. 2412)")
    ev.add_argument("--alpha", type=float, default=4.0, help="Angle of attack in degrees")
    ev.add_argument("--reynolds", type=float, default=1e6, help="Reynolds number (default 1e6)")

    # polar
    pol = subparsers.add_parser("polar", help="Generate aerodynamic polar curve")
    pol.add_argument("--airfoil", type=str, default="0012", help="NACA 4-digit code")
    pol.add_argument("--alpha-min", type=float, default=-4.0)
    pol.add_argument("--alpha-max", type=float, default=16.0)
    pol.add_argument("--step", type=float, default=1.0)
    pol.add_argument("--reynolds", type=float, default=1e6)
    pol.add_argument("--json", action="store_true", help="Output JSON format")

    args = parser.parse_args()

    if args.command == "evaluate":
        surrogate = AerodynamicSurrogateModel()
        pred = surrogate.predict(args.airfoil, args.alpha, args.reynolds)
        print("=" * 60)
        print(f"  AeroML Evaluation: NACA {args.airfoil} @ alpha = {args.alpha} deg")
        print(f"  Reynolds Number:   {args.reynolds:.2e}")
        print("-" * 60)
        print(f"  Lift Coeff (CL):   {pred.cl:.4f}")
        print(f"  Drag Coeff (CD):   {pred.cd:.5f}")
        print(f"  Moment Coeff (Cm): {pred.cm:.4f}")
        print(f"  Lift-to-Drag (L/D):{pred.cl_over_cd:.2f}")
        print("=" * 60)

    elif args.command == "polar":
        polar = compute_polar(
            args.airfoil,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_step=args.step,
            reynolds=args.reynolds,
        )
        if args.json:
            print(json.dumps(polar.to_dict(), indent=2))
        else:
            print("=" * 70)
            print(f"  Aerodynamic Polar: NACA {args.airfoil} (Re = {args.reynolds:.1e})")
            print(f"  CL_max: {polar.cl_max:.3f} | CD_min: {polar.cd_min:.4f} | Max L/D: {polar.max_ld:.1f}")
            print("-" * 70)
            print(f"  {'Alpha (deg)':>12} | {'CL':>8} | {'CD':>8} | {'Cm':>8} | {'L/D':>8} | {'Stall':>6}")
            print("-" * 70)
            for p in polar.points:
                stall_str = "YES" if p.separation else "no"
                print(f"  {p.alpha_deg:12.1f} | {p.cl:8.4f} | {p.cd:8.5f} | {p.cm:8.4f} | {p.cl_over_cd:8.2f} | {stall_str:>6}")
            print("=" * 70)


if __name__ == "__main__":
    main()
