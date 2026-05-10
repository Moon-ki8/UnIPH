from pathlib import Path
import argparse
import sys

try:
    import ase.io
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.visualize import view
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    print(
        f"Missing Python package: {missing}\n"
        "Install the finite_T requirements first, for example:\n"
        "  python -m pip install ase\n",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


HERE = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY = HERE / "outputs" / "trajectory.xyz"


def parse_index(index):
    if index.lstrip("-").isdigit():
        return int(index)
    return index


def attach_forces_if_needed(atoms):
    if atoms.calc is not None or "forces" not in atoms.arrays:
        return atoms

    results = {"forces": atoms.arrays["forces"]}
    energy = atoms.info.get("energy", atoms.info.get("potential_energy_eV"))
    if energy is not None:
        results["energy"] = float(energy)

    atoms.calc = SinglePointCalculator(atoms, **results)
    return atoms


def main():
    parser = argparse.ArgumentParser(
        description="Open the finite-temperature graphene trajectory in ASE GUI."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=DEFAULT_TRAJECTORY,
        help=f"Trajectory to view. Default: {DEFAULT_TRAJECTORY}",
    )
    parser.add_argument(
        "--index",
        default=":",
        help="ASE frame index, e.g. ':' for all frames, '0' for first, '-1' for last.",
    )
    args = parser.parse_args()

    images = ase.io.read(args.trajectory, index=parse_index(args.index), format="extxyz")
    if not isinstance(images, list):
        images = [images]

    images = [attach_forces_if_needed(atoms) for atoms in images]
    view(images, viewer="ase")


if __name__ == "__main__":
    main()
