from pathlib import Path
import os
import sys


UNIPH_ROOT = Path(__file__).resolve().parents[2]
if str(UNIPH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIPH_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(UNIPH_ROOT / ".matplotlib"))

import ase.io
from ase import units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.md.verlet import VelocityVerlet
import numpy as np
import torch

from utils.utils_calculator import UnIPHCalculator, load_graphene_model
from utils.utils_device import resolve_device


# Finite-temperature settings.
DEVICE = resolve_device("auto")
TEMPERATURE_K = 300
TIME_STEP_FS = 0.5
N_STEPS = 100
RANDOM_SEED = 7

# Start from AB-stacked bilayer graphene and repeat it to 2x2.
HERE = Path(__file__).resolve().parent
SOURCE_AB_PATH = HERE.parents[0] / "inter_layer_E" / "structures" / "relax_AB_32.xyz"
INITIAL_PATH = HERE / "structures" / "graphene_2x2_initial.xyz"
OUTPUT_DIR = HERE / "outputs"
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory.xyz"
LOG_PATH = OUTPUT_DIR / "md.log"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INITIAL_PATH.parent.mkdir(parents=True, exist_ok=True)


def prepare_initial_structure():
    atoms = ase.io.read(SOURCE_AB_PATH, format="extxyz")
    atoms_2x2 = atoms.repeat((2, 2, 1))
    ase.io.write(INITIAL_PATH, atoms_2x2, format="extxyz")
    return atoms_2x2


def write_frame(atoms, step, append=True):
    atoms.info["md_step"] = step
    atoms.info["temperature_K"] = atoms.get_temperature()
    atoms.info["potential_energy_eV"] = atoms.get_potential_energy()
    atoms.info["kinetic_energy_eV"] = atoms.get_kinetic_energy()
    ase.io.write(TRAJECTORY_PATH, atoms, format="extxyz", append=append)


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

for path in (TRAJECTORY_PATH, LOG_PATH):
    if path.exists():
        path.unlink()

atoms = prepare_initial_structure()
context = load_graphene_model(device=DEVICE)
atoms.calc = UnIPHCalculator(context)

# Assign the initial 300 K Maxwell-Boltzmann velocity distribution, then run NVE.
MaxwellBoltzmannDistribution(atoms, temperature_K=TEMPERATURE_K, rng=np.random)
Stationary(atoms)
ZeroRotation(atoms)

dyn = VelocityVerlet(atoms, timestep=TIME_STEP_FS * units.fs)

with LOG_PATH.open("w") as log:
    log.write("step temperature_K potential_eV kinetic_eV total_eV\n")
    for step in range(1, N_STEPS + 1):
        dyn.run(1)
        potential = atoms.get_potential_energy()
        kinetic = atoms.get_kinetic_energy()
        temperature = atoms.get_temperature()
        log.write(f"{step} {temperature:.8f} {potential:.12f} {kinetic:.12f} {potential + kinetic:.12f}\n")
        write_frame(atoms, step=step, append=step > 1)

print(f"Saved {INITIAL_PATH}")
print(f"Saved {TRAJECTORY_PATH}")
print(f"Saved {LOG_PATH}")
