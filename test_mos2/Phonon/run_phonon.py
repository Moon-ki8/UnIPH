from pathlib import Path
import os
import sys


UNIPH_ROOT = Path(__file__).resolve().parents[2]
if str(UNIPH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIPH_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(UNIPH_ROOT / ".matplotlib"))

import ase.io
from ase.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils_calculator import UnIPHCalculator, load_mos2_model
from utils.utils_device import resolve_device


# Calculation settings from the archive MoS2 phonon script.
DEVICE = resolve_device("auto")
SUPERCELL = (3, 3, 1)
DELTA = 0.01
KPTS = 40
DFT_GROUP_TOL = 0.1

# Input/output paths are local to this test folder.
HERE = Path(__file__).resolve().parent
STRUCTURE_PATH = HERE / "structures" / "mos2_1x1.xyz"
DFT_PHONON_PATH = HERE / "reference" / "df_phonon.csv"
OUTPUT_DIR = HERE / "outputs"
PHONON_CACHE = OUTPUT_DIR / "phonon_cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PHONON_CACHE.mkdir(parents=True, exist_ok=True)


def group_dft_points(points, tol):
    points = points[points[:, 0].argsort()]
    groups = []
    current = [points[0]]

    for point in points[1:]:
        if abs(point[0] - current[-1][0]) < tol:
            current.append(point)
        else:
            groups.append(current)
            current = [point]
    groups.append(current)

    grouped = []
    for group in groups:
        group = np.array(group)
        avg_x = np.mean(group[:, 0])
        for _, energy in group:
            grouped.append([avg_x, energy])
    return np.array(grouped)


# Load UnIPH as an ASE calculator. Force calls must keep autograd enabled inside
# the calculator, so do not wrap phonon evaluation in torch.no_grad().
context = load_mos2_model(device=DEVICE)
calculator = UnIPHCalculator(context)
atoms = ase.io.read(STRUCTURE_PATH, format="extxyz")
atoms.calc = calculator

# ASE finite-displacement phonons. The cache path keeps generated displacement
# files out of the source/reference directories.
ph = Phonons(
    atoms,
    calculator,
    supercell=SUPERCELL,
    delta=DELTA,
    name=str(PHONON_CACHE / "mos2"),
)
ph.run()
ph.read(reuse=False, acoustic=True)

path = atoms.cell.bandpath("GMKG", npoints=KPTS)
bs = ph.get_band_structure(path)
kpath, kpoints, labels = bs.path.get_linear_kpoint_axis()
labels = [r"$\Gamma$" if label == "G" else label for label in labels]
energies = bs.energies.squeeze()
energies_mev = energies * 1000.0

dft_points = group_dft_points(
    np.loadtxt(DFT_PHONON_PATH, delimiter=","),
    tol=DFT_GROUP_TOL,
)

np.savez(
    OUTPUT_DIR / "phonon_mos2.npz",
    kpath=kpath,
    kpoints=kpoints,
    labels=np.array(labels),
    energies=energies,
    energies_mev=energies_mev,
    dft_points=dft_points,
)

pd.DataFrame(energies_mev, columns=[f"mode_{idx}" for idx in range(energies_mev.shape[1])]).assign(
    kpath=kpath
).to_csv(OUTPUT_DIR / "phonon_mos2_uniph.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
for mode_idx in range(energies_mev.shape[1]):
    ax.plot(
        kpath,
        energies_mev[:, mode_idx],
        color="tab:orange",
        linewidth=2.0,
        label="UnIPH" if mode_idx == 0 else None,
    )

ax.plot(
    dft_points[:, 0],
    dft_points[:, 1],
    color="black",
    marker="o",
    linestyle="None",
    markersize=4,
    label="DFT",
)

for point in kpoints:
    ax.axvline(point, color="gray", linestyle="--", linewidth=1.0)

ax.set_xticks(kpoints)
ax.set_xticklabels(labels)
ax.set_ylabel("Energy (meV)")
ax.set_xlim(-0.1, float(np.max(kpath)) + 0.1)
ax.set_ylim(0.0, 75.0)
ax.legend(frameon=False, loc="upper right", ncols=2)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "phonon_mos2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_DIR / 'phonon_mos2.png'}")
print(f"Saved {OUTPUT_DIR / 'phonon_mos2.npz'}")
print(f"Saved {OUTPUT_DIR / 'phonon_mos2_uniph.csv'}")
