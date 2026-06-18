from pathlib import Path
import os
import sys


UNIPH_ROOT = Path(__file__).resolve().parents[2]
if str(UNIPH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIPH_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(UNIPH_ROOT / ".matplotlib"))

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.utils_calculator import UnIPHCalculator, load_graphene_model
from utils.utils_device import resolve_device


# Calculation settings. GRID_N controls the UnIPH sliding grid resolution.
DEVICE = resolve_device("auto")
GRID_N = 20

# Input/output paths are kept local to this test folder so the script can be
# run from any working directory.
HERE = Path(__file__).resolve().parent
STRUCTURE_PATH = HERE / "structures" / "AA_3.534.xyz"
DFT_STRUCTURES_PATH = HERE / "reference" / "total_AA_z_3.534.xyz"
DFT_VAL_PATH = HERE / "reference" / "val.txt"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the graphene checkpoint and wrap it as an ASE calculator so energy calls
# look like normal ASE workflows.
context = load_graphene_model(device=DEVICE)
calculator = UnIPHCalculator(context)

# The top graphene layer is translated by fractional amounts of the in-plane
# lattice vectors a1 and a2. The xy components are stored for plotting.
base_atoms = ase.io.read(STRUCTURE_PATH)
cell = base_atoms.get_cell()
a1 = np.array(cell[0])
a2 = np.array(cell[1])
a1_xy = a1[:2]
a2_xy = a2[:2]

val1_list = np.linspace(0.0, 1.0, GRID_N)
val2_list = np.linspace(0.0, 1.0, GRID_N)
uniph_rows = []

# Sweep a full unit-cell sliding map and evaluate each translated structure
# with UnIPH. Energies are stored raw first, then shifted to a relative scale.
for val1 in val1_list:
    for val2 in val2_list:
        atoms = base_atoms.copy()
        positions = atoms.get_positions()
        top_mask = positions[:, 2] > np.max(positions[:, 2]) - 1.0
        positions[top_mask] += val1 * a1 + val2 * a2
        atoms.set_positions(positions)
        atoms.calc = calculator
        energy = atoms.get_potential_energy()
        xy = val1 * a1_xy + val2 * a2_xy
        uniph_rows.append({
            "val1": val1,
            "val2": val2,
            "x_A": xy[0],
            "y_A": xy[1],
            "energy_meV_per_atom_raw": energy / 4.0 * 1000.0,
        })

uniph_df = pd.DataFrame(uniph_rows)
uniph_df["energy_meV_per_atom"] = (
    uniph_df["energy_meV_per_atom_raw"] - uniph_df["energy_meV_per_atom_raw"].min()
)
uniph_df.to_csv(OUTPUT_DIR / "gsfe_uniph.csv", index=False)

# Load the reference DFT structures and put them into the same tabular layout as
# the UnIPH results so the comparison plot uses identical coordinates/units.
dft_structures = ase.io.read(DFT_STRUCTURES_PATH, index=":", format="extxyz")
dft_values = np.loadtxt(DFT_VAL_PATH)
dft_grid_n = int(np.sqrt(len(dft_structures)))
dft_val1 = np.linspace(0.0, 1.0, dft_grid_n)
dft_val2 = np.linspace(0.0, 1.0, dft_grid_n)
dft_rows = []

for idx, atoms in enumerate(dft_structures):
    i = idx // dft_grid_n
    j = idx % dft_grid_n
    val1 = dft_val1[j]
    val2 = dft_val2[i]
    xy = val1 * a1_xy + val2 * a2_xy
    dft_rows.append({
        "index": int(dft_values[idx, 0]) if idx < len(dft_values) else idx,
        "val1": val1,
        "val2": val2,
        "x_A": xy[0],
        "y_A": xy[1],
        "energy_meV_per_atom_raw": atoms.get_potential_energy() / 4.0 * 1000.0,
    })

dft_df = pd.DataFrame(dft_rows)
dft_df["energy_meV_per_atom"] = (
    dft_df["energy_meV_per_atom_raw"] - dft_df["energy_meV_per_atom_raw"].min()
)
dft_df.to_csv(OUTPUT_DIR / "gsfe_dft.csv", index=False)

# Reshape the tables back into 2D grids for contour plotting.
uniph_z = uniph_df["energy_meV_per_atom"].to_numpy().reshape(GRID_N, GRID_N).T
uniph_x = uniph_df["x_A"].to_numpy().reshape(GRID_N, GRID_N).T
uniph_y = uniph_df["y_A"].to_numpy().reshape(GRID_N, GRID_N).T

dft_z = dft_df["energy_meV_per_atom"].to_numpy().reshape(dft_grid_n, dft_grid_n)
dft_x = dft_df["x_A"].to_numpy().reshape(dft_grid_n, dft_grid_n)
dft_y = dft_df["y_A"].to_numpy().reshape(dft_grid_n, dft_grid_n)

# Use one color scale for both panels so color differences are directly
# comparable between DFT and UnIPH.
vmax = max(3.6, float(np.nanmax([uniph_z.max(), dft_z.max()])))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

contour_dft = axes[0].contourf(dft_x, dft_y, dft_z, levels=30, cmap="viridis", vmin=0.0, vmax=vmax)
axes[0].set_title("DFT")
axes[0].set_aspect("equal", adjustable="box")
axes[0].set_xticks([])
axes[0].set_yticks([])

contour_uniph = axes[1].contourf(uniph_x, uniph_y, uniph_z, levels=30, cmap="viridis", vmin=0.0, vmax=vmax)
axes[1].set_title("UnIPH")
axes[1].set_aspect("equal", adjustable="box")
axes[1].set_xticks([])
axes[1].set_yticks([])

fig.colorbar(contour_uniph, ax=axes, label="Relative energy (meV/atom)")
fig.savefig(OUTPUT_DIR / "gsfe_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_DIR / 'gsfe_comparison.png'}")
print(f"Saved {OUTPUT_DIR / 'gsfe_uniph.csv'}")
print(f"Saved {OUTPUT_DIR / 'gsfe_dft.csv'}")
