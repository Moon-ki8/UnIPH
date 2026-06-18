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


# Calculation settings. The script scans vertical layer spacing for three
# stacking registries and compares the relative energy curves with DFT.
DEVICE = resolve_device("auto")
D_MIN = 3.4
D_MAX = 3.8
N_POINTS = 20

# Input/output paths are kept local to this test folder so the script can be
# run from any working directory. Environment variables make checkpoint/output
# overrides convenient for quick model comparisons.
HERE = Path(__file__).resolve().parent
STRUCTURE_DIR = HERE / "structures"
REFERENCE_PATH = HERE / "reference" / "energy_rec.txt"
OUTPUT_DIR = Path(os.environ.get("UNIPH_INTER_LAYER_OUTPUT_DIR", HERE / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = os.environ.get("UNIPH_GRAPHENE_CHECKPOINT")

# Load the graphene checkpoint and wrap it as an ASE calculator for energy
# evaluation. By default this uses UnIPH/uniph_graphene.torch.
context = load_graphene_model(device=DEVICE, checkpoint_path=CHECKPOINT_PATH)
print(f"Using checkpoint: {context['checkpoint_path']}")
calculator = UnIPHCalculator(context)

d_layer = np.linspace(D_MIN, D_MAX, N_POINTS)
stacking_files = {
    "AA": STRUCTURE_DIR / "relax_AA_32.xyz",
    "AB": STRUCTURE_DIR / "relax_AB_32.xyz",
    "SP": STRUCTURE_DIR / "relax_SP_32.xyz",
}

pred_total = {name: [] for name in stacking_files}
for name, path in stacking_files.items():
    atoms = ase.io.read(path, format="extxyz")
    original_positions = atoms.get_positions().copy()
    atoms.calc = calculator

    # Keep the bottom layer fixed and move only the top layer to the requested
    # interlayer spacing.
    for spacing in d_layer:
        positions = original_positions.copy()
        top_mask = original_positions[:, 2] > 3.0
        positions[top_mask, 2] = 1.0 + spacing
        atoms.set_positions(positions)
        pred_total[name].append(atoms.get_potential_energy())

# Reference file columns are spacing, AA, AB, and SP total energies.
reference = np.loadtxt(REFERENCE_PATH)
ref_spacing = reference[:, 0]
ref_total = {
    "AA": reference[:, 1],
    "AB": reference[:, 2],
    "SP": reference[:, 3],
}

# Express all curves as meV/atom relative to the AB minimum, matching the common
# convention for bilayer graphene interlayer energy comparisons.
ref_rel = {
    name: values / 4.0 - np.min(ref_total["AB"] / 4.0)
    for name, values in ref_total.items()
}
pred_rel = {
    name: np.array(values) / 4.0 - np.min(np.array(pred_total["AB"]) / 4.0)
    for name, values in pred_total.items()
}

rows = []
for idx, spacing in enumerate(d_layer):
    row = {"spacing_A": spacing}
    for name in ["AA", "AB", "SP"]:
        row[f"uniph_{name}_meV_per_atom"] = pred_rel[name][idx] * 1000.0
    rows.append(row)

for idx, spacing in enumerate(ref_spacing):
    row = {"spacing_A": spacing}
    for name in ["AA", "AB", "SP"]:
        row[f"dft_{name}_meV_per_atom"] = ref_rel[name][idx] * 1000.0
    rows.append(row)

pd.DataFrame(rows).to_csv(OUTPUT_DIR / "inter_layer_E.csv", index=False)

# Overlay DFT points and UnIPH curves for each stacking.
fig, ax = plt.subplots(figsize=(7, 6))
colors = {"AA": "tab:blue", "AB": "tab:orange", "SP": "tab:green"}
markers = {"AA": "o", "AB": "s", "SP": "^"}

for name in ["AA", "AB", "SP"]:
    ax.plot(
        ref_spacing,
        ref_rel[name] * 1000.0,
        marker=markers[name],
        linestyle="None",
        color=colors[name],
        label=f"{name} DFT",
    )
    ax.plot(
        d_layer,
        pred_rel[name] * 1000.0,
        linestyle="--",
        linewidth=2.0,
        color=colors[name],
        label=f"{name} UnIPH",
    )

ax.set_xlabel("Interlayer spacing (Angstrom)")
ax.set_ylabel("Relative energy (meV/atom)")
ax.set_xlim(D_MIN, D_MAX)
ax.legend(frameon=False, ncols=2)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "inter_layer_E.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_DIR / 'inter_layer_E.png'}")
print(f"Saved {OUTPUT_DIR / 'inter_layer_E.csv'}")
