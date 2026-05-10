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
import torch

from utils.utils_calculator import load_mos2_model, predict_tb


# Calculation settings. The path follows the archive MoS2 comparison: G-M-K-G.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
KPTS = 30

# Input/output paths are local to this test folder.
HERE = Path(__file__).resolve().parent
STRUCTURE_PATH = HERE / "structures" / "mos2_1x1.xyz"
DFT_BAND_PATH = HERE / "reference" / "bands.out.gnu"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def orbital_count(angular_channels):
    return sum(2 * int(l_val) + 1 for l_val in angular_channels)


def atom_orbital_slices(atoms, orbital_types):
    slices = []
    start = 0
    for symbol in atoms.get_chemical_symbols():
        width = orbital_count(orbital_types[symbol])
        slices.append(slice(start, start + width))
        start += width
    return slices, start


def parse_dft_bands(path):
    bands = []
    current = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            if current:
                bands.append(current)
                current = []
            continue
        current.append((float(parts[0]), float(parts[1])))
    if current:
        bands.append(current)

    dft_k = np.array([row[0] for row in bands[0]])
    dft_energies = np.array([[row[1] for row in band] for band in bands])
    return dft_k, dft_energies


# Load the MoS2 model and predict edge-wise tight-binding hopping blocks.
context = load_mos2_model(device=DEVICE)
atoms = ase.io.read(STRUCTURE_PATH, format="extxyz")
graph, hopping_mats = predict_tb(atoms, context)

edge_src = graph.edge_index[0].detach().cpu().numpy()
edge_dst = graph.edge_index[1].detach().cpu().numpy()
edge_shift = graph.edge_shift.detach().cpu().numpy()
orbital_slices, n_orb = atom_orbital_slices(atoms, context["model_cfg"]["orbital_types"])

# Build the high-symmetry k-path and its plotting axis.
bandpath = atoms.cell.bandpath("GMKG", npoints=KPTS)
k_vecs = np.array(bandpath.kpts)
k_dist, node_dist, labels = bandpath.get_linear_kpoint_axis()
labels = [r"$\Gamma$" if label == "G" else label for label in labels]

# Convert flat predicted hopping blocks into matrix blocks for H(k).
hoppings = []
for hop, src, dst, shift in zip(hopping_mats, edge_src, edge_dst, edge_shift):
    src_slice = orbital_slices[int(src)]
    dst_slice = orbital_slices[int(dst)]
    src_dim = src_slice.stop - src_slice.start
    dst_dim = dst_slice.stop - dst_slice.start
    hop_matrix = np.asarray(hop).reshape(src_dim, dst_dim)
    hop_matrix = np.real_if_close(hop_matrix).astype(np.complex128)
    hoppings.append((hop_matrix, src_slice, dst_slice, np.array(shift, dtype=float)))

# Build and diagonalize the Bloch Hamiltonian at each k-point.
energies_uniph = []
for k_frac in k_vecs:
    ham = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for hop_matrix, src_slice, dst_slice, shift in hoppings:
        phase = np.exp(2j * np.pi * np.dot(k_frac, shift))
        ham[src_slice, dst_slice] += hop_matrix * phase
    ham = 0.5 * (ham + ham.conj().T)
    energies_uniph.append(np.linalg.eigvalsh(ham))
energies_uniph = np.array(energies_uniph)

# Parse the DFT band file. It has one band per block separated by blank lines.
dft_k, dft_energies = parse_dft_bands(DFT_BAND_PATH)
dft_k_plot = k_dist if len(dft_k) == len(k_dist) else dft_k / np.max(dft_k) * np.max(k_dist)

np.savez(
    OUTPUT_DIR / "band_mos2.npz",
    k_dist=k_dist,
    node_dist=node_dist,
    labels=np.array(labels),
    energies_uniph=energies_uniph,
    dft_k=dft_k,
    dft_k_plot=dft_k_plot,
    dft_energies=dft_energies,
)

pd.DataFrame(energies_uniph, columns=[f"band_{idx}" for idx in range(energies_uniph.shape[1])]).assign(
    k_dist=k_dist
).to_csv(OUTPUT_DIR / "band_mos2_uniph.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 6))
for idx in range(dft_energies.shape[0]):
    ax.plot(
        dft_k_plot,
        dft_energies[idx],
        color="black",
        linestyle="-",
        linewidth=1.5,
        label="DFT" if idx == 0 else None,
    )

for idx in range(energies_uniph.shape[1]):
    ax.plot(
        k_dist,
        energies_uniph[:, idx],
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label="UnIPH" if idx == 0 else None,
    )

for x_val in node_dist:
    ax.axvline(x_val, color="black", linestyle=":", linewidth=1.0)

ax.set_xticks(node_dist)
ax.set_xticklabels(labels)
ax.set_ylabel("Energy (eV)")
ax.set_xlim(0.0, float(np.max(k_dist)))
ax.set_ylim(-10.0, 4.0)
ax.legend(frameon=False, loc="upper right")
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "band_mos2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_DIR / 'band_mos2.png'}")
print(f"Saved {OUTPUT_DIR / 'band_mos2.npz'}")
print(f"Saved {OUTPUT_DIR / 'band_mos2_uniph.csv'}")
