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

from utils.utils_calculator import load_graphene_model, predict_tb


# Calculation settings. KPTS is the approximate number of sampled k-points along
# the full M-Gamma-K-M path.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
KPTS = 40
DFT_ENERGY_SHIFT = 0.5
UNIPH_ENERGY_SCALE = 1.05

# Input/output paths are kept local to this test folder so the script can be
# run from any working directory.
HERE = Path(__file__).resolve().parent
STRUCTURE_PATH = HERE / "structures" / "relax_AB_rotated.xyz"
DFT_BAND_PATH = HERE / "reference" / "AB_band" / "bands.out.gnu"
OUTPUT_DIR = HERE / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the graphene model, then predict edge-wise tight-binding hoppings for the
# relaxed AB bilayer structure.
context = load_graphene_model(device=DEVICE)
atoms = ase.io.read(STRUCTURE_PATH, format="extxyz")
graph, hopping_mats = predict_tb(atoms, context)

# Pull graph connectivity and periodic image shifts onto NumPy arrays for the
# explicit Bloch Hamiltonian construction below.
cell = graph.lattice[0].detach().cpu().numpy()
edge_src = graph.edge_index[0].detach().cpu().numpy()
edge_dst = graph.edge_index[1].detach().cpu().numpy()
edge_shift = graph.edge_shift.detach().cpu().numpy()
n_orb = len(atoms)

k_nodes = np.array([
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [2.0 / 3.0, 1.0 / 3.0, 0.0],
    [0.5, 0.0, 0.0],
], dtype=float)
k_labels = ["M", r"$\Gamma$", "K", "M"]

# Convert fractional k-node positions into cumulative path distances for the
# x-axis of the band plot.
reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
segment_lengths = [
    np.linalg.norm((k_nodes[idx + 1] - k_nodes[idx]) @ reciprocal)
    for idx in range(len(k_nodes) - 1)
]
node_dist = [0.0]
for length in segment_lengths:
    node_dist.append(node_dist[-1] + length)
node_dist = np.array(node_dist)

k_vecs = []
k_dist = []
segments = len(k_nodes) - 1
base_points = max(2, KPTS // segments)
for idx in range(segments):
    include_endpoint = idx == segments - 1
    points = base_points + (KPTS % segments if idx == segments - 1 else 0)
    fractions = np.linspace(0.0, 1.0, points, endpoint=include_endpoint)
    if idx > 0:
        # Drop duplicate high-symmetry points at segment boundaries.
        fractions = fractions[1:]
    for frac in fractions:
        k_frac = (1.0 - frac) * k_nodes[idx] + frac * k_nodes[idx + 1]
        k_vecs.append(k_frac)
        k_dist.append(node_dist[idx] + frac * segment_lengths[idx])

k_vecs = np.array(k_vecs)
k_dist = np.array(k_dist)

# For graphene in this setup each hopping matrix is 1x1, so the scalar value is
# enough to fill the Hamiltonian.
hoppings = []
for hop, src, dst, shift in zip(hopping_mats, edge_src, edge_dst, edge_shift):
    hop_scalar = np.asarray(hop).reshape(-1)[0]
    hop_scalar = complex(np.real_if_close(hop_scalar))
    hoppings.append((hop_scalar, int(src), int(dst), np.array(shift, dtype=float)))

# Build H(k) from the predicted hoppings and diagonalize it at each sampled
# k-point. The Hermitian symmetrization removes small numerical asymmetries.
energies_uniph = []
for k_frac in k_vecs:
    ham = np.zeros((n_orb, n_orb), dtype=np.complex128)
    for hop, src, dst, shift in hoppings:
        phase = np.exp(2j * np.pi * np.dot(k_frac, shift))
        ham[src, dst] += hop * phase
    ham = 0.5 * (ham + ham.conj().T)
    energies = np.linalg.eigvalsh(ham)
    energies_uniph.append(energies)

energies_uniph = np.array(energies_uniph)
# Center UnIPH bands at the mid-gap Fermi level and apply the empirical plotting
# scale used for this comparison.
occupied_idx = n_orb // 2 - 1
unoccupied_idx = n_orb // 2
fermi_energy = 0.5 * (
    np.max(energies_uniph[:, occupied_idx]) + np.min(energies_uniph[:, unoccupied_idx])
)
energies_uniph = (energies_uniph - fermi_energy) * UNIPH_ENERGY_SCALE

# Quantum ESPRESSO-style gnu band output stores one band per block separated by
# blank lines. Apply the fixed DFT energy shift while parsing.
dft_bands = []
current_band = []
for line in DFT_BAND_PATH.read_text().splitlines():
    parts = line.split()
    if len(parts) == 0:
        if current_band:
            dft_bands.append(current_band)
            current_band = []
        continue
    current_band.append((float(parts[0]), float(parts[1]) + DFT_ENERGY_SHIFT))
if current_band:
    dft_bands.append(current_band)

dft_k = np.array([row[0] for row in dft_bands[0]])
dft_energies = np.array([[row[1] for row in band] for band in dft_bands])

k_norm = k_dist / np.max(k_dist)
node_norm = node_dist / np.max(node_dist)
dft_k_norm = dft_k / np.max(dft_k)

# Save machine-readable arrays and the UnIPH band table before making the plot.
np.savez(
    OUTPUT_DIR / "band_AB.npz",
    k_dist=k_dist,
    k_norm=k_norm,
    node_norm=node_norm,
    energies_uniph=energies_uniph,
    dft_k=dft_k,
    dft_k_norm=dft_k_norm,
    dft_energies=dft_energies,
)

pd.DataFrame(energies_uniph, columns=[f"band_{idx}" for idx in range(energies_uniph.shape[1])]).assign(
    k_norm=k_norm
).to_csv(OUTPUT_DIR / "band_AB_uniph.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 8))
for idx in range(dft_energies.shape[0]):
    ax.plot(
        dft_k_norm,
        dft_energies[idx],
        linestyle="-",
        linewidth=1.5,
        color="black",
        label="DFT" if idx == 0 else None,
    )

for idx in range(energies_uniph.shape[1]):
    ax.plot(
        k_norm,
        energies_uniph[:, idx],
        linestyle="--",
        linewidth=2.0,
        color="tab:orange",
        label="UnIPH" if idx == 0 else None,
    )

for x_val in node_norm:
    ax.axvline(x=x_val, color="black", linestyle=":", linewidth=1.0)

ax.axhline(0.0, color="green", linestyle="--", linewidth=0.8)
ax.set_xticks(node_norm)
ax.set_xticklabels(k_labels)
ax.set_ylabel(r"$E - E_F$ (eV)")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-9.5, 13.0)
ax.legend(frameon=False)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "band_AB.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved {OUTPUT_DIR / 'band_AB.png'}")
print(f"Saved {OUTPUT_DIR / 'band_AB.npz'}")
print(f"Saved {OUTPUT_DIR / 'band_AB_uniph.csv'}")
