from pathlib import Path
import os
import site
import sys


UNIPH_ROOT = Path(__file__).resolve().parents[2]
if str(UNIPH_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIPH_ROOT))
USER_SITE = site.getusersitepackages()
if USER_SITE not in sys.path and Path(USER_SITE).exists():
    sys.path.append(USER_SITE)
os.environ.setdefault("MPLCONFIGDIR", str(UNIPH_ROOT / ".matplotlib"))

import ase.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils.utils_calculator import load_graphene_model, predict_tb


# Band settings. The MD trajectory has 100 frames; this samples every 5th frame.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BAND_EVERY = 5
KPTS = 40
Y_LIMIT = (-9.0, 12.0)

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory.xyz"
BAND_DIR = OUTPUT_DIR / "bands"
BAND_DIR.mkdir(parents=True, exist_ok=True)


def build_band_path(atoms):
    bandpath = atoms.cell.bandpath("MGKM", npoints=KPTS)
    k_vecs = np.array(bandpath.kpts)
    k_dist, node_dist, labels = bandpath.get_linear_kpoint_axis()
    labels = [r"$\Gamma$" if label == "G" else label for label in labels]
    return k_vecs, k_dist, node_dist, labels


def compute_frame_bands(atoms, context):
    graph, hopping_mats = predict_tb(atoms, context)
    edge_src = graph.edge_index[0].detach().cpu().numpy()
    edge_dst = graph.edge_index[1].detach().cpu().numpy()
    edge_shift = graph.edge_shift.detach().cpu().numpy()

    k_vecs, k_dist, node_dist, labels = build_band_path(atoms)
    hoppings = []
    for hop, src, dst, shift in zip(hopping_mats, edge_src, edge_dst, edge_shift):
        hop_scalar = np.asarray(hop).reshape(-1)[0]
        hop_scalar = complex(np.real_if_close(hop_scalar))
        hoppings.append((hop_scalar, int(src), int(dst), np.array(shift, dtype=float)))

    n_orb = len(atoms)
    energies = []
    for k_frac in k_vecs:
        ham = np.zeros((n_orb, n_orb), dtype=np.complex128)
        for hop, src, dst, shift in hoppings:
            phase = np.exp(2j * np.pi * np.dot(k_frac, shift))
            ham[src, dst] += hop * phase
        ham = 0.5 * (ham + ham.conj().T)
        energies.append(np.linalg.eigvalsh(ham))

    return np.array(energies), k_dist, node_dist, labels


def save_band_plot(frame_id, md_step, energies, k_dist, node_dist, labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    for band_idx in range(energies.shape[1]):
        ax.plot(k_dist, energies[:, band_idx], color="tab:orange", linewidth=1.6)

    for x_val in node_dist:
        ax.axvline(x_val, color="black", linestyle=":", linewidth=1.0)

    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xticks(node_dist)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy (eV)")
    ax.set_xlim(0.0, float(np.max(k_dist)))
    ax.set_ylim(*Y_LIMIT)
    ax.grid(False)
    fig.tight_layout()
    png_path = BAND_DIR / f"band_frame_{frame_id:04d}.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return png_path


for old_path in BAND_DIR.glob("band_frame_*.png"):
    old_path.unlink()
for old_path in BAND_DIR.glob("band_frame_*.npz"):
    old_path.unlink()

frames = ase.io.read(TRAJECTORY_PATH, index=":", format="extxyz")
selected = [
    (idx, atoms)
    for idx, atoms in enumerate(frames)
    if int(atoms.info.get("md_step", idx + 1)) % BAND_EVERY == 0
]
context = load_graphene_model(device=DEVICE)

summary_rows = []
for frame_id, (trajectory_index, atoms) in enumerate(selected):
    md_step = atoms.info.get("md_step", trajectory_index + 1)
    energies, k_dist, node_dist, labels = compute_frame_bands(atoms, context)
    np.savez(
        BAND_DIR / f"band_frame_{frame_id:04d}.npz",
        trajectory_index=trajectory_index,
        md_step=md_step,
        k_dist=k_dist,
        node_dist=node_dist,
        labels=np.array(labels),
        energies=energies,
    )
    png_path = save_band_plot(frame_id, md_step, energies, k_dist, node_dist, labels)
    summary_rows.append(
        {
            "frame_id": frame_id,
            "trajectory_index": trajectory_index,
            "md_step": md_step,
            "png_path": str(png_path),
            "energy_min_eV": float(np.min(energies)),
            "energy_max_eV": float(np.max(energies)),
        }
    )
    print(f"Saved {png_path}")

pd.DataFrame(summary_rows).to_csv(BAND_DIR / "band_frames_summary.csv", index=False)
print(f"Saved {BAND_DIR / 'band_frames_summary.csv'}")
