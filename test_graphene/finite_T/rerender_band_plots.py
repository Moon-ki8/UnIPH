from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BAND_DIR = HERE / "outputs" / "bands"
Y_LIMIT = (-9.0, 12.0)


def render_band_plot(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    frame_id = int(npz_path.stem.split("_")[-1])
    energies = data["energies"]
    k_dist = data["k_dist"]
    node_dist = data["node_dist"]
    labels = data["labels"].tolist()

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
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
    fig.savefig(png_path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return png_path


def main():
    paths = sorted(BAND_DIR.glob("band_frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No band npz files found under {BAND_DIR}")
    for path in paths:
        print(f"Saved {render_band_plot(path)}")


if __name__ == "__main__":
    main()
