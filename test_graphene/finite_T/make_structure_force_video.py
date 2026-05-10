from pathlib import Path
import argparse
import csv
import re
import shutil
import site
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
TRAJECTORY_PATH = OUTPUT_DIR / "trajectory.xyz"
BAND_SUMMARY_PATH = OUTPUT_DIR / "bands" / "band_frames_summary.csv"
FRAME_DIR = OUTPUT_DIR / "structure_forces"
VIDEO_DIR = OUTPUT_DIR / "videos"
MP4_PATH = VIDEO_DIR / "structure_forces.mp4"
GIF_PATH = VIDEO_DIR / "structure_forces.gif"

FPS = 8
FRAME_SIZE = (1240, 1062)
FIG_DPI = 180
DEFAULT_FORCE_SCALE = 0.55
VACUUM_Z_FRACTION = 0.48
DEFAULT_VIEW_ELEV = 34
BASE_VIEW_AZIM = -58
DEFAULT_Z_ROTATION = 90


def parse_key_value(header, key):
    match = re.search(rf'{key}=("[^"]*"|\S+)', header)
    if not match:
        return None
    value = match.group(1)
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def parse_lattice(header):
    value = parse_key_value(header, "Lattice")
    if value is None:
        return np.eye(3)
    return np.array([float(x) for x in value.split()], dtype=float).reshape(3, 3)


def parse_md_step(header, fallback):
    value = parse_key_value(header, "md_step")
    return int(float(value)) if value is not None else fallback


def read_extxyz_frames(path):
    frames = []
    with path.open() as handle:
        frame_number = 0
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            n_atoms = int(line)
            header = handle.readline().strip()
            symbols = []
            positions = []
            forces = []
            for _ in range(n_atoms):
                parts = handle.readline().split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                forces.append([float(x) for x in parts[8:11]])

            frame_number += 1
            frames.append(
                {
                    "symbols": symbols,
                    "positions": np.array(positions, dtype=float),
                    "forces": np.array(forces, dtype=float),
                    "cell": parse_lattice(header),
                    "md_step": parse_md_step(header, frame_number),
                }
            )
    return frames


def read_band_selection(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "frame_id": int(row["frame_id"]),
            "trajectory_index": int(row["trajectory_index"]),
            "md_step": int(float(row["md_step"])),
        }
        for row in rows
    ]


def cell_outline_xy(cell, z):
    origin = np.array([0.0, 0.0, z])
    a = cell[0]
    b = cell[1]
    points = np.array([origin, origin + a, origin + a + b, origin + b, origin])
    return points


def cell_prism_points(cell, z_low, z_high):
    return np.vstack([cell_outline_xy(cell, z_low), cell_outline_xy(cell, z_high)])


def draw_cell_prism(ax, cell, z_low, z_high):
    bottom = cell_outline_xy(cell, z_low)
    top = cell_outline_xy(cell, z_high)
    for outline in (bottom, top):
        ax.plot(outline[:, 0], outline[:, 1], outline[:, 2], color="#222222", linewidth=1.0, alpha=0.35)
    for idx in range(4):
        ax.plot(
            [bottom[idx, 0], top[idx, 0]],
            [bottom[idx, 1], top[idx, 1]],
            [bottom[idx, 2], top[idx, 2]],
            color="#222222",
            linewidth=0.9,
            alpha=0.25,
        )


def make_equal_limits(selected_frames, force_scale):
    points = []
    for frame in selected_frames:
        positions = frame["positions"]
        forces = frame["forces"] * force_scale
        points.append(positions)
        points.append(positions + forces)
        z_center = float(np.mean(positions[:, 2]))
        z_half = max(0.5 * float(np.linalg.norm(frame["cell"][2])), 1.0) * VACUUM_Z_FRACTION
        z_low = z_center - z_half
        z_high = z_center + z_half
        points.append(cell_prism_points(frame["cell"], z_low, z_high))

    all_points = np.vstack(points)
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.48 * float(np.max(maxs - mins))
    radius = max(radius, 1.0)
    return center - radius, center + radius


def draw_bonds(ax, positions, cutoff=1.75):
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= cutoff:
                xs, ys, zs = zip(positions[i], positions[j])
                ax.plot(xs, ys, zs, color="#424242", linewidth=1.4, alpha=0.65, zorder=1)


def render_frame(frame, selection, limits, view_elev, view_azim, force_scale):
    positions = frame["positions"]
    forces = frame["forces"]
    force_norm = np.linalg.norm(forces, axis=1)
    lower, upper = limits

    fig = plt.figure(figsize=(FRAME_SIZE[0] / FIG_DPI, FRAME_SIZE[1] / FIG_DPI), dpi=FIG_DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    ax.set_position([-0.03, -0.02, 1.06, 1.04])
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    ax.view_init(elev=view_elev, azim=view_azim)

    draw_bonds(ax, positions)

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=170,
        c="#2f80ed",
        edgecolors="#111111",
        linewidths=0.7,
        depthshade=True,
        zorder=3,
    )

    linewidths = 0.9 + 1.6 * force_norm / max(float(force_norm.max()), 1e-12)
    ax.quiver(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        forces[:, 0],
        forces[:, 1],
        forces[:, 2],
        length=force_scale,
        normalize=False,
        color="#d92525",
        linewidths=linewidths,
        arrow_length_ratio=0.22,
        zorder=4,
    )

    draw_cell_prism(
        ax,
        frame["cell"],
        float(np.mean(positions[:, 2])) - 0.5 * float(np.linalg.norm(frame["cell"][2])) * VACUUM_Z_FRACTION,
        float(np.mean(positions[:, 2])) + 0.5 * float(np.linalg.norm(frame["cell"][2])) * VACUUM_Z_FRACTION,
    )

    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    try:
        ax.set_box_aspect((upper - lower).tolist(), zoom=1.12)
    except TypeError:
        ax.set_box_aspect((upper - lower).tolist())
    except AttributeError:
        pass

    ax.set_axis_off()

    frame_path = FRAME_DIR / f"structure_force_frame_{selection['frame_id']:04d}.png"
    fig.savefig(frame_path, dpi=FIG_DPI, transparent=True)
    plt.close(fig)

    with Image.open(frame_path) as image:
        if image.size != FRAME_SIZE:
            image = image.resize(FRAME_SIZE, Image.Resampling.LANCZOS)
            image.save(frame_path)
    return frame_path


def save_mp4_with_imageio(frame_paths):
    user_site = site.getusersitepackages()
    if user_site not in sys.path and Path(user_site).exists():
        sys.path.append(user_site)

    import imageio.v3 as iio

    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    iio.imwrite(MP4_PATH, frames, fps=FPS)


def save_mp4_with_ffmpeg(frame_paths):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(FPS),
            "-i",
            str(FRAME_DIR / "structure_force_frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            str(MP4_PATH),
        ],
        check=True,
    )


def save_gif(frame_paths):
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    duration_ms = int(1000 / FPS)
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render graphene structure and force-vector frames matching the band video."
    )
    parser.add_argument(
        "--z-rotation-deg",
        type=float,
        default=DEFAULT_Z_ROTATION,
        help=(
            "Extra camera rotation around the z axis in degrees, relative to "
            f"the base azimuth {BASE_VIEW_AZIM}. Default: {DEFAULT_Z_ROTATION}."
        ),
    )
    parser.add_argument(
        "--view-azim-deg",
        type=float,
        default=None,
        help="Absolute camera azimuth in degrees. If set, overrides --z-rotation-deg.",
    )
    parser.add_argument(
        "--view-elev-deg",
        type=float,
        default=DEFAULT_VIEW_ELEV,
        help=f"Camera elevation in degrees. Default: {DEFAULT_VIEW_ELEV}.",
    )
    parser.add_argument(
        "--force-scale",
        type=float,
        default=DEFAULT_FORCE_SCALE,
        help=f"Force arrow scale. Smaller values draw shorter arrows. Default: {DEFAULT_FORCE_SCALE}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    view_azim = (
        args.view_azim_deg
        if args.view_azim_deg is not None
        else BASE_VIEW_AZIM + args.z_rotation_deg
    )

    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in FRAME_DIR.glob("structure_force_frame_*.png"):
        old_path.unlink()
    for stale_path in (MP4_PATH, GIF_PATH):
        if stale_path.exists():
            stale_path.unlink()

    frames = read_extxyz_frames(TRAJECTORY_PATH)
    selection = read_band_selection(BAND_SUMMARY_PATH)
    selected_frames = [frames[item["trajectory_index"]] for item in selection]
    limits = make_equal_limits(selected_frames, args.force_scale)

    frame_paths = []
    for item, frame in zip(selection, selected_frames):
        frame_paths.append(render_frame(frame, item, limits, args.view_elev_deg, view_azim, args.force_scale))
        print(f"Saved {frame_paths[-1]}")

    try:
        save_mp4_with_imageio(frame_paths)
        print(f"Saved {MP4_PATH}")
    except Exception as imageio_exc:
        try:
            save_mp4_with_ffmpeg(frame_paths)
            print(f"Saved {MP4_PATH}")
        except Exception as ffmpeg_exc:
            save_gif(frame_paths)
            print(f"MP4 creation failed: imageio={imageio_exc}; ffmpeg={ffmpeg_exc}")
            print(f"Saved fallback {GIF_PATH}")


if __name__ == "__main__":
    main()
