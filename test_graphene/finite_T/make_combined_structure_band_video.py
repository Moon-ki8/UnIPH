from pathlib import Path
import site
import sys

from PIL import Image, ImageDraw, ImageFont
import numpy as np


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs"
STRUCTURE_DIR = OUTPUT_DIR / "structure_forces"
BAND_DIR = OUTPUT_DIR / "bands"
COMBINED_DIR = OUTPUT_DIR / "combined_structure_band"
VIDEO_DIR = OUTPUT_DIR / "videos"
MP4_PATH = VIDEO_DIR / "combined_structure_band.mp4"
GIF_PATH = VIDEO_DIR / "combined_structure_band.gif"

FPS = 8
PANEL_SIZE = (1240, 1062)
HEADER_HEIGHT = 122
CANVAS_SIZE = (PANEL_SIZE[0] * 2, HEADER_HEIGHT + PANEL_SIZE[1])
BACKGROUND = (255, 255, 255, 255)
TITLE_COLOR = (20, 20, 20)


def load_font(size):
    for path in (
        "/usr/share/fonts/truetype/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def load_paths(directory, pattern):
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No frames found under {directory} matching {pattern}")
    return paths


def fit_panel(path):
    image = Image.open(path).convert("RGBA")
    if image.size != PANEL_SIZE:
        image = image.resize(PANEL_SIZE, Image.Resampling.LANCZOS)
    return image


def render_combined_frames():
    structure_paths = load_paths(STRUCTURE_DIR, "structure_force_frame_*.png")
    band_paths = load_paths(BAND_DIR, "band_frame_*.png")
    if len(structure_paths) != len(band_paths):
        raise ValueError(
            f"Frame count mismatch: {len(structure_paths)} structure frames, {len(band_paths)} band frames"
        )

    COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for old_path in COMBINED_DIR.glob("combined_frame_*.png"):
        old_path.unlink()

    font = load_font(68)
    frame_paths = []
    for idx, (structure_path, band_path) in enumerate(zip(structure_paths, band_paths)):
        canvas = Image.new("RGBA", CANVAS_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        structure = fit_panel(structure_path)
        band = fit_panel(band_path)
        canvas.alpha_composite(structure, (0, HEADER_HEIGHT))
        canvas.alpha_composite(band, (PANEL_SIZE[0], HEADER_HEIGHT))
        draw.text((52, 18), "Molecular dynamics", fill=TITLE_COLOR, font=font)
        draw.text((PANEL_SIZE[0] + 52, 18), "Band structure", fill=TITLE_COLOR, font=font)

        frame_path = COMBINED_DIR / f"combined_frame_{idx:04d}.png"
        canvas.save(frame_path)
        frame_paths.append(frame_path)
        print(f"Saved {frame_path}")
    return frame_paths


def save_mp4(frame_paths):
    user_site = site.getusersitepackages()
    if user_site not in sys.path and Path(user_site).exists():
        sys.path.append(user_site)

    import imageio.v3 as iio

    frames = [np.asarray(Image.open(path).convert("RGB")) for path in frame_paths]
    iio.imwrite(MP4_PATH, frames, fps=FPS)


def save_gif(frame_paths):
    images = [Image.open(path).convert("RGB").convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    duration_ms = int(1000 / FPS)
    images[0].save(
        GIF_PATH,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for stale_path in (MP4_PATH, GIF_PATH):
        if stale_path.exists():
            stale_path.unlink()

    frame_paths = render_combined_frames()
    try:
        save_mp4(frame_paths)
        print(f"Saved {MP4_PATH}")
    except Exception as exc:
        print(f"MP4 creation failed: {exc}")

    save_gif(frame_paths)
    print(f"Saved {GIF_PATH}")


if __name__ == "__main__":
    main()
