from pathlib import Path
import site
import sys

from PIL import Image


HERE = Path(__file__).resolve().parent
BAND_DIR = HERE / "outputs" / "bands"
VIDEO_DIR = HERE / "outputs" / "videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATTERN = "band_frame_*.png"
MP4_PATH = VIDEO_DIR / "band_structure.mp4"
GIF_PATH = VIDEO_DIR / "band_structure.gif"
FPS = 8


def load_png_paths():
    paths = sorted(BAND_DIR.glob(PNG_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found under {BAND_DIR}")
    return paths


def save_mp4_with_imageio(paths):
    user_site = site.getusersitepackages()
    if user_site not in sys.path and Path(user_site).exists():
        sys.path.append(user_site)

    import imageio.v3 as iio

    frames = [iio.imread(path) for path in paths]
    iio.imwrite(MP4_PATH, frames, fps=FPS)
    return MP4_PATH


def save_gif_with_pillow(paths):
    frames = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths]
    duration_ms = int(1000 / FPS)
    frames[0].save(
        GIF_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return GIF_PATH


paths = load_png_paths()
for stale_path in (MP4_PATH, GIF_PATH):
    if stale_path.exists():
        stale_path.unlink()
try:
    video_path = save_mp4_with_imageio(paths)
    print(f"Saved {video_path}")
except Exception as exc:
    print(f"MP4 creation failed: {exc}")
    video_path = save_gif_with_pillow(paths)
    print(f"Saved fallback {video_path}")
