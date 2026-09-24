# ╔══════════════════════════════════════════════════════════════════════════════════════╗
# ║  ⠀⠀⠀⠀⣠⠶⡒⠒⢬⡲⣮⠂⣆⣀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣦⣤⡀⠀⠀⠀⠀   MATEUS OLIVEIRA                        ║
# ║  ⠀⠀⠀⣀⣥⠠⣿⠆⠐⣻⣾⣿⣿⢷⡄⠀⠀⠀⠀⢠⡿⠋⠉⠉⠙⢿⡄⠀⠀⠀   m203656@dac.unicamp.edu.br             ║
# ║  ⠀⠀⢘⡵⢋⠄⡙⠒⣤⣄⣉⠙⣿⣗⠑⡄⠀⠀⠀⠘⡇⠀⠀⠀⠀⠈⡇⠀⠀⠀   UNICAMP - Universidade Estadual de     ║
# ║  ⠀⣴⢿⡜⢡⡞⢀⢼⣿⣿⣿⣿⣿⣿⠟⣂⠀⠀⢀⣀⠱⡀⠀⠀⠀⢰⠁⠀⠀⠀               Campinas                     ║
# ║  ⠰⢫⢟⡇⢸⡇⢸⢾⣿⣿⣿⣿⣿⣿⡷⠰⠀⢰⡏⠀⠀⢡⠀⠀⢠⠃⠀⠀⠀⠀   IC - Institute of Computing            ║
# ║  ⢰⠁⣿⢣⣿⠇⢀⣿⣿⡿⠿⠤⣭⣥⣶⡆⠀⠸⣷⣤⣠⡾⠀⢀⡇⠀⠀⠀⠀⠀   Computer Science Department              ║
# ║  ⡞⣰⣧⠟⡝⢸⢸⣿⣥⠖⣴⡆⣤⣬⠉⠀⠀⠀⠈⠉⠉⠀⠀⢸⣇⠀⠀⠀⠀⠀   github.com/oliveiraMats2              ║
# ║  ⠀⡿⡟⢸⡇⠸⡄⢹⣿⢸⣿⣇⡏⠟⣰⣄⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠁⠀⠀⠀   linkedin.com/in/mateus-eng            ║
# ║  ⠀⠇⣧⠘⡇⠦⣹⣸⣿⡇⡿⡿⣡⣼⣿⣿⣷⣦⣄⡀⠀⠀⣸⣿⣿⠄⠻⢷⣦⠀                                            ║
# ║  ⠀⢀⠘⣇⢹⡸⣿⣿⣿⢹⢃⣠⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠑⠋⠉⠀⠀⠈⣿⣧   UNICAMP · IC · 2026                    ║
# ║  ⠀⢸⣿⡌⠘⢷⣿⣿⡏⢀⣾⣿⣿⣿⣿⣿⣿⢻⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣿⡿                                            ║
# ║  ⠀⠈⣿⣿⣦⡌⢿⠏⣰⣿⣿⣿⣿⣿⣿⡿⡏⣼⣿⣿⣿⡇⣄⠀⠀⠀⢀⣼⣿⠇                                            ║
# ║  ⠀⠀⠹⣿⣿⢻⡀⣼⣿⣿⢻⣿⣿⣿⣿⡇⡇⢻⣿⣿⣿⡇⣿⣿⣶⣿⣿⠟⠁⠀                                            ║
# ║  ⠀⠀⠀⢻⣿⣦⡓⢿⣿⣿⡆⣿⣿⣿⣿⢃⣶⡸⣿⣿⣿⡇⠀⠉⠉⠁⠀⠀⠀⠀                                            ║
# ║  ⠀⠀⠀⠈⣿⣿⣿⡆⠀⠀⠀⣿⣿⣿⡟⣼⡿⠁⢹⣿⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════╝

"""Decode a scene directory (Canon RAW included) into lossless PNG.

The pipeline reads one scene at a time and times detection over many repetitions,
so paying the rawpy decode on every run would dominate the measurements. This step
runs once and leaves `data/processed/<scene>/` ready for the rest of the CLI.

The scan is recursive and the source subfolder tree is mirrored under the
destination, so this works on any image folder, not only on a flat scene.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from ..io import HEIC_EXTENSIONS, RAW_EXTENSIONS, list_images, load_image
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Everything worth scanning for: every RAW the loader handles, plus the usual
#: already decoded formats. Derived from io.RAW_EXTENSIONS so a new RAW format
#: registered there is picked up here instead of being silently skipped.
SOURCE_EXTENSIONS: tuple[str, ...] = tuple(sorted(RAW_EXTENSIONS | HEIC_EXTENSIONS)) + (
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".png",
)

#: PNG is lossless, so maximum effort only costs encode time, never image quality.
PNG_COMPRESSION: int = 9

#: Largest PNG ever written, (long side, short side) so it holds in either
#: orientation: 1920x1080 landscape, 1080x1920 portrait. Aspect ratio is kept.
MAX_SIZE: tuple[int, int] = (1920, 1080)


def fit_max_size(image: np.ndarray, max_size: tuple[int, int] = MAX_SIZE) -> np.ndarray:
    """Downscale so the long side fits ``max_size[0]`` and the short one ``max_size[1]``."""
    height, width = image.shape[:2]
    scale = min(1.0, max_size[0] / max(height, width), max_size[1] / min(height, width))
    if scale == 1.0:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

# EXIF orientation is deliberately not handled here. cv2.imread with IMREAD_COLOR
# (what io.load_image uses) already applies the orientation tag itself for every
# format it decodes, verified on OpenCV 4.14: a 40x20 JPEG or PNG tagged
# orientation 6 comes back as 20x40, upright. HEIC goes through Pillow instead,
# and io._read_heic applies the tag there. Rotating again on top of that would
# turn every portrait phone shot sideways, which is the bug this would be fixing.
# Only a decoder reading with IMREAD_UNCHANGED, or an OpenCV older than 3.4, would
# need a manual tag-to-cv2.rotate mapping here.


def convert_directory(
    source: Path,
    destination: Path,
    *,
    max_dimension: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every supported image under ``source`` to PNG under ``destination``.

    Every PNG fits :data:`MAX_SIZE`; ``max_dimension`` can only cap the long side
    further, never lift that limit.

    The scan is recursive and the tree is mirrored: ``source/a/b/x.CR2`` becomes
    ``destination/a/b/x.png``. One unreadable file is logged and counted, never
    fatal, so a single bad codec does not cost the rest of the folder.

    Returns the list of PNG paths that now exist, skipped files included.
    """
    sources = list_images(source, SOURCE_EXTENSIONS, recursive=True)
    if not sources:
        logger.warning("no convertible image found in %s", source)
        return []

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    claimed: dict[Path, Path] = {}  # target PNG -> the source that got there first
    failed = 0
    for path in tqdm(sources, desc="convert", unit="img"):
        target = (destination / path.relative_to(source)).with_suffix(".png")
        if target in claimed:
            # e.g. a.jpg and a.CR2 side by side: both want a.png. Silently
            # overwriting would lose one of the two, so the second one is dropped.
            logger.error(
                "stem collision: %s and %s both map to %s, skipping the latter",
                claimed[target],
                path,
                target,
            )
            failed += 1
            continue
        claimed[target] = path

        if target.exists() and not overwrite:
            logger.debug("skipping %s: %s already exists", path.name, target)
            written.append(target)
            continue
        try:
            image = fit_max_size(load_image(path, grayscale=False, max_dimension=max_dimension))
            target.parent.mkdir(parents=True, exist_ok=True)
            # io.save_image takes no encoder flags, so cv2.imwrite is called directly here.
            if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
                raise OSError(f"cv2.imwrite failed for {target}")
        except Exception as error:  # any codec, on any file, may throw anything
            logger.error("failed to convert %s: %s", path, error)
            failed += 1
            continue
        logger.debug("wrote %s %s", target, image.shape)
        written.append(target)

    logger.info(
        "%d PNG(s) from %s in %s, %d file(s) failed", len(written), source, destination, failed
    )
    return written
