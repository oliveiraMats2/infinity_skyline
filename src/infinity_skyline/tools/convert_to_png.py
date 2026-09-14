"""Decode a scene directory (Canon RAW included) into lossless PNG.

The pipeline reads one scene at a time and times detection over many repetitions,
so paying the rawpy decode on every run would dominate the measurements. This step
runs once and leaves `data/processed/<scene>/` ready for the rest of the CLI.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm.auto import tqdm

from ..io import list_images, load_image
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Everything worth scanning for: camera RAW plus the usual already decoded formats.
SOURCE_EXTENSIONS: tuple[str, ...] = (
    ".cr2",
    ".nef",
    ".arw",
    ".dng",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".png",
    ".heic",
)

#: PNG is lossless, so maximum effort only costs encode time, never image quality.
PNG_COMPRESSION: int = 9


def convert_directory(
    source: Path,
    destination: Path,
    *,
    max_dimension: int | None,
    overwrite: bool = False,
) -> list[Path]:
    """Convert every supported image in ``source`` to PNG under ``destination``.

    Returns the list of PNG paths that now exist, skipped files included.
    """
    sources = list_images(source, SOURCE_EXTENSIONS)
    if not sources:
        logger.warning("no convertible image found in %s", source)
        return []

    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path in tqdm(sources, desc="convert", unit="img"):
        target = destination / f"{path.stem}.png"
        if target.exists() and not overwrite:
            logger.debug("skipping %s: %s already exists", path.name, target)
            written.append(target)
            continue
        image = load_image(path, grayscale=False, max_dimension=max_dimension)
        # io.save_image takes no encoder flags, so cv2.imwrite is called directly here.
        if not cv2.imwrite(str(target), image, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
            raise OSError(f"cv2.imwrite failed for {target}")
        logger.debug("wrote %s %s", target, image.shape)
        written.append(target)

    logger.info("converted %d image(s) from %s to %s", len(written), source, destination)
    return written
