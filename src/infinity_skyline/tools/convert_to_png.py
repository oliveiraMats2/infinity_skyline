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
from tqdm.auto import tqdm

from ..io import RAW_EXTENSIONS, list_images, load_image
from ..logging_setup import get_logger

logger = get_logger(__name__)

#: Everything worth scanning for: every RAW the loader handles, plus the usual
#: already decoded formats. Derived from io.RAW_EXTENSIONS so a new RAW format
#: registered there is picked up here instead of being silently skipped.
#: ".heic" is absent on purpose: cv2.imread returns None for it, and decoding it
#: would need the pillow_heif plugin, a dependency this project does not carry.
SOURCE_EXTENSIONS: tuple[str, ...] = tuple(sorted(RAW_EXTENSIONS)) + (
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".png",
)

#: PNG is lossless, so maximum effort only costs encode time, never image quality.
PNG_COMPRESSION: int = 9

# EXIF orientation is deliberately not handled here. cv2.imread with IMREAD_COLOR
# (what io.load_image uses) already applies the orientation tag itself for every
# format in SOURCE_EXTENSIONS, verified on OpenCV 4.14: a 40x20 JPEG or PNG tagged
# orientation 6 comes back as 20x40, upright. Rotating again on top of that would
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
            image = load_image(path, grayscale=False, max_dimension=max_dimension)
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
