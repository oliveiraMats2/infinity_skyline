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

"""Pure metric functions for the benchmark and for stage 6.5.

Every function here is side effect free: no I/O, no logging, no global state, so
each one can be exercised in isolation. Degenerate inputs (empty arrays, a mask
with a single region, a singular homography) return ``nan`` instead of raising.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..io import MatchRecord

_NAN: float = float("nan")

#: Half width, in pixels, of the band considered "on the seam".
_SEAM_BAND: int = 1
#: Pixels of clearance between the seam band and the control ring.
_CONTROL_GAP: int = 3
#: Outer radius of the control ring around the seam.
_CONTROL_RADIUS: int = 7


def _kernel(radius: int) -> np.ndarray:
    """Square structuring element of the given radius."""
    return np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)


def spatial_dispersion(keypoints: np.ndarray, shape: tuple[int, int]) -> float:
    """Spread of the keypoints over the frame, as a single scalar in [0, 0.5].

    Convention: the x and y columns are normalized by width and height, and the
    returned value is the mean of their standard deviations. 0.0 means every
    keypoint sits on the same spot, about 0.29 means uniform coverage of the whole
    frame (a uniform distribution on [0, 1] has std 1/sqrt(12)), and 0.5 is the
    degenerate extreme of half the points on each opposite border. Low values mean
    the keypoints are clustered in one region, which is bad for homography
    estimation because the fit is unconstrained everywhere else.
    """
    if keypoints is None or keypoints.size == 0 or keypoints.shape[0] < 2:
        return _NAN
    height, width = float(shape[0]), float(shape[1])
    if height <= 0.0 or width <= 0.0:
        return _NAN
    x = keypoints[:, 0].astype(np.float64) / width
    y = keypoints[:, 1].astype(np.float64) / height
    return float((x.std() + y.std()) / 2.0)


def inlier_ratio(record: MatchRecord) -> float:
    """Inliers over filtered matches, 0.0 when RANSAC never ran on this pair."""
    if record.inlier_mask is None or record.n_filtered == 0:
        return 0.0
    return float(record.inlier_ratio)


def reprojection_rmse(record: MatchRecord) -> float:
    """RMSE stored by ``geometry.homography``, nan when there is no estimate."""
    value = record.meta.get("reprojection_rmse")
    if value is None or record.homography is None:
        return _NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def seam_line_continuity(panorama: np.ndarray, seam_mask: np.ndarray) -> float:
    """Stage 6.5: gradient energy on the seam over the energy right beside it.

    The seam line is the boundary between regions of ``seam_mask`` (the per pixel
    index of the source image), found as the non zero response of a Laplacian on
    the label image. Sobel energy on the luminance is averaged over that line and
    divided by the average over a control ring a few pixels away, on the same
    panorama. Interpretation: ~1.0 means the seam is invisible (lines cross it
    continuously), values well above 1 mean a visible cut (the stitch introduced an
    edge of its own), values well below 1 mean the seam was routed through a flat,
    textureless region. Returns nan for degenerate inputs: empty arrays, mismatched
    shapes, a uniform mask (no seam) or an empty control ring.
    """
    if panorama is None or seam_mask is None or panorama.size == 0 or seam_mask.size == 0:
        return _NAN
    if panorama.shape[:2] != seam_mask.shape[:2]:
        return _NAN

    gray = panorama if panorama.ndim == 2 else cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    energy = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )

    labels = seam_mask.astype(np.float32)
    inside = (seam_mask > 0).astype(np.uint8)  # label 0 is the empty canvas border
    line = (np.abs(cv2.Laplacian(labels, cv2.CV_32F, ksize=3)) > 0).astype(np.uint8) & inside
    if not line.any():
        return _NAN

    on_seam = cv2.dilate(line, _kernel(_SEAM_BAND)) & inside
    near = cv2.dilate(line, _kernel(_CONTROL_RADIUS)) & inside
    control = near & (1 - cv2.dilate(line, _kernel(_CONTROL_GAP)))

    seam_pixels = energy[on_seam.astype(bool)]
    control_pixels = energy[control.astype(bool)]
    if seam_pixels.size == 0 or control_pixels.size == 0:
        return _NAN
    reference = float(control_pixels.mean())
    if reference <= 0.0:
        return _NAN
    return float(seam_pixels.mean() / reference)


def distortion_score(homography: np.ndarray) -> float:
    """Stage 6.5: how far the 2x2 block of H is from a similarity transform.

    Convention: singular values s_max and s_min of ``H[:2, :2]``, returned as
    ``s_max / s_min - 1``. A similarity (rotation plus uniform scale) has equal
    singular values, so the score is exactly 0.0; the value grows with anisotropic
    stretching and shear, which is what makes a warped image look distorted. It
    ignores translation and the perspective row on purpose: those move the image
    without deforming it locally. Returns nan for a missing, malformed or singular
    homography.
    """
    if homography is None:
        return _NAN
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        return _NAN
    try:
        singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
    except np.linalg.LinAlgError:
        return _NAN
    if singular.size < 2 or singular[-1] <= np.finfo(np.float64).eps:
        return _NAN
    return float(singular[0] / singular[-1] - 1.0)
