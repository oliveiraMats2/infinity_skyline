"""infinity-skyline: feature detection, matching, geometry and panorama composition.

Everything public is re-exported here, so callers write

    from infinity_skyline import detect_image, estimate_homography, compose_panorama

instead of reaching into the submodules. The submodules keep their relative
imports; this is the flat surface for whoever uses the package.
"""

from __future__ import annotations

__version__ = "0.1.0"

# config and logging first: every other submodule imports from these two.
from .config import Config, dump_config, load_config
from .logging_setup import get_logger, setup_logging

from .io import (
    KeypointRecord,
    MatchRecord,
    array_to_keypoints,
    keypoints_to_array,
    list_images,
    load_image,
    load_keypoints,
    load_matches,
    pair_stem,
    resize_to_max,
    save_image,
    save_keypoints,
    save_matches,
)

from .detection import (
    BINARY_DETECTORS,
    DETECTORS,
    create_detector,
    detect_directory,
    detect_image,
    is_binary,
    norm_type,
)
from .matching import (
    build_pairs,
    create_matcher,
    cross_check_filter,
    lowe_ratio_test,
    match_all,
    match_pair,
    max_distance_filter,
)
from .geometry import (
    build_graph,
    connectivity_matrix,
    estimate_all,
    estimate_homography,
    infer_order,
    largest_component,
    rejected_images,
    reprojection_errors,
)
from .compose import (
    ComposeResult,
    chain_homographies,
    compose_panorama,
    pick_reference,
)
from .stitcher import stitch_with_opencv
from .evaluate import (
    distortion_score,
    inlier_ratio,
    reprojection_rmse,
    run_benchmark,
    seam_line_continuity,
    spatial_dispersion,
    summarize,
)
from .visualize import (
    draw_before_after,
    draw_graph,
    draw_keypoints,
    draw_matches,
    draw_progressive,
    plot_benchmark,
    save_graph_figures,
    save_keypoint_figures,
    save_match_figures,
    save_panorama_figures,
)
from .tools import convert_directory

__all__ = [
    "__version__",
    # config and logging
    "Config",
    "load_config",
    "dump_config",
    "get_logger",
    "setup_logging",
    # io
    "KeypointRecord",
    "MatchRecord",
    "array_to_keypoints",
    "keypoints_to_array",
    "list_images",
    "load_image",
    "load_keypoints",
    "load_matches",
    "pair_stem",
    "resize_to_max",
    "save_image",
    "save_keypoints",
    "save_matches",
    # detection
    "BINARY_DETECTORS",
    "DETECTORS",
    "create_detector",
    "detect_directory",
    "detect_image",
    "is_binary",
    "norm_type",
    # matching
    "build_pairs",
    "create_matcher",
    "cross_check_filter",
    "lowe_ratio_test",
    "match_all",
    "match_pair",
    "max_distance_filter",
    # geometry
    "build_graph",
    "connectivity_matrix",
    "estimate_all",
    "estimate_homography",
    "infer_order",
    "largest_component",
    "rejected_images",
    "reprojection_errors",
    # compose
    "ComposeResult",
    "chain_homographies",
    "compose_panorama",
    "pick_reference",
    "stitch_with_opencv",
    # evaluate
    "distortion_score",
    "inlier_ratio",
    "reprojection_rmse",
    "run_benchmark",
    "seam_line_continuity",
    "spatial_dispersion",
    "summarize",
    # visualize
    "draw_before_after",
    "draw_graph",
    "draw_keypoints",
    "draw_matches",
    "draw_progressive",
    "plot_benchmark",
    "save_graph_figures",
    "save_keypoint_figures",
    "save_match_figures",
    "save_panorama_figures",
    # tools
    "convert_directory",
]
