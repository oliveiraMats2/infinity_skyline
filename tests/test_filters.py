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

"""Match filters: the Lowe ratio boundary and mutual consistency."""

from __future__ import annotations

from typing import Sequence

import cv2

from infinity_skyline.matching.filters import cross_check_filter, lowe_ratio_test


def _match(query_idx: int, train_idx: int, distance: float) -> cv2.DMatch:
    return cv2.DMatch(_queryIdx=query_idx, _trainIdx=train_idx, _distance=distance)


def test_lowe_ratio_keeps_only_strictly_better_neighbours() -> None:
    # every second neighbour sits at 100.0, so the ratio is the first distance / 100
    knn: list[Sequence[cv2.DMatch]] = [
        [_match(0, 0, 30.0), _match(0, 1, 100.0)],   # ratio 0.30, kept
        [_match(1, 2, 74.0), _match(1, 3, 100.0)],   # ratio 0.74, kept
        [_match(2, 4, 75.0), _match(2, 5, 100.0)],   # ratio 0.75, exactly at the threshold
        [_match(3, 6, 76.0), _match(3, 7, 100.0)],   # ratio 0.76, dropped
        [_match(4, 8, 99.0), _match(4, 9, 100.0)],   # ratio 0.99, dropped
    ]
    kept = lowe_ratio_test(knn, 0.75)

    # the comparison is strict, so the pair sitting exactly on the threshold goes away
    assert [m.queryIdx for m in kept] == [0, 1]
    assert [m.trainIdx for m in kept] == [0, 2]


def test_lowe_ratio_threshold_moves_with_the_parameter() -> None:
    knn: list[Sequence[cv2.DMatch]] = [
        [_match(0, 0, 30.0), _match(0, 1, 100.0)],
        [_match(1, 2, 74.0), _match(1, 3, 100.0)],
        [_match(2, 4, 76.0), _match(2, 5, 100.0)],
    ]
    assert [m.queryIdx for m in lowe_ratio_test(knn, 0.5)] == [0]
    assert [m.queryIdx for m in lowe_ratio_test(knn, 0.9)] == [0, 1, 2]


def test_lowe_ratio_discards_short_neighbourhoods() -> None:
    knn: list[Sequence[cv2.DMatch]] = [
        [_match(0, 0, 1.0)],                          # a single neighbour has no ratio
        [],                                           # and neither has an empty one
        [_match(2, 2, 10.0), _match(2, 3, 100.0)],
    ]
    kept = lowe_ratio_test(knn, 0.75)
    assert [m.queryIdx for m in kept] == [2]


def test_lowe_ratio_on_empty_input() -> None:
    assert lowe_ratio_test([], 0.75) == []


def test_cross_check_keeps_only_the_mutual_match() -> None:
    ab = [_match(0, 5, 1.0), _match(1, 6, 2.0), _match(2, 7, 3.0)]
    ba = [_match(5, 0, 1.0), _match(6, 9, 2.0), _match(7, 3, 3.0)]

    kept = cross_check_filter(ab, ba)

    assert len(kept) == 1
    assert (kept[0].queryIdx, kept[0].trainIdx) == (0, 5)


def test_cross_check_on_empty_input() -> None:
    assert cross_check_filter([], []) == []
    assert cross_check_filter([_match(0, 5, 1.0)], []) == []
