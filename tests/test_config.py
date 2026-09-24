"""Config schema: defaults, strictness, the two custom validators and ``extends``."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from infinity_skyline.config import Config, load_config


def _write(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_minimal_yaml_applies_schema_defaults(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "minimal.yaml",
        "run:\n  seed: 7\ndata:\n  extensions: [PNG, .jpg]\n",
    )
    config: Config = load_config(path)

    assert config.run.seed == 7
    # everything not named in the YAML must come from the schema
    assert config.run.cache is True
    assert config.run.log_level == "INFO"
    assert config.detection.name == "sift"
    assert config.detection.repetitions == 10
    assert config.filters.lowe_ratio == pytest.approx(0.75)
    assert config.matching.name == "flann"
    assert config.graph.min_inliers == 30
    assert config.compose.seam.finder == "graphcut"
    assert config.compose.blend.bands == 5
    # the extensions validator lowercases and prefixes the dot
    assert config.data.extensions == [".png", ".jpg"]


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path, "typo.yaml", "run:\n  seed: 1\n  sed: 2\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_cross_check_true_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "cc.yaml",
        "matching:\n  name: brute_force\n  brute_force:\n    cross_check: true\n",
    )
    with pytest.raises(ValidationError) as excinfo:
        load_config(path)
    assert "cross_check" in str(excinfo.value)


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.2, 1.5])
def test_lowe_ratio_must_be_strictly_inside_zero_one(tmp_path: Path, ratio: float) -> None:
    path = _write(tmp_path, f"ratio_{ratio}.yaml", f"filters:\n  lowe_ratio: {ratio}\n")
    with pytest.raises(ValidationError):
        load_config(path)


def test_extends_merges_recursively_and_keeps_siblings(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "base.yaml",
        "matching:\n"
        "  name: brute_force\n"
        "  strategy: sequential\n"
        "  flann:\n"
        "    checks: 99\n"
        "    float_descriptors:\n"
        "      algorithm: 1\n"
        "      trees: 5\n"
        "filters:\n"
        "  lowe_ratio: 0.6\n",
    )
    child = _write(
        tmp_path,
        "child.yaml",
        "extends: base.yaml\nmatching:\n  flann:\n    checks: 7\n",
    )
    config = load_config(child)

    assert config.matching.flann.checks == 7  # the child wins on the leaf it names
    # siblings of the overridden leaf survive at every level of the tree
    assert config.matching.flann.float_descriptors == {"algorithm": 1, "trees": 5}
    assert config.matching.strategy == "sequential"
    assert config.matching.name == "brute_force"
    assert config.filters.lowe_ratio == pytest.approx(0.6)


def test_circular_extends_raises_value_error(tmp_path: Path) -> None:
    _write(tmp_path, "a.yaml", "extends: b.yaml\nrun:\n  seed: 1\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\nrun:\n  seed: 2\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "a.yaml")


def test_override_wins_over_the_file_and_keeps_siblings(tmp_path: Path) -> None:
    path = _write(tmp_path, "c.yaml", "data:\n  input_dir: a\n  max_dimension: 800\n")
    config = load_config(path, {"data": {"input_dir": "b"}})

    assert config.data.input_dir == Path("b")
    assert config.data.max_dimension == 800
