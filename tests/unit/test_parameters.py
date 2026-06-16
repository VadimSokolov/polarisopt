from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from polarisopt.parameters import (
    Parameter,
    ParameterSpace,
    ParameterType,
    inject_values,
    load_parameter_file,
)
from polarisopt.parameters.space import parameter_space_from_records


def test_parameter_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="must exceed low"):
        Parameter(name="x", file="a.json", low=1.0, high=0.5)


def test_parameter_int_clip_rounds() -> None:
    p = Parameter(name="x", file="a.json", low=0, high=10, ptype=ParameterType.INT)
    assert p.clip(3.7) == 4
    assert p.clip(-5) == 0
    assert p.clip(99) == 10


def test_parameter_space_duplicate_name_rejected() -> None:
    a = Parameter(name="x", file="a.json", low=0, high=1)
    b = Parameter(name="x", file="b.json", low=0, high=1)
    with pytest.raises(ValueError, match="Duplicate"):
        ParameterSpace.from_iterable([a, b])


def test_parameter_space_clip_batch() -> None:
    space = ParameterSpace.from_iterable(
        [
            Parameter("a", "a.json", 0.0, 1.0),
            Parameter("b", "b.json", 10, 20, ParameterType.INT),
        ]
    )
    out = space.clip(np.array([[0.5, 15.4], [-1.0, 99.0]]))
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(0.5)
    assert out[0, 1] == 15  # rounded int
    assert out[1, 0] == 0.0  # clipped
    assert out[1, 1] == 20  # clipped


def test_parameter_space_values_dict() -> None:
    space = ParameterSpace.from_iterable(
        [Parameter("a", "a.json", 0, 1), Parameter("b", "b.json", 0, 1)]
    )
    out = space.values_dict(np.array([0.3, 0.9]))
    assert out == {"a": pytest.approx(0.3), "b": pytest.approx(0.9)}


def test_parameter_space_bounds_shape() -> None:
    space = ParameterSpace.from_iterable(
        [Parameter("a", "a.json", 0.0, 1.0), Parameter("b", "b.json", -5.0, 5.0)]
    )
    assert space.bounds.shape == (2, 2)
    assert space.ndim == 2
    assert space.names == ("a", "b")


def test_parameter_space_from_records() -> None:
    records = [
        {"name": "a", "file": "x.json", "min": 0.0, "max": 1.0, "type": "float"},
        {"name": "b", "file": "x.json", "min": 0, "max": 10, "type": "int"},
    ]
    space = parameter_space_from_records(records)
    assert space.ndim == 2
    assert space.parameters[1].ptype is ParameterType.INT


def test_load_parameter_file_yaml(tmp_path) -> None:
    p = tmp_path / "params.yaml"
    p.write_text(
        yaml.safe_dump(
            [
                {"name": "a", "file": "x.json", "min": 0.0, "max": 1.0},
                {"name": "b", "file": "y.json", "min": 1, "max": 5, "type": "int"},
            ]
        )
    )
    space = load_parameter_file(p)
    assert space.ndim == 2
    assert space.parameters[1].ptype is ParameterType.INT


def test_inject_values_two_level(tmp_path) -> None:
    target = tmp_path / "DestinationChoice.json"
    target.write_text(
        json.dumps(
            {
                "Hard constraints": {"trip_threshold": 0.0, "min_distance": 0.0},
                "Soft constraints": {"weight_alpha": 0.0},
            }
        )
    )
    space = ParameterSpace.from_iterable(
        [
            Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0),
            Parameter("weight_alpha", "DestinationChoice.json", 0.0, 5.0),
        ]
    )
    missing = inject_values(np.array([0.42, 2.5]), space, tmp_path)
    assert missing == {}
    data = json.loads(target.read_text())
    assert data["Hard constraints"]["trip_threshold"] == pytest.approx(0.42)
    assert data["Soft constraints"]["weight_alpha"] == pytest.approx(2.5)
    # Untouched key preserved
    assert data["Hard constraints"]["min_distance"] == 0.0


def test_inject_values_reports_missing(tmp_path) -> None:
    target = tmp_path / "DestinationChoice.json"
    target.write_text(json.dumps({"Hard constraints": {"trip_threshold": 0.0}}))
    space = ParameterSpace.from_iterable(
        [
            Parameter("trip_threshold", "DestinationChoice.json", 0.0, 1.0),
            Parameter("does_not_exist", "DestinationChoice.json", 0.0, 1.0),
        ]
    )
    missing = inject_values(np.array([0.5, 0.5]), space, tmp_path)
    assert missing == {"DestinationChoice.json": ["does_not_exist"]}


def test_inject_values_skips_missing_file(tmp_path) -> None:
    space = ParameterSpace.from_iterable(
        [Parameter("trip_threshold", "DoesNotExist.json", 0.0, 1.0)]
    )
    missing = inject_values(np.array([0.5]), space, tmp_path)
    assert missing == {"DoesNotExist.json": ["trip_threshold"]}
