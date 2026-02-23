import numpy as np

from blop.plans.utils import InferredReadable, get_route_index, route_suggestions
from blop.protocols import ID_KEY

# InferredReadable tests


def test_inferred_readable_scalar_number():
    r = InferredReadable("x", 1.5)
    assert r.name == "x"
    assert r.parent is None
    read = r.read()
    assert read["x"]["value"] == 1.5
    assert "timestamp" in read["x"]
    assert r.describe()["x"]["dtype"] == "number"
    assert r.hints["fields"] == ["x"]


def test_inferred_readable_scalar_string():
    r = InferredReadable("ids", ["0"])
    assert r.read()["ids"]["value"] == "0"
    assert r.describe()["ids"]["dtype"] == "string"


def test_inferred_readable_array():
    r = InferredReadable("arr", [0.0, 0.1])
    assert r.read()["arr"]["value"] == [0.0, 0.1]
    assert r.describe()["arr"]["dtype"] == "array"


def test_inferred_readable_update():
    r = InferredReadable("x", 1.5)
    r.update(2.0)
    assert r.read()["x"]["value"] == 2.0

    r2 = InferredReadable("arr", [0.0, 0.1])
    r2.update(np.array([1.0, 2.0]))
    assert list(r2.read()["arr"]["value"]) == [1.0, 2.0]


# get_route_index tests


def test_get_route_index_two_points_no_start():
    points = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = get_route_index(points)
    assert set(result) == {0, 1}


def test_get_route_index_multiple_points_no_start():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    result = get_route_index(points)
    assert set(result) == {0, 1, 2}


def test_get_route_index_with_starting_point():
    points = np.array([[1.0, 0.0], [2.0, 0.0]])
    start = np.array([0.0, 0.0])
    result = get_route_index(points, starting_point=start)
    assert set(result) == {0, 1}


# route_suggestions tests


def test_route_suggestions_single_returns_unchanged():
    suggestions = [{"x": 1.0, "y": 2.0, ID_KEY: "a"}]
    result = route_suggestions(suggestions)
    assert result == suggestions


def test_route_suggestions_multiple_no_start():
    suggestions = [
        {"x": 0.0, "y": 0.0, ID_KEY: "a"},
        {"x": 1.0, "y": 0.0, ID_KEY: "b"},
    ]
    result = route_suggestions(suggestions)
    assert len(result) == 2
    assert {s[ID_KEY] for s in result} == {"a", "b"}


def test_route_suggestions_multiple_with_start():
    suggestions = [
        {"x": 10.0, "y": 0.0, ID_KEY: "far"},
        {"x": 1.0, "y": 0.0, ID_KEY: "near"},
    ]
    start = {"x": 0.0, "y": 0.0}
    result = route_suggestions(suggestions, starting_position=start)
    # "near" should come first since it's closer to start
    assert result[0][ID_KEY] == "near"


def test_route_suggestions_ignores_non_float_values():
    suggestions = [
        {"x": 0.0, "label": "foo", ID_KEY: "a"},
        {"x": 1.0, "label": "bar", ID_KEY: "b"},
    ]
    result = route_suggestions(suggestions)
    assert len(result) == 2
