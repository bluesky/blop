import numpy as np
import pytest

from blop.protocols import ID_KEY
from blop.utils import InferredReadable, Source, _infer_data_key, route_suggestions

# InferredReadable tests


def test_inferred_readable_scalar_number():
    r = InferredReadable("x", Source.OTHER, 1.5)
    assert r.name == "x"
    assert r.parent is None
    read = r.read()
    assert read["x"]["value"] == 1.5
    assert "timestamp" in read["x"]
    assert r.describe()["x"]["dtype"] == "number"
    assert r.hints["fields"] == ["x"]


def test_inferred_readable_scalar_string():
    r = InferredReadable("ids", Source.OTHER, ["0"])
    assert r.read()["ids"]["value"] == "0"
    assert r.describe()["ids"]["dtype"] == "string"


def test_inferred_readable_array():
    r = InferredReadable("arr", Source.OTHER, [0.0, 0.1])
    assert r.read()["arr"]["value"] == [0.0, 0.1]
    assert r.describe()["arr"]["dtype"] == "array"


def test_inferred_readable_update():
    r = InferredReadable("x", Source.OTHER, 1.5)
    r.update(2.0)
    assert r.read()["x"]["value"] == 2.0

    r2 = InferredReadable("arr", Source.OTHER, [0.0, 0.1])
    r2.update(np.array([1.0, 2.0]))
    assert list(r2.read()["arr"]["value"]) == [1.0, 2.0]


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

    result = route_suggestions(suggestions)
    assert len(result) == 2


def test_route_suggestions_visits_each_reported_suggestion_once():
    suggestions = [
        {ID_KEY: 0, "big_r": 152982.84327559808, "toroid_focus:toroidMirror01:r": 1162.0765699687756},
        {ID_KEY: 1, "big_r": 161916.56982474664, "toroid_focus:toroidMirror01:r": 1488.1603493037976},
        {ID_KEY: 2, "big_r": 139461.95450559893, "toroid_focus:toroidMirror01:r": 798.5515472161926},
        {ID_KEY: 3, "big_r": 150664.77394611278, "toroid_focus:toroidMirror01:r": 1403.9407343286432},
        {ID_KEY: 4, "big_r": 158210.14488064387, "toroid_focus:toroidMirror01:r": 960.4307756441988},
        {ID_KEY: 5, "big_r": 155033.62997383514, "toroid_focus:toroidMirror01:r": 1226.6657248753465},
        {ID_KEY: 6, "big_r": 147491.13174103835, "toroid_focus:toroidMirror01:r": 1041.026067795173},
        {ID_KEY: 7, "big_r": 143785.51106848457, "toroid_focus:toroidMirror01:r": 1654.6320047843612},
        {ID_KEY: 8, "big_r": 166242.9699762229, "toroid_focus:toroidMirror01:r": 722.9002070295967},
        {ID_KEY: 9, "big_r": 165175.66349305847, "toroid_focus:toroidMirror01:r": 1338.998358571843},
    ]
    start = {"big_r": 150000.0, "toroid_focus:toroidMirror01:r": 1500.0}

    result = route_suggestions(suggestions, starting_position=start)
    routed_ids = [suggestion[ID_KEY] for suggestion in result]

    assert routed_ids[0] == 3
    assert len(routed_ids) == len(suggestions)
    assert set(routed_ids) == {suggestion[ID_KEY] for suggestion in suggestions}


# _infer_data_key source value tests


@pytest.mark.parametrize("source", list(Source))
def test_infer_data_key_source_is_enum_value(source):
    """The 'source' field in the DataKey must be the enum's string value, not its repr."""
    data_key = _infer_data_key(source, 1.0)
    assert data_key["source"] == source.value
    assert "Source." not in data_key["source"]
