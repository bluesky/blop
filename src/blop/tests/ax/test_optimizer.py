import numpy as np
import pytest
from ax import ChoiceParameterConfig, RangeParameterConfig
from ax.exceptions.core import UnsupportedError

from blop.ax.optimizer import AxOptimizer


def test_ax_optimizer_init():
    parameters = [
        RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
        RangeParameterConfig(name="x2", bounds=(-5.0, 5.0), parameter_type="float"),
        ChoiceParameterConfig(name="x3", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True),
    ]
    optimizer = AxOptimizer(
        parameters=parameters,
        objective="y1,-y2",
        parameter_constraints=["x1 + x2 <= 10"],
        outcome_constraints=["y1 >= 0", "y2 <= 0"],
    )

    assert optimizer.ax_client is not None
    with pytest.raises(UnsupportedError):
        optimizer.ax_client.configure_experiment(parameters)


def test_ax_optimizer_suggest():
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
            RangeParameterConfig(name="x2", bounds=(-5.0, 5.0), parameter_type="float"),
            ChoiceParameterConfig(name="x3", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True),
        ],
        objective="y1,-y2",
        parameter_constraints=["x1 + x2 <= 10"],
        outcome_constraints=["y1 >= 0", "y2 <= 0"],
    )
    suggestions = optimizer.suggest(num_points=2)
    assert len(suggestions) == 2
    for i, suggestion in enumerate(suggestions):
        assert suggestion["_id"] == i
        assert "x1" in suggestion
        assert "x2" in suggestion
        assert "x3" in suggestion


def test_ax_optimizer_ingest():
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
            RangeParameterConfig(name="x2", bounds=(-5.0, 5.0), parameter_type="float"),
            ChoiceParameterConfig(name="x3", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True),
        ],
        objective="y1,-y2",
        parameter_constraints=["x1 + x2 <= 10"],
        outcome_constraints=["y1 >= 0", "y2 <= 0"],
    )
    optimizer.ingest(
        [
            {"x1": 0.0, "x2": 0.0, "x3": 0, "y1": 1.0, "y2": 2.0},
            {"x1": 0.1, "x2": 0.2, "x3": 1, "y1": 3.0, "y2": 4.0},
        ]
    )

    optimizer.ax_client.configure_generation_strategy()
    summary_df = optimizer.ax_client.summarize()
    assert len(summary_df) == 2
    assert np.all(summary_df["x1"].values == [0.0, 0.1])
    assert np.all(summary_df["x2"].values == [0.0, 0.2])
    assert np.all(summary_df["x3"].values == [0, 1])
    assert np.all(summary_df["y1"].values == [1.0, 3.0])
    assert np.all(summary_df["y2"].values == [2.0, 4.0])


def test_ax_optimizer_ingest_baseline():
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
            RangeParameterConfig(name="x2", bounds=(-5.0, 5.0), parameter_type="float"),
            ChoiceParameterConfig(name="x3", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True),
        ],
        objective="y1,-y2",
        parameter_constraints=["x1 + x2 <= 10"],
        outcome_constraints=["y1 >= 0", "y2 <= 0"],
    )
    optimizer.ingest([{"x1": 0.0, "x2": 0.0, "x3": 0, "y1": 1.0, "y2": 2.0, "_id": "baseline"}])
    optimizer.ax_client.configure_generation_strategy()
    summary_df = optimizer.ax_client.summarize()
    assert len(summary_df) == 1
    assert summary_df["arm_name"].values[0] == "baseline"
    assert np.all(summary_df["x1"].values == [0.0])
    assert np.all(summary_df["x2"].values == [0.0])
    assert np.all(summary_df["x3"].values == [0])
    assert np.all(summary_df["y1"].values == [1.0])
    assert np.all(summary_df["y2"].values == [2.0])


def test_ax_optimizer_suggest_ingest():
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
            RangeParameterConfig(name="x2", bounds=(-5.0, 5.0), parameter_type="float"),
            ChoiceParameterConfig(name="x3", values=[0, 1, 2, 3, 4, 5], parameter_type="int", is_ordered=True),
        ],
        objective="y1,-y2",
        parameter_constraints=["x1 + x2 <= 10"],
        outcome_constraints=["y1 >= 0", "y2 <= 0"],
    )
    suggestions = optimizer.suggest(num_points=2)
    outcomes = [
        {"_id": suggestions[0]["_id"], "y1": 1.0, "y2": 2.0},
        {"_id": suggestions[1]["_id"], "y1": 3.0, "y2": 4.0},
    ]
    optimizer.ingest(outcomes)

    summary_df = optimizer.ax_client.summarize()
    assert len(summary_df) == 2
    assert np.all(summary_df["y1"].values == [1.0, 3.0])
    assert np.all(summary_df["y2"].values == [2.0, 4.0])


def test_ax_optimizer_checkpoint(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.json"

    # Save to checkpoint
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
        ],
        objective="y1",
        checkpoint_path=str(checkpoint_path),
    )
    suggestions = optimizer.suggest(num_points=2)
    outcomes = [
        {"_id": suggestions[0]["_id"], "y1": 1.0},
        {"_id": suggestions[1]["_id"], "y1": 3.0},
    ]
    optimizer.ingest(outcomes)

    assert not checkpoint_path.exists()
    optimizer.checkpoint()
    assert checkpoint_path.exists()

    # Load from checkpoint
    optimizer = AxOptimizer.from_checkpoint(str(checkpoint_path))
    summary_df = optimizer.ax_client.summarize()
    assert "x1" in summary_df.columns
    assert "y1" in summary_df.columns
    assert len(summary_df) == 2
    assert optimizer.checkpoint_path == str(checkpoint_path)


def test_ax_optimizer_checkpoint_no_path():
    optimizer = AxOptimizer(
        parameters=[
            RangeParameterConfig(name="x1", bounds=(-5.0, 5.0), parameter_type="float"),
        ],
        objective="y1",
    )

    with pytest.raises(ValueError):
        optimizer.checkpoint()
