from ax import Client, RangeParameterConfig
from ax.api.protocols import IMetric

from blop.ax.objective import Objective, OutcomeConstraint, ScalarizedObjective, to_ax_objective_str


def test_objective_to_ax_objective_str():
    objective = Objective(name="test_objective", minimize=True)
    assert to_ax_objective_str([objective]) == "-test_objective"

    objective = Objective(name="test_objective", minimize=False)
    assert to_ax_objective_str([objective]) == "test_objective"


def test_multiple_objectives_to_ax_objective_str():
    objective1 = Objective(name="test_objective1", minimize=True)
    objective2 = Objective(name="test_objective2", minimize=False)
    assert to_ax_objective_str([objective1, objective2]) == "-test_objective1, test_objective2"


def test_scalarized_objective_to_ax_objective_str():
    scalarized_objective = ScalarizedObjective(expression="x + y", minimize=True, x="test_objective1", y="test_objective2")
    assert scalarized_objective.ax_expression == "-(test_objective1 + test_objective2)"


def test_scalarized_objective_to_ax_objective_str_minimize():
    scalarized_objective = ScalarizedObjective(expression="x + y", minimize=False, x="test_objective1", y="test_objective2")
    assert scalarized_objective.ax_expression == "test_objective1 + test_objective2"


def test_outcome_constraint_on_objective():
    objective = Objective(name="test_objective", minimize=True)
    constraint = OutcomeConstraint(constraint="x >= 0.5", x=objective)
    assert constraint.ax_constraint == "test_objective >= 0.5"


def test_outcome_constraint_on_metrics():
    metric1 = IMetric(name="test_metric1")
    metric2 = IMetric(name="test_metric2")
    constraint = OutcomeConstraint(constraint="x + y >= 0.5", x=metric1, y=metric2)
    assert constraint.ax_constraint == "test_metric1 + test_metric2 >= 0.5"


def test_ax_api_consistency_multi_objective():
    client = Client()
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(name="x", bounds=(0, 1), parameter_type="float"),
            RangeParameterConfig(name="y", bounds=(0, 1), parameter_type="float"),
        ]
    )

    objectives = [
        Objective(name="objective1", minimize=True),
        Objective(name="objective2", minimize=False),
    ]
    metrics = [
        IMetric(name="metric1"),
    ]
    constraints = [
        OutcomeConstraint(constraint="x <= 0.5", x=objectives[0]),
        OutcomeConstraint(constraint="y >= 0.5", y=objectives[1]),
        OutcomeConstraint(constraint="x >= 0.5", x=metrics[0]),
    ]
    client.configure_metrics(metrics)
    client.configure_optimization(
        objective=to_ax_objective_str(objectives),
        outcome_constraints=[c.ax_constraint for c in constraints],
    )


def test_ax_api_consistency_single_objective():
    client = Client()
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(name="x", bounds=(0, 1), parameter_type="float"),
        ]
    )
    objective = Objective(name="objective", minimize=True)
    metric = IMetric(name="metric")
    constraint = OutcomeConstraint(constraint="2 * x >= 0.5", x=metric)
    client.configure_metrics([metric])
    client.configure_optimization(objective=to_ax_objective_str([objective]), outcome_constraints=[constraint.ax_constraint])


def test_ax_api_consistency_scalarized_objective():
    client = Client()
    client.configure_experiment(
        parameters=[
            RangeParameterConfig(name="x", bounds=(0, 1), parameter_type="float"),
        ]
    )
    scalarized_objective = ScalarizedObjective(expression="2 * x + y", minimize=True, x="objective1", y="objective2")
    client.configure_optimization(objective=scalarized_objective.ax_expression)
