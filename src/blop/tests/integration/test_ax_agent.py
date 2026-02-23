from blop.ax.agent import Agent
from blop.ax.dof import RangeDOF
from blop.ax.objective import Objective
from simulations.beamline import TiledBeamline


def test_ax_agent_sim_beamline(RE, setup):
    beamline = TiledBeamline(name="bl")
    beamline.det.noise.put(False)

    dofs = [
        RangeDOF(actuator=beamline.kbv_dsv, bounds=(-5.0, 5.0), parameter_type="float"),
        RangeDOF(actuator=beamline.kbv_usv, bounds=(-5.0, 5.0), parameter_type="float"),
        RangeDOF(actuator=beamline.kbh_dsh, bounds=(-5.0, 5.0), parameter_type="float"),
        RangeDOF(actuator=beamline.kbh_ush, bounds=(-5.0, 5.0), parameter_type="float"),
    ]

    objectives = [
        Objective(name="bl_det_sum", minimize=False),
        Objective(name="bl_det_wid_x", minimize=True),
        Objective(name="bl_det_wid_y", minimize=True),
    ]

    def evaluation_function(uid: str, suggestions: list[dict]) -> list[dict]:
        run = setup[uid]

        bl_det_sums = run["primary/bl_det_sum"].read()
        bl_det_wid_x = run["primary/bl_det_wid_x"].read()
        bl_det_wid_y = run["primary/bl_det_wid_y"].read()

        trial_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]
        outcomes = []
        for suggestion in suggestions:
            idx = trial_ids.index(suggestion["_id"])
            outcome = {
                "_id": suggestion["_id"],
                "bl_det_sum": bl_det_sums[idx],
                "bl_det_wid_x": bl_det_wid_x[idx],
                "bl_det_wid_y": bl_det_wid_y[idx],
            }
            outcomes.append(outcome)

        return outcomes

    agent = Agent(
        sensors=[beamline.det],
        dofs=dofs,
        objectives=objectives,
        evaluation_function=evaluation_function,
    )
    RE(agent.optimize(iterations=12, n_points=1))
