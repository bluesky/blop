"""
Queueserver startup script for the Blop tutorial.

This script runs inside the RE Manager process. It sets up:
- A RunEngine with ZMQ document publishing
- Simulated motors and a Himmelblau detector
- The default_acquire plan from Blop
"""

from bluesky import RunEngine
from bluesky_queueserver import is_re_worker_active
from ophyd.sim import SynAxis, SynSignal

# ---------------------------------------------------------------------------
# RunEngine setup
# ---------------------------------------------------------------------------
RE = RunEngine({})

if is_re_worker_active():
    import os

    from bluesky.callbacks.zmq import Publisher as ZmqPublisher

    _zmq_proxy_in = os.environ.get("BLUESKY_ZMQ_PROXY_IN_ADDR", "tcp://zmq-proxy:5577")
    _host_port = _zmq_proxy_in.replace("tcp://", "").split(":")
    _zmq_addr_tuple = (_host_port[0], int(_host_port[1]))
    _publisher = ZmqPublisher(_zmq_addr_tuple)
    RE.subscribe(_publisher)
    print(f"[STARTUP] RE subscribed ZMQ Publisher -> {_zmq_addr_tuple}")


# ---------------------------------------------------------------------------
# Simulated devices
# ---------------------------------------------------------------------------
motor1 = SynAxis(name="motor1", labels={"motors"})
motor2 = SynAxis(name="motor2", labels={"motors"})


def _compute_himmelblau():
    x = motor1.read()["motor1"]["value"]  # type: ignore
    y = motor2.read()["motor2"]["value"]  # type: ignore
    return float((x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2)


himmel_det = SynSignal(name="himmel_det", func=_compute_himmelblau, labels={"detectors"})


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------
import bluesky.plans as bp  # noqa: E402


def default_acquire(suggestions, actuators, sensors, *, md=None):
    """
    Acquire data by moving actuators to suggested positions and reading sensors.

    This is a simplified version of blop.plans.default_acquire without complex
    type annotations that would cause pydantic validation issues in queueserver.
    """
    # Build list_scan args: [motor1, [val1, ...], motor2, [val2, ...], ...]
    plan_args = []
    for actuator in actuators:
        values = [s[actuator.name] for s in suggestions]
        plan_args.append(actuator)
        plan_args.append(values)

    _md = {"blop_suggestions": suggestions}
    if md:
        _md.update(md)

    yield from bp.list_scan(list(sensors), *plan_args, md=_md)
