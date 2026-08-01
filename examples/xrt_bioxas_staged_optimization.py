"""Prototype staged optimization workflow for the XRT BioXAS simulation.

This script is intentionally separate from the rendered documentation because full
XRT ray tracing is expensive. It exercises the workflow we want to explain in the
future tutorial:

1. Optimize M1 against an upstream diagnostic screen.
2. Optimize M2 against a downstream diagnostic screen.
3. Optimize DBHR settings at the sample while holding upstream settings fixed.
4. Reuse the same sample agent, tighten bounds, release the fixed DOFs, and run a
   final global sample optimization.

Run a cheap setup-only validation with:

    pixi run -e docs python examples/xrt_bioxas_staged_optimization.py --setup-only

Run the full workflow with small default iteration counts with:

    pixi run -e docs python examples/xrt_bioxas_staged_optimization.py
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from blop_sim.backends.xrt_bioxas import XRTBIOXASBackend
from blop_sim.devices import DetectorDevice
from blop_sim.devices.xrt import KBMirror
from blop_sim.devices.xrt_bioxas import DBHR
from bluesky.run_engine import RunEngine
from bluesky_tiled_plugins import TiledWriter
from ophyd_async.core import StaticPathProvider, UUIDFilenameProvider
from tiled.client import from_uri  # type: ignore[import-untyped]
from tiled.client.container import Container
from tiled.server import SimpleTiledServer

from blop.ax import Agent, Objective, RangeDOF
from blop.protocols import EvaluationFunction

logger = logging.getLogger(__name__)

DETECTOR_STORAGE: Final = "/tmp/blop/sim/xrt-bioxas-staged"

MIRROR1_RADIUS_BOUNDS: Final = (2_500_000.0, 45_000_000.0)
MIRROR2_RADIUS_BOUNDS: Final = (650_000.0, 4_000_000.0)
MIRROR_EXTRA_PITCH_BOUNDS: Final = (-1e-4, 1e-4)
DBHR_EXTRA_PITCH_BOUNDS: Final = (-1e-4, 1e-4)
DBHR_EXTRA_ROLL_BOUNDS: Final = (-1e-3, 1e-3)

TIGHT_RADIUS_HALF_WIDTH: Final = 0.10
TIGHT_MIRROR_PITCH_HALF_WIDTH: Final = 2e-5
TIGHT_DBHR_PITCH_HALF_WIDTH: Final = 2e-5
TIGHT_DBHR_ROLL_HALF_WIDTH: Final = 2e-4

STAGE_SCORE_WEIGHTS: Final = {
    "pre_m2": {"flux_weight": 0.35, "width_weight": 0.15, "height_weight": 0.15, "centroid_weight": 0.35},
    "pre_dbhr": {"flux_weight": 0.35, "width_weight": 0.20, "height_weight": 0.15, "centroid_weight": 0.30},
    "sample": {"flux_weight": 0.35, "width_weight": 0.20, "height_weight": 0.20, "centroid_weight": 0.25},
}

STAGE_TARGET_TOLERANCES: Final = {
    "pre_m2": {"width_fraction": 0.20, "height_fraction": 0.20, "centroid_pixels": 5.0},
    "pre_dbhr": {"width_fraction": 0.20, "height_fraction": 0.20, "centroid_pixels": 4.0},
    "sample": {"width_fraction": 0.15, "height_fraction": 0.15, "centroid_pixels": 2.0},
}

PERTURBATION_PRESETS: Final = {
    "none": {},
    "mild": {
        "mirror1-radius": 9_000_000.0,
        "mirror1-extraPitch": 1.5e-5,
        "mirror2-radius": 2_100_000.0,
        "mirror2-extraPitch": -1.5e-5,
        "dbhr1-extraPitch": 1.5e-5,
        "dbhr1-extraRoll": 1.5e-4,
        "dbhr2-extraPitch": -1.5e-5,
        "dbhr2-extraRoll": -1.5e-4,
    },
    "moderate": {
        "mirror1-radius": 12_000_000.0,
        "mirror1-extraPitch": 3.0e-5,
        "mirror2-radius": 1_800_000.0,
        "mirror2-extraPitch": -3.0e-5,
        "dbhr1-extraPitch": 3.0e-5,
        "dbhr1-extraRoll": 3.0e-4,
        "dbhr2-extraPitch": -3.0e-5,
        "dbhr2-extraRoll": -3.0e-4,
    },
}


@dataclass(frozen=True)
class ImageStats:
    """Reduced beam-image statistics used by the stage evaluators."""

    intensity: float
    width: float
    height: float
    area: float
    centroid_x: float
    centroid_y: float
    centroid_error: float


@dataclass(frozen=True)
class BeamlineDevices:
    """Devices used by the staged BioXAS workflow."""

    backend: XRTBIOXASBackend
    detector: DetectorDevice
    mirror1: KBMirror
    mirror2: KBMirror
    dbhr1: DBHR
    dbhr2: DBHR


@dataclass(frozen=True)
class RuntimeServices:
    """Runtime services that must stay alive for the workflow duration."""

    run_engine: RunEngine
    tiled_client: Container
    tiled_server: SimpleTiledServer


@dataclass(frozen=True)
class StagedAgents:
    """Agents and DOFs used by the staged BioXAS workflow."""

    m1_agent: Agent
    m2_agent: Agent
    sample_agent: Agent
    m1_radius: RangeDOF
    m1_pitch: RangeDOF
    m2_radius: RangeDOF
    m2_pitch: RangeDOF
    dbhr1_pitch: RangeDOF
    dbhr1_roll: RangeDOF
    dbhr2_pitch: RangeDOF
    dbhr2_roll: RangeDOF


@dataclass(frozen=True)
class StageBest:
    """Best parameterization and metrics returned by an agent."""

    parameters: dict[str, float]
    metrics: dict[str, float]


@dataclass(frozen=True)
class BeamMeasurement:
    """A reduced beam measurement at one diagnostic screen."""

    label: str
    target: str
    stage_name: str
    stats: ImageStats
    score: float


@dataclass(frozen=True)
class StageTarget:
    """Nominal target beam properties for one diagnostic screen."""

    stage_name: str
    target: str
    intensity: float
    width: float
    height: float
    centroid_x: float
    centroid_y: float
    width_tolerance: float
    height_tolerance: float
    centroid_tolerance: float


@dataclass(frozen=True)
class TargetScore:
    """Target-relative score components for one beam measurement."""

    score: float
    target_distance: float
    flux_ratio: float
    width_error: float
    height_error: float
    centroid_error: float


def _as_float_metrics(metrics: dict) -> dict[str, float]:
    """Convert Ax metric values into plain floats."""
    flattened = {}
    for name, value in metrics.items():
        if isinstance(value, tuple):
            flattened[name] = float(value[0])
        else:
            flattened[name] = float(value)
    return flattened


def _best_from_agent(agent: Agent) -> StageBest:
    """Return the single best point from a single-objective agent."""
    best_points = agent.get_best_points()
    if len(best_points) != 1:
        raise RuntimeError(f"Expected one best point for a single-objective agent, got {len(best_points)}.")

    _trial_index, parameters, metrics = best_points[0]
    return StageBest(parameters=dict(parameters), metrics=_as_float_metrics(dict(metrics)))


def _clipped_window(center: float, half_width: float, limits: tuple[float, float]) -> tuple[float, float]:
    """Build a bound window around a center value while respecting absolute limits."""
    lower, upper = limits
    return max(lower, center - half_width), min(upper, center + half_width)


def _relative_window(center: float, relative_half_width: float, limits: tuple[float, float]) -> tuple[float, float]:
    """Build a relative bound window around a center value."""
    return _clipped_window(center, abs(center) * relative_half_width, limits)


def build_stage_target(measurement: BeamMeasurement) -> StageTarget:
    """Build a target from a nominal measurement."""
    tolerances = STAGE_TARGET_TOLERANCES[measurement.stage_name]
    stats = measurement.stats
    return StageTarget(
        stage_name=measurement.stage_name,
        target=measurement.target,
        intensity=max(stats.intensity, 1.0),
        width=max(stats.width, 1.0),
        height=max(stats.height, 1.0),
        centroid_x=stats.centroid_x,
        centroid_y=stats.centroid_y,
        width_tolerance=max(stats.width * tolerances["width_fraction"], 1.0),
        height_tolerance=max(stats.height * tolerances["height_fraction"], 1.0),
        centroid_tolerance=tolerances["centroid_pixels"],
    )


def score_stats(stats: ImageStats, target: StageTarget) -> TargetScore:
    """Score a measurement by proximity to the nominal BioXAS beam target."""
    weights = STAGE_SCORE_WEIGHTS[target.stage_name]
    width_error = (stats.width - target.width) / target.width_tolerance
    height_error = (stats.height - target.height) / target.height_tolerance
    centroid_error = np.hypot(stats.centroid_x - target.centroid_x, stats.centroid_y - target.centroid_y)
    normalized_centroid_error = centroid_error / target.centroid_tolerance
    flux_ratio = stats.intensity / target.intensity
    low_flux_penalty = max(0.0, 1.0 - flux_ratio)

    target_distance = float(np.sqrt(width_error**2 + height_error**2 + normalized_centroid_error**2 + low_flux_penalty**2))
    score = float(
        weights["flux_weight"] * np.log1p(flux_ratio)
        - weights["width_weight"] * width_error**2
        - weights["height_weight"] * height_error**2
        - weights["centroid_weight"] * normalized_centroid_error**2
        - low_flux_penalty**2
    )
    return TargetScore(
        score=score,
        target_distance=target_distance,
        flux_ratio=float(flux_ratio),
        width_error=float(width_error),
        height_error=float(height_error),
        centroid_error=float(centroid_error),
    )


def compute_image_stats(image: np.ndarray, threshold_fraction: float = 0.05) -> ImageStats:
    """Compute intensity, spot size, area, and centroid statistics from a detector image."""
    gray = np.asarray(image).squeeze().astype(np.float64)
    if gray.ndim == 3:
        gray = gray.sum(axis=-1)

    max_value = float(np.nanmax(gray)) if gray.size else 0.0
    if max_value <= 0:
        return ImageStats(
            intensity=0.0,
            width=float(gray.shape[1]) if gray.ndim == 2 else 0.0,
            height=float(gray.shape[0]) if gray.ndim == 2 else 0.0,
            area=float(gray.size),
            centroid_x=0.0,
            centroid_y=0.0,
            centroid_error=float("inf"),
        )

    thresholded = np.where(gray >= threshold_fraction * max_value, gray, 0.0)
    total = float(thresholded.sum())
    if total <= 0:
        return ImageStats(
            intensity=0.0,
            width=float(gray.shape[1]),
            height=float(gray.shape[0]),
            area=float(gray.size),
            centroid_x=0.0,
            centroid_y=0.0,
            centroid_error=float("inf"),
        )

    height_px, width_px = thresholded.shape
    x_coords = np.arange(width_px, dtype=np.float64)
    y_coords = np.arange(height_px, dtype=np.float64)
    x_profile = thresholded.sum(axis=0)
    y_profile = thresholded.sum(axis=1)

    centroid_x = float((x_coords * x_profile).sum() / total)
    centroid_y = float((y_coords * y_profile).sum() / total)
    x_var = float(((x_coords - centroid_x) ** 2 * x_profile).sum() / total)
    y_var = float(((y_coords - centroid_y) ** 2 * y_profile).sum() / total)
    center_x = 0.5 * (width_px - 1)
    center_y = 0.5 * (height_px - 1)

    return ImageStats(
        intensity=total,
        width=2 * np.sqrt(max(x_var, 0.0)),
        height=2 * np.sqrt(max(y_var, 0.0)),
        area=float(np.count_nonzero(thresholded)),
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        centroid_error=float(np.hypot(centroid_x - center_x, centroid_y - center_y)),
    )


class StageEvaluation(EvaluationFunction):
    """Base image evaluator for a single optimization stage."""

    def __init__(self, tiled_client: Container, target: StageTarget):
        self.tiled_client = tiled_client
        self.target = target
        self.stage_name = target.stage_name

    @property
    def score_name(self) -> str:
        """Name of the scalar objective returned by this evaluator."""
        return f"{self.stage_name}_score"

    def score(self, stats: ImageStats) -> float:
        """Compute the stage-specific scalar score to maximize."""
        return score_stats(stats, self.target).score

    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:
        """Read images from Tiled and return one outcome per acquired suggestion."""
        run = self.tiled_client[uid]
        images = run["primary/det_image"].read()
        suggestion_ids = [suggestion["_id"] for suggestion in run.metadata["start"]["blop_suggestions"]]

        outcomes = []
        for index, suggestion_id in enumerate(suggestion_ids):
            stats = compute_image_stats(images[index])
            target_score = score_stats(stats, self.target)
            outcomes.append(
                {
                    "_id": suggestion_id,
                    self.score_name: target_score.score,
                    f"{self.stage_name}_intensity": stats.intensity,
                    f"{self.stage_name}_width": stats.width,
                    f"{self.stage_name}_height": stats.height,
                    f"{self.stage_name}_area": stats.area,
                    f"{self.stage_name}_centroid_x": stats.centroid_x,
                    f"{self.stage_name}_centroid_y": stats.centroid_y,
                    f"{self.stage_name}_centroid_error": stats.centroid_error,
                    f"{self.stage_name}_target_distance": target_score.target_distance,
                    f"{self.stage_name}_flux_ratio": target_score.flux_ratio,
                    f"{self.stage_name}_width_error": target_score.width_error,
                    f"{self.stage_name}_height_error": target_score.height_error,
                }
            )

        return outcomes


class PreM2Evaluation(StageEvaluation):
    """Evaluate M1 delivery using the PreM2 screen."""

    def __init__(self, tiled_client: Container, target: StageTarget):
        super().__init__(tiled_client, target)


class PreDBHREvaluation(StageEvaluation):
    """Evaluate M2 delivery using the PreDBHR screen."""

    def __init__(self, tiled_client: Container, target: StageTarget):
        super().__init__(tiled_client, target)


class SampleEvaluation(StageEvaluation):
    """Evaluate endstation delivery using the sample screen."""

    def __init__(self, tiled_client: Container, target: StageTarget):
        super().__init__(tiled_client, target)


def setup_run_engine(storage: str) -> RuntimeServices:
    """Create a RunEngine subscribed to an in-process Tiled server."""
    Path(storage).mkdir(parents=True, exist_ok=True)
    tiled_server = SimpleTiledServer(readable_storage=[storage])
    tiled_client = from_uri(tiled_server.uri)
    tiled_writer = TiledWriter(tiled_client)

    run_engine = RunEngine({})
    run_engine.subscribe(tiled_writer)
    return RuntimeServices(run_engine=run_engine, tiled_client=tiled_client, tiled_server=tiled_server)


def setup_devices(storage: str) -> BeamlineDevices:
    """Create the BioXAS simulation backend and ophyd-async devices."""
    backend = XRTBIOXASBackend(target="SampleScreen")
    path_provider = StaticPathProvider(UUIDFilenameProvider(), PurePath(storage))
    detector = DetectorDevice(backend, path_provider, name="det")
    mirror1 = KBMirror(backend, mirror_index=0, initial_radius=7_120_000.0, name="mirror1")
    mirror2 = KBMirror(backend, mirror_index=1, initial_radius=2_500_000.0, name="mirror2")
    dbhr1 = DBHR(backend, optic_index=0, extraPitch=0.0, extraRoll=0.0, name="dbhr1")
    dbhr2 = DBHR(backend, optic_index=1, extraPitch=0.0, extraRoll=0.0, name="dbhr2")

    return BeamlineDevices(
        backend=backend,
        detector=detector,
        mirror1=mirror1,
        mirror2=mirror2,
        dbhr1=dbhr1,
        dbhr2=dbhr2,
    )


async def _set_signal_values(values: dict) -> None:
    """Set ophyd-async soft-signal values outside a Bluesky plan."""
    for signal, value in values.items():
        await signal.set(value)


async def _read_signal_values(signals: list) -> dict[str, float]:
    """Read ophyd-async soft-signal values outside a Bluesky plan."""
    return {signal.name: float(await signal.get_value()) for signal in signals}


def current_parameterization(agent: Agent) -> dict[str, float]:
    """Read the current values of the actuators controlled by an agent."""
    return asyncio.run(_read_signal_values(list(agent.actuators)))


def set_parameterization(devices: BeamlineDevices, parameters: dict[str, float]) -> None:
    """Move known BioXAS DOFs to an explicit parameterization for direct measurement."""
    signals_by_name = {
        devices.mirror1.radius.name: devices.mirror1.radius,
        devices.mirror1.extraPitch.name: devices.mirror1.extraPitch,
        devices.mirror2.radius.name: devices.mirror2.radius,
        devices.mirror2.extraPitch.name: devices.mirror2.extraPitch,
        devices.dbhr1.extraPitch.name: devices.dbhr1.extraPitch,
        devices.dbhr1.extraRoll.name: devices.dbhr1.extraRoll,
        devices.dbhr2.extraPitch.name: devices.dbhr2.extraPitch,
        devices.dbhr2.extraRoll.name: devices.dbhr2.extraRoll,
    }
    values = {signals_by_name[name]: value for name, value in parameters.items() if name in signals_by_name}
    asyncio.run(_set_signal_values(values))


def apply_initial_perturbation(devices: BeamlineDevices, preset: str) -> None:
    """Apply a named starting-point perturbation before baseline measurements."""
    parameters = PERTURBATION_PRESETS[preset]
    if not parameters:
        logger.info("No initial perturbation applied")
        return

    logger.info("Applying %s initial perturbation: %s", preset, parameters)
    set_parameterization(devices, parameters)


def measure_current_beam(
    devices: BeamlineDevices,
    *,
    label: str,
    target: str,
    stage_name: str,
    stage_target: StageTarget | None = None,
) -> BeamMeasurement:
    """Directly ray trace the current beam and summarize it at a target screen."""
    devices.backend.target = target
    image = asyncio.run(devices.backend.generate_beam())
    stats = compute_image_stats(image)
    if stage_target is None:
        score = 0.0
    else:
        score = score_stats(stats, stage_target).score
    measurement = BeamMeasurement(label=label, target=target, stage_name=stage_name, stats=stats, score=score)
    log_measurement(measurement)
    return measurement


def measure_nominal_targets(devices: BeamlineDevices) -> dict[str, StageTarget]:
    """Measure nominal BioXAS beam targets before applying any perturbation."""
    target_specs = {
        "pre_m2": ("PreM2Screen", "Nominal PreM2"),
        "pre_dbhr": ("PreDBHRScreen", "Nominal PreDBHR"),
        "sample": ("SampleScreen", "Nominal sample"),
    }
    targets = {}
    for stage_name, (screen, label) in target_specs.items():
        measurement = measure_current_beam(devices, label=label, target=screen, stage_name=stage_name)
        target = build_stage_target(measurement)
        targets[stage_name] = target
        log_target(target)
    return targets


def measurement_metrics(measurement: BeamMeasurement, target: StageTarget) -> dict[str, float]:
    """Convert a direct beam measurement into optimizer metric names."""
    stage_name = measurement.stage_name
    stats = measurement.stats
    target_score = score_stats(stats, target)
    return {
        f"{stage_name}_score": target_score.score,
        f"{stage_name}_intensity": stats.intensity,
        f"{stage_name}_width": stats.width,
        f"{stage_name}_height": stats.height,
        f"{stage_name}_area": stats.area,
        f"{stage_name}_centroid_x": stats.centroid_x,
        f"{stage_name}_centroid_y": stats.centroid_y,
        f"{stage_name}_centroid_error": stats.centroid_error,
        f"{stage_name}_target_distance": target_score.target_distance,
        f"{stage_name}_flux_ratio": target_score.flux_ratio,
        f"{stage_name}_width_error": target_score.width_error,
        f"{stage_name}_height_error": target_score.height_error,
    }


def log_measurement(measurement: BeamMeasurement) -> None:
    """Log one reduced beam measurement."""
    stats = measurement.stats
    message = (
        f"{measurement.label} at {measurement.target}: score={measurement.score:g} intensity={stats.intensity:g} "
        f"width={stats.width:g} height={stats.height:g} centroid_error={stats.centroid_error:g} area={stats.area:g}"
    )
    logger.info(message)
    print(message)


def log_stagebest_improvement(label: str, baseline: StageBest, result: StageBest, score_name: str, stage_name: str) -> None:
    """Log a compact comparison between two ingested stage points."""
    baseline_metrics = baseline.metrics
    result_metrics = result.metrics
    message = (
        f"{label} improvement vs baseline: score {result_metrics[score_name] - baseline_metrics[score_name]:+g}, "
        f"intensity {result_metrics[f'{stage_name}_intensity'] - baseline_metrics[f'{stage_name}_intensity']:+g}, "
        f"width {result_metrics[f'{stage_name}_width'] - baseline_metrics[f'{stage_name}_width']:+g}, "
        f"height {result_metrics[f'{stage_name}_height'] - baseline_metrics[f'{stage_name}_height']:+g}, "
        "centroid_error "
        f"{result_metrics[f'{stage_name}_centroid_error'] - baseline_metrics[f'{stage_name}_centroid_error']:+g}, "
        "target_distance "
        f"{result_metrics[f'{stage_name}_target_distance'] - baseline_metrics[f'{stage_name}_target_distance']:+g}"
    )
    logger.info(message)
    print(message)


def log_measurement_improvement(label: str, baseline: BeamMeasurement, result: BeamMeasurement) -> None:
    """Log a compact comparison between two direct beam measurements."""
    message = (
        f"{label} improvement vs initial: score {result.score - baseline.score:+g}, "
        f"intensity {result.stats.intensity - baseline.stats.intensity:+g}, "
        f"width {result.stats.width - baseline.stats.width:+g}, "
        f"height {result.stats.height - baseline.stats.height:+g}, "
        f"centroid_error {result.stats.centroid_error - baseline.stats.centroid_error:+g}"
    )
    logger.info(message)
    print(message)


def log_target(target: StageTarget) -> None:
    """Print the target beam properties for a stage."""
    message = (
        f"{target.stage_name} target at {target.target}: intensity={target.intensity:g} width={target.width:g} "
        f"height={target.height:g} centroid=({target.centroid_x:g}, {target.centroid_y:g}) "
        f"tolerances=(width {target.width_tolerance:g}, height {target.height_tolerance:g}, "
        f"centroid {target.centroid_tolerance:g})"
    )
    logger.info(message)
    print(message)


def log_agent_dofs(agents: StagedAgents) -> None:
    """Log which DOFs each stage agent owns."""
    lines = [
        f"M1 agent DOFs: {[actuator.name for actuator in agents.m1_agent.actuators]}",
        f"M2 agent DOFs: {[actuator.name for actuator in agents.m2_agent.actuators]}",
        f"Sample agent DOFs: {[actuator.name for actuator in agents.sample_agent.actuators]}",
    ]
    for line in lines:
        logger.info(line)
        print(line)


def print_parameter_summary(label: str, parameters: dict[str, float]) -> None:
    """Print and log a compact parameter summary."""
    message = f"{label}: {parameters}"
    logger.info(message)
    print(message)


def log_stagebest(label: str, best: StageBest) -> None:
    """Log and print a StageBest result."""
    print_parameter_summary(f"{label} parameters", best.parameters)
    print_parameter_summary(f"{label} metrics", best.metrics)


def build_agents(devices: BeamlineDevices, tiled_client: Container, targets: dict[str, StageTarget]) -> StagedAgents:
    """Build one agent per independent stage and one shared sample agent."""
    m1_radius = RangeDOF(actuator=devices.mirror1.radius, bounds=MIRROR1_RADIUS_BOUNDS, parameter_type="float")
    m1_pitch = RangeDOF(actuator=devices.mirror1.extraPitch, bounds=MIRROR_EXTRA_PITCH_BOUNDS, parameter_type="float")
    m2_radius = RangeDOF(actuator=devices.mirror2.radius, bounds=MIRROR2_RADIUS_BOUNDS, parameter_type="float")
    m2_pitch = RangeDOF(actuator=devices.mirror2.extraPitch, bounds=MIRROR_EXTRA_PITCH_BOUNDS, parameter_type="float")
    dbhr1_pitch = RangeDOF(actuator=devices.dbhr1.extraPitch, bounds=DBHR_EXTRA_PITCH_BOUNDS, parameter_type="float")
    dbhr1_roll = RangeDOF(actuator=devices.dbhr1.extraRoll, bounds=DBHR_EXTRA_ROLL_BOUNDS, parameter_type="float")
    dbhr2_pitch = RangeDOF(actuator=devices.dbhr2.extraPitch, bounds=DBHR_EXTRA_PITCH_BOUNDS, parameter_type="float")
    dbhr2_roll = RangeDOF(actuator=devices.dbhr2.extraRoll, bounds=DBHR_EXTRA_ROLL_BOUNDS, parameter_type="float")

    m1_agent = Agent(
        sensors=[devices.detector],
        dofs=[m1_radius, m1_pitch],
        objectives=[Objective(name="pre_m2_score", minimize=False)],
        evaluation_function=PreM2Evaluation(tiled_client, targets["pre_m2"]),
        name="bioxas-m1-pre-m2",
        description="Stage 1: optimize M1 delivery at PreM2Screen.",
        experiment_type="xrt-bioxas-staged",
    )
    m2_agent = Agent(
        sensors=[devices.detector],
        dofs=[m2_radius, m2_pitch],
        objectives=[Objective(name="pre_dbhr_score", minimize=False)],
        evaluation_function=PreDBHREvaluation(tiled_client, targets["pre_dbhr"]),
        name="bioxas-m2-pre-dbhr",
        description="Stage 2: optimize M2 delivery at PreDBHRScreen.",
        experiment_type="xrt-bioxas-staged",
    )
    sample_agent = Agent(
        sensors=[devices.detector],
        dofs=[m1_radius, m1_pitch, m2_radius, m2_pitch, dbhr1_pitch, dbhr1_roll, dbhr2_pitch, dbhr2_roll],
        objectives=[Objective(name="sample_score", minimize=False)],
        evaluation_function=SampleEvaluation(tiled_client, targets["sample"]),
        name="bioxas-sample-local-global",
        description="Stages 3 and 4: local sample optimization followed by global fine tuning.",
        experiment_type="xrt-bioxas-staged",
    )

    return StagedAgents(
        m1_agent=m1_agent,
        m2_agent=m2_agent,
        sample_agent=sample_agent,
        m1_radius=m1_radius,
        m1_pitch=m1_pitch,
        m2_radius=m2_radius,
        m2_pitch=m2_pitch,
        dbhr1_pitch=dbhr1_pitch,
        dbhr1_roll=dbhr1_roll,
        dbhr2_pitch=dbhr2_pitch,
        dbhr2_roll=dbhr2_roll,
    )


def run_stage(
    run_engine: RunEngine,
    backend: XRTBIOXASBackend,
    *,
    target: str,
    agent: Agent,
    iterations: int,
    n_points: int,
) -> None:
    """Set the active diagnostic screen and run an optimization stage."""
    backend.target = target
    fixed_dofs = agent.fixed_dofs or {}
    fixed_names = set(fixed_dofs)
    free_names = [actuator.name for actuator in agent.actuators if actuator.name not in fixed_names]
    logger.info("%s free DOFs: %s", target, free_names)
    if fixed_dofs:
        logger.info("%s fixed DOFs: %s", target, fixed_dofs)
    if iterations <= 0:
        logger.info("Skipping %s because iterations=%s", target, iterations)
        return
    logger.info("Running %s for %s iteration(s) with %s point(s) per iteration", target, iterations, n_points)
    run_engine(agent.optimize(iterations=iterations, n_points=n_points))


def acquire_stage_baseline(
    devices: BeamlineDevices,
    backend: XRTBIOXASBackend,
    *,
    label: str,
    target: str,
    stage_name: str,
    stage_target: StageTarget,
    agent: Agent,
) -> StageBest:
    """Acquire the current point and ingest it into the stage agent as a baseline."""
    backend.target = target
    parameters = current_parameterization(agent)
    logger.info("Acquiring %s baseline at %s with %s", label, target, parameters)
    measurement = measure_current_beam(
        devices, label=f"{label} baseline", target=target, stage_name=stage_name, stage_target=stage_target
    )
    metrics = measurement_metrics(measurement, stage_target)
    agent.ingest([{**parameters, **metrics}])
    baseline = StageBest(parameters=parameters, metrics=metrics)
    summarize_best(f"{label} baseline", baseline)
    return baseline


def best_after_stage(label: str, agent: Agent, iterations: int) -> StageBest | None:
    """Return and log an agent's best point when its stage has run."""
    if iterations <= 0:
        logger.info("Skipping %s best-point lookup because the stage did not run", label)
        return None

    best = _best_from_agent(agent)
    summarize_best(label, best)
    return best


def configure_sample_local_stage(agents: StagedAgents, m1_best: StageBest, m2_best: StageBest) -> None:
    """Hold upstream mirror settings fixed for the local sample stage."""
    agents.sample_agent.fixed_dofs = {
        agents.m1_radius: m1_best.parameters[agents.m1_radius.parameter_name],
        agents.m1_pitch: m1_best.parameters[agents.m1_pitch.parameter_name],
        agents.m2_radius: m2_best.parameters[agents.m2_radius.parameter_name],
        agents.m2_pitch: m2_best.parameters[agents.m2_pitch.parameter_name],
    }


def configure_sample_global_stage(agents: StagedAgents, sample_best: StageBest) -> None:
    """Tighten sample-agent bounds around the best staged point and release fixed DOFs."""
    best = sample_best.parameters
    agents.sample_agent.reconfigure_search_space(
        {
            agents.m1_radius: _relative_window(
                best[agents.m1_radius.parameter_name], TIGHT_RADIUS_HALF_WIDTH, MIRROR1_RADIUS_BOUNDS
            ),
            agents.m1_pitch: _clipped_window(
                best[agents.m1_pitch.parameter_name], TIGHT_MIRROR_PITCH_HALF_WIDTH, MIRROR_EXTRA_PITCH_BOUNDS
            ),
            agents.m2_radius: _relative_window(
                best[agents.m2_radius.parameter_name], TIGHT_RADIUS_HALF_WIDTH, MIRROR2_RADIUS_BOUNDS
            ),
            agents.m2_pitch: _clipped_window(
                best[agents.m2_pitch.parameter_name], TIGHT_MIRROR_PITCH_HALF_WIDTH, MIRROR_EXTRA_PITCH_BOUNDS
            ),
            agents.dbhr1_pitch: _clipped_window(
                best[agents.dbhr1_pitch.parameter_name], TIGHT_DBHR_PITCH_HALF_WIDTH, DBHR_EXTRA_PITCH_BOUNDS
            ),
            agents.dbhr1_roll: _clipped_window(
                best[agents.dbhr1_roll.parameter_name], TIGHT_DBHR_ROLL_HALF_WIDTH, DBHR_EXTRA_ROLL_BOUNDS
            ),
            agents.dbhr2_pitch: _clipped_window(
                best[agents.dbhr2_pitch.parameter_name], TIGHT_DBHR_PITCH_HALF_WIDTH, DBHR_EXTRA_PITCH_BOUNDS
            ),
            agents.dbhr2_roll: _clipped_window(
                best[agents.dbhr2_roll.parameter_name], TIGHT_DBHR_ROLL_HALF_WIDTH, DBHR_EXTRA_ROLL_BOUNDS
            ),
        }
    )
    agents.sample_agent.fixed_dofs = None


def summarize_best(label: str, best: StageBest) -> None:
    """Log the best parameterization and primary metrics from a stage."""
    logger.info("%s best parameters: %s", label, best.parameters)
    logger.info("%s best metrics: %s", label, best.metrics)


def run_workflow(args: argparse.Namespace) -> None:
    """Run the staged BioXAS optimization workflow."""
    runtime = setup_run_engine(args.storage)
    devices = setup_devices(args.storage)

    logger.info("Available detector targets: %s", devices.backend.available_targets)
    if args.setup_only:
        logger.info("Setup succeeded; exiting before ray tracing.")
        return

    if args.target_mode != "nominal":
        raise ValueError(f"Unsupported target mode: {args.target_mode!r}")

    targets = measure_nominal_targets(devices)
    agents = build_agents(devices, runtime.tiled_client, targets)
    log_agent_dofs(agents)

    apply_initial_perturbation(devices, args.perturb_initial)

    initial_sample = measure_current_beam(
        devices,
        label="Initial sample",
        target="SampleScreen",
        stage_name="sample",
        stage_target=targets["sample"],
    )

    if args.measure_only:
        measure_current_beam(
            devices,
            label="Initial PreM2",
            target="PreM2Screen",
            stage_name="pre_m2",
            stage_target=targets["pre_m2"],
        )
        measure_current_beam(
            devices,
            label="Initial PreDBHR",
            target="PreDBHRScreen",
            stage_name="pre_dbhr",
            stage_target=targets["pre_dbhr"],
        )
        logger.info("Measure-only mode complete; exiting before optimization.")
        return

    m1_baseline = None
    if args.m1_iterations > 0:
        m1_baseline = acquire_stage_baseline(
            devices,
            devices.backend,
            label="M1",
            target="PreM2Screen",
            stage_name="pre_m2",
            stage_target=targets["pre_m2"],
            agent=agents.m1_agent,
        )
    run_stage(
        runtime.run_engine,
        devices.backend,
        target="PreM2Screen",
        agent=agents.m1_agent,
        iterations=args.m1_iterations,
        n_points=args.n_points,
    )
    m1_best = best_after_stage("M1", agents.m1_agent, args.m1_iterations)
    if m1_best is not None:
        if m1_baseline is not None:
            log_stagebest_improvement("M1", m1_baseline, m1_best, "pre_m2_score", "pre_m2")
        runtime.run_engine(agents.m1_agent.navigate_to_best(m1_best.parameters))

    m2_baseline = None
    if args.m2_iterations > 0:
        m2_baseline = acquire_stage_baseline(
            devices,
            devices.backend,
            label="M2",
            target="PreDBHRScreen",
            stage_name="pre_dbhr",
            stage_target=targets["pre_dbhr"],
            agent=agents.m2_agent,
        )
    run_stage(
        runtime.run_engine,
        devices.backend,
        target="PreDBHRScreen",
        agent=agents.m2_agent,
        iterations=args.m2_iterations,
        n_points=args.n_points,
    )
    m2_best = best_after_stage("M2", agents.m2_agent, args.m2_iterations)
    if m2_best is not None:
        if m2_baseline is not None:
            log_stagebest_improvement("M2", m2_baseline, m2_best, "pre_dbhr_score", "pre_dbhr")
        runtime.run_engine(agents.m2_agent.navigate_to_best(m2_best.parameters))

    if args.sample_local_iterations > 0 and (m1_best is None or m2_best is None):
        raise RuntimeError("Local sample optimization requires completed M1 and M2 stages.")

    if m1_best is not None and m2_best is not None:
        configure_sample_local_stage(agents, m1_best, m2_best)

    sample_local_baseline = None
    if args.sample_local_iterations > 0:
        sample_local_baseline = acquire_stage_baseline(
            devices,
            devices.backend,
            label="Local sample",
            target="SampleScreen",
            stage_name="sample",
            stage_target=targets["sample"],
            agent=agents.sample_agent,
        )
    run_stage(
        runtime.run_engine,
        devices.backend,
        target="SampleScreen",
        agent=agents.sample_agent,
        iterations=args.sample_local_iterations,
        n_points=args.n_points,
    )
    sample_local_best = best_after_stage("Local sample", agents.sample_agent, args.sample_local_iterations)
    if sample_local_best is not None:
        if sample_local_baseline is not None:
            log_stagebest_improvement("Local sample", sample_local_baseline, sample_local_best, "sample_score", "sample")
        runtime.run_engine(agents.sample_agent.navigate_to_best(sample_local_best.parameters))

    if args.global_iterations > 0 and sample_local_best is None:
        raise RuntimeError("Global sample optimization requires a completed local sample stage.")

    if sample_local_best is not None:
        configure_sample_global_stage(agents, sample_local_best)
    sample_global_baseline = None
    if args.global_iterations > 0:
        sample_global_baseline = acquire_stage_baseline(
            devices,
            devices.backend,
            label="Global sample",
            target="SampleScreen",
            stage_name="sample",
            stage_target=targets["sample"],
            agent=agents.sample_agent,
        )
    run_stage(
        runtime.run_engine,
        devices.backend,
        target="SampleScreen",
        agent=agents.sample_agent,
        iterations=args.global_iterations,
        n_points=args.n_points,
    )
    sample_global_best = best_after_stage("Global sample", agents.sample_agent, args.global_iterations)
    if sample_global_best is not None:
        if sample_global_baseline is not None:
            log_stagebest_improvement("Global sample", sample_global_baseline, sample_global_best, "sample_score", "sample")
        runtime.run_engine(agents.sample_agent.navigate_to_best(sample_global_best.parameters))

    final_best = sample_global_best or sample_local_best
    if final_best is not None:
        set_parameterization(devices, final_best.parameters)
        final_sample = measure_current_beam(
            devices,
            label="Final sample",
            target="SampleScreen",
            stage_name="sample",
            stage_target=targets["sample"],
        )
        log_measurement_improvement("Final sample", initial_sample, final_sample)

    if args.plot_final:
        if sample_global_best is None:
            raise RuntimeError("--plot-final requires a completed global sample stage.")
        devices.backend.target = "SampleScreen"
        uid = runtime.run_engine(agents.sample_agent.sample_suggestions([sample_global_best.parameters]))
        image = runtime.tiled_client[uid[0]]["primary/det_image"].read().squeeze()
        plt.imshow(image, origin="lower")
        plt.colorbar()
        plt.title("Final sample-screen image")
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage", default=DETECTOR_STORAGE, help="Directory used for simulated detector HDF5 files.")
    parser.add_argument("--n-points", type=int, default=1, help="Number of suggestions per optimization iteration.")
    parser.add_argument("--m1-iterations", type=int, default=1, help="Iterations for the PreM2/M1 stage.")
    parser.add_argument("--m2-iterations", type=int, default=1, help="Iterations for the PreDBHR/M2 stage.")
    parser.add_argument("--sample-local-iterations", type=int, default=1, help="Iterations for local sample optimization.")
    parser.add_argument("--global-iterations", type=int, default=1, help="Iterations for final global sample optimization.")
    parser.add_argument(
        "--setup-only", action="store_true", help="Construct devices and agents, then exit before ray tracing."
    )
    parser.add_argument(
        "--plot-final", action="store_true", help="Acquire and show the final sample image after optimization."
    )
    parser.add_argument(
        "--perturb-initial",
        default="none",
        choices=sorted(PERTURBATION_PRESETS),
        help="Apply a named starting-point perturbation before measuring baselines.",
    )
    parser.add_argument(
        "--target-mode",
        default="nominal",
        choices=["nominal"],
        help="How to define beam targets. Currently only nominal unperturbed targets are supported.",
    )
    parser.add_argument(
        "--measure-only", action="store_true", help="Measure initial diagnostics, then exit before optimization."
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser


def main() -> None:
    """Parse arguments and run the workflow."""
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s:%(name)s:%(message)s", force=True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    run_workflow(args)


if __name__ == "__main__":
    main()
