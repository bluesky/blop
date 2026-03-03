"""Ophyd-async device exports for blop_sim."""

from .detector import DetectorDevice
from .slit import SlitDevice

__all__ = [
    "DetectorDevice",
    "SlitDevice",
]
