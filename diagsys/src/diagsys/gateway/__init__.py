"""Data sources (interface I1): files and a simulator now, MQTT/OPC UA later. Always read-only."""

from .simulator import InjectedDirt, SimConfig, SimResult, simulate, write_simulation

__all__ = ["InjectedDirt", "SimConfig", "SimResult", "simulate", "write_simulation"]
