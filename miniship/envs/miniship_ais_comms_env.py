"""
MiniShip AISComms environment (PettingZoo ParallelEnv).

This module is intentionally kept small.
The full implementation lives in `_ais_comms_env_impl.py`.
"""

from __future__ import annotations

from ._ais_comms_env_impl import MiniShipAISCommsEnv

__all__ = ["MiniShipAISCommsEnv"]
