"""
staging — Comprehensive Stage-3/4 Recording System

This package provides complete data recording for the MiniShip MARL environment,
enabling analysis of:
  - AIS communication parameters and effects
  - PF tracking accuracy vs ground truth
  - RL agent behavior and performance

Core Classes:
  - StagingIdentity / RunIdentity: Identity contract for multi-worker alignment
  - StageRecorder / StagingRecorder: Main recording class

Data Classes:
  - ShipStateSnapshot: Per-ship state at a given time
  - StepRLData: RL data for a single agent at a single step
  - PFEstimate: PF tracking estimate with error metrics

CLI utilities:
  - python -m staging.smoke      # Run smoke test
  - python -m staging.validate   # Validate output directory
"""

from .recorder import (
    StagingIdentity,
    StageRecorder,
    StagingRecorder,  # alias
    RunIdentity,      # alias
    ShipStateSnapshot,
    StepRLData,
    PFEstimate,
)

__all__ = [
    "StagingIdentity",
    "StageRecorder",
    "StagingRecorder",
    "RunIdentity",
    "ShipStateSnapshot",
    "StepRLData",
    "PFEstimate",
]
