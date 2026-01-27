"""
staging package

This package provides a strict, file-backed "stage3/stage4" recording subsystem.

Stable public API:
  - StagingIdentity
  - StageRecorder (alias: StagingRecorder)

CLI utilities:
  - python -m staging.smoke
  - python -m staging.validate
  - python -m staging.acceptance
"""

from .recorder import StagingIdentity, StageRecorder, StagingRecorder, RunIdentity

__all__ = [
    "StagingIdentity",
    "StageRecorder",
    "StagingRecorder",
    "RunIdentity",
]
