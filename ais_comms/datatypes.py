from __future__ import annotations

"""
================================================================================
ais_comms.datatypes — SEMANTICS CONTRACT (DO NOT CHANGE LIGHTLY)
================================================================================

This module defines the *shared* datatypes used by:
  - AISCommsSim (ais_comms.py)
  - AISTrackManagerPF (track_manager_pf.py)
  - MiniShipAISCommsEnv wrapper (miniship/envs/_ais_comms_env_impl.py)
  - Staging writers (stage3/4)

Key angle conventions used in this project:

(A1) reported_cog / yaw_east_ccw_rad:
    - Meaning: direction of velocity vector in ENU/world frame.
    - 0 rad at +x (East), positive counter-clockwise.
    - Range: (-pi, pi], Unit: radians.
    - In code: RxMsg.reported_cog, TrueState.yaw_east_ccw_rad (alias: yaw_sim_rad)

This is NOT the nautical COG convention (North=0, clockwise, degrees).

Time conventions:
  - t: simulation time in seconds (float), consistent across env, AISCommsSim, and PF.
================================================================================
"""

from dataclasses import dataclass
from typing import Any, Dict
import math

ShipId = int
AgentId = str
Ts = float


@dataclass
class RawTxMsg:
    """
    Raw message emitted by a ship's AIS transmitter (before channel effects).

    Fields:
      - msg_id: unique message identifier
      - tx_ship: transmitting ship id
      - mmsi: MMSI number
      - tx_ts_true: true transmission timestamp (seconds)
      - x, y: position (ENU/world coordinates)
      - sog: speed over ground (m/s)
      - cog: course over ground (rad, ENU convention)
      - rot: rate of turn (rad/s, positive=turning left/CCW, per CTRV model)
             IMO AIS encodes rot as AIS_ROT = 4.733 * sqrt(|rot_deg_min|) * sign(rot)
             but here we keep raw rad/s for internal use.
      - nav_status: navigation status code (0-15 per ITU-R M.1371)
             0=underway using engine, 1=at anchor, 2=not under command,
             3=restricted maneuverability, 4=constrained by draught,
             5=moored, 6=aground, 7=fishing, 8=sailing, 15=undefined
    """
    msg_id: str
    tx_ship: ShipId
    mmsi: int
    tx_ts_true: Ts
    x: float
    y: float
    sog: float
    cog: float  # yaw_east_ccw_rad (ENU, rad)
    rot: float = 0.0  # rate of turn (rad/s)
    nav_status: int = 0  # navigation status (0=underway using engine)
    meta: Dict[str, Any] | None = None

    # Legacy alias for ship_id
    @property
    def ship_id(self) -> ShipId:
        return self.tx_ship

    @property
    def t_tx(self) -> Ts:
        return self.tx_ts_true


@dataclass(frozen=True, slots=True)
class RxMsg:
    """
    Received AIS message after channel effects (drop/delay/noise).

    reported_* fields are the *received* values (may be noisy).

    Additional fields:
      - reported_rot: rate of turn (rad/s, may have noise)
      - reported_nav_status: navigation status code (0-15, passed through)
    """
    msg_id: str
    rx_agent: AgentId
    mmsi: int
    reported_x: float
    reported_y: float
    reported_sog: float
    reported_cog: float  # yaw_east_ccw_rad (ENU, rad)
    reported_rot: float  # rate of turn (rad/s)
    reported_nav_status: int  # navigation status (0-15)
    reported_ts: Ts
    arrival_time: Ts
    age: float
    meta: Dict[str, Any] | None = None

    # Legacy aliases
    @property
    def ship_id(self) -> ShipId:
        return self.mmsi

    @property
    def arrival_t(self) -> Ts:
        return self.arrival_time

    @property
    def reported_t(self) -> Ts:
        return self.reported_ts


@dataclass(frozen=True, slots=True)
class TrueState:
    """
    Ground-truth kinematic state used by AISCommsSim and PF for evaluation.

    Required fields:
      - x, y, vx, vy are ENU/world coordinates.
      - yaw_east_ccw_rad is atan2(vy, vx) in ENU (East=0, CCW+), unit rad.
      - t is simulation time (seconds).
      - rot: rate of turn (rad/s), optional, defaults to 0.0 for straight motion.
             Positive = turning left (CCW), per CTRV model convention.
      - nav_status: navigation status code (0-15), defaults to 0 (underway using engine).
    """
    ship_id: ShipId
    t: Ts
    x: float
    y: float
    vx: float
    vy: float
    yaw_east_ccw_rad: float
    rot: float = 0.0  # rate of turn (rad/s)
    nav_status: int = 0  # navigation status code

    @property
    def yaw_sim_rad(self) -> float:
        return self.yaw_east_ccw_rad

    @property
    def sog(self) -> float:
        """Speed over ground (m/s), derived from vx, vy."""
        return math.hypot(self.vx, self.vy)

    @property
    def cog(self) -> float:
        """Course over ground (rad, ENU convention: East=0, CCW+)."""
        return self.yaw_east_ccw_rad


@dataclass(frozen=True, slots=True)
class ShipState:
    """
    Convenience state container used by environment wrapper.

    Note: this is not required by AISCommsSim; it exists to keep env code explicit.
    """
    ship_id: ShipId
    x: float
    y: float
    vx: float
    vy: float
    yaw_east_ccw_rad: float

    @property
    def yaw_sim_rad(self) -> float:
        return self.yaw_east_ccw_rad


@dataclass(frozen=True, slots=True)
class WorldState:
    """
    Convenience world snapshot: mapping ship_id -> TrueState.
    """
    t: Ts
    ships: Dict[ShipId, TrueState]
