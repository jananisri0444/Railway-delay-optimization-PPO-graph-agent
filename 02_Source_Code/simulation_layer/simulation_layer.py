"""
Simulation Layer
=================
This layer's ONLY job is to faithfully simulate railway operations and
produce, tick by tick, the current **Railway State**. It does not build
any graph and does not know anything about GraphSAGE or PPO -- that is
the responsibility of the downstream **DORG Builder** / learning layer,
which consumes `RailwayState` objects produced here.

Modules (mirrors the architecture diagram 1:1)
-----------------------------------------------
    Simulation Clock            -> SimulationClock
    Train Position Updater      -> TrainPositionUpdater
    Track Occupancy Manager     -> TrackOccupancyManager
    Platform Occupancy Manager  -> PlatformOccupancyManager
    Loop Line Manager           -> LoopLineManager
    Conflict Detector           -> ConflictDetector
    Delay Propagation Engine    -> DelayPropagationEngine
    Disruption Injector         -> DisruptionInjector
    Railway State Manager       -> RailwayStateManager

Orchestrator
------------
    SimulationLayer  -- owns one instance of every module above and runs
    them, tick after tick, in exactly the order shown in the "Simulation
    Flow" diagram:

        Simulation Clock
              |
              v
        Train Position Updater
              |
              v
        Track Occupancy Manager
              |
              v
        Platform Occupancy Manager
              |
              v
        Loop Line Manager
              |
              v
        Disruption Injector
              |
              v
        Conflict Detector
              |
              v
        Delay Propagation Engine
              |
              v
        Railway State Manager
              |
              v
        Current Railway State

`SimulationLayer.step()` returns the `RailwayState` snapshot for that
tick. That snapshot (and nothing else -- no graphs, no adjacency lists)
is the hand-off point to the DORG Builder.

Notes on assumptions
---------------------
No dedicated "block section" dataset was provided (only per-station
platform/loop-line counts in `loop_line_data.csv`). The Track Occupancy
Manager therefore treats each pair of *adjacent* stations along the
corridor (per `StationDB.ordered_codes`) as one track section, with a
configurable default capacity (`SECTION_CAPACITY`, default 2 -- i.e. a
double line, one train per direction). This is documented here so it can
be swapped for real block-section data later without touching any other
module.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data_layer.data_layer import (
    TimetableDB, StationDB, PlatformInfo, TrainPriorityInfo,
    DisruptionGenerator, DisruptionEvent, TrainSchedule, load_all,
)


# ==========================================================================
# Shared state
# ==========================================================================
class TrainStatus(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    IN_TRANSIT = "IN_TRANSIT"
    HALTED = "HALTED"          # occupying a platform
    HELD_LOOP = "HELD_LOOP"    # held on a loop line (overtaking / awaiting a platform)
    WAITING = "WAITING"        # wants a platform/loop line but none is free
    TERMINATED = "TERMINATED"


@dataclass
class TrainState:
    train_number: int
    status: TrainStatus = TrainStatus.NOT_STARTED
    station_code: Optional[str] = None      # set if HALTED / HELD_LOOP / WAITING
    from_station: Optional[str] = None      # set if IN_TRANSIT
    to_station: Optional[str] = None        # set if IN_TRANSIT
    position_km: float = 0.0
    speed_kmph: float = 0.0
    delay_min: float = 0.0
    stop_idx: int = 0                       # index into schedule.stops
    platform_id: Optional[int] = None       # allocated by PlatformOccupancyManager
    loop_line_id: Optional[int] = None      # allocated by LoopLineManager
    waiting_since_min: Optional[float] = None


Section = Tuple[str, str]   # (station_a, station_b), corridor-adjacent pair


def section_key(a: str, b: str) -> str:
    """Canonical (direction-agnostic) key for a track section."""
    lo, hi = sorted((a, b))
    return f"{lo}-{hi}"


# ==========================================================================
# 1. Simulation Clock
# ==========================================================================
class SimulationClock:
    """Advances the simulation one tick at a time between a start and end time."""

    def __init__(self, tick_min: float = 1.0):
        self.tick_min = tick_min
        self.start_time_min: float = 0.0
        self.end_time_min: Optional[float] = None
        self.current_time_min: float = 0.0

    def reset(self, start_time_min: float = 0.0, end_time_min: Optional[float] = None) -> float:
        self.start_time_min = start_time_min
        self.end_time_min = end_time_min
        self.current_time_min = start_time_min
        return self.current_time_min

    def tick(self) -> float:
        """Advance the clock by one tick and return the new current time."""
        self.current_time_min += self.tick_min
        return self.current_time_min

    def now(self) -> float:
        return self.current_time_min

    def is_finished(self) -> bool:
        return self.end_time_min is not None and self.current_time_min > self.end_time_min


# ==========================================================================
# 2. Train Position Updater
# ==========================================================================
class TrainPositionUpdater:
    """
    Updates train positions, arrivals, departures, running speed, and dwell
    time based on the timetable.

    Each train's schedule times are its nominal timeline. A train's
    accumulated `delay_min` shifts it backward on that timeline -- at sim
    time t, a train with delay D behaves as though the schedule clock reads
    (t - D). Position is interpolated linearly between consecutive stops;
    running speed is derived from that interpolation.
    """

    def __init__(self, timetable: TimetableDB, platform_info: Optional[PlatformInfo] = None):
        self.timetable = timetable
        self.platform_info = platform_info

    def _has_hold_capacity(self, station_code: str) -> bool:
        """False for technical points (signal cabins, etc.) that have zero
        platforms AND zero loop lines -- there is physically nowhere for a
        train to be held there, so it must never be asked to halt/wait."""
        if self.platform_info is None:
            return True
        return self.platform_info.capacity_of(station_code) > 0

    def update(self, state: TrainState, sim_time_min: float) -> TrainState:
        schedule = self.timetable.get(state.train_number)
        eff_time = sim_time_min - state.delay_min

        if eff_time < schedule.origin_min:
            state.status = TrainStatus.NOT_STARTED
            state.speed_kmph = 0.0
            return state
        if eff_time >= schedule.terminus_min:
            state.status = TrainStatus.TERMINATED
            state.station_code = schedule.stops[-1].station_code
            state.from_station = state.to_station = None
            state.speed_kmph = 0.0
            return state

        stops = schedule.stops
        for i in range(len(stops) - 1):
            cur, nxt = stops[i], stops[i + 1]
            if cur.scheduled_arrival_min <= eff_time <= cur.scheduled_departure_min:
                genuine_halt = cur.is_halt and self._has_hold_capacity(cur.station_code)
                if not genuine_halt:
                    # Technical point / no infrastructure to hold a train --
                    # pass straight through without ever requesting a
                    # platform or loop line (prevents a permanent WAITING
                    # deadlock at zero-capacity signal cabins).
                    span = nxt.scheduled_arrival_min - cur.scheduled_departure_min
                    dist = nxt.distance_from_source_km - cur.distance_from_source_km
                    state.status = TrainStatus.IN_TRANSIT
                    state.from_station = cur.station_code
                    state.to_station = nxt.station_code
                    state.station_code = None
                    state.position_km = cur.distance_from_source_km
                    state.speed_kmph = 0.0 if span <= 0 else (dist / span) * 60.0
                    state.stop_idx = i
                    return state
                # dwelling at this stop -- requests a platform this tick.
                # (Whether it actually gets one is decided downstream by the
                # Platform Occupancy Manager / Loop Line Manager.)
                if state.status not in (TrainStatus.WAITING, TrainStatus.HELD_LOOP):
                    state.status = TrainStatus.HALTED
                state.station_code = cur.station_code
                state.from_station = state.to_station = None
                state.position_km = cur.distance_from_source_km
                state.speed_kmph = 0.0
                state.stop_idx = i
                return state
            if cur.scheduled_departure_min < eff_time < nxt.scheduled_arrival_min:
                # in transit between cur and nxt
                span = nxt.scheduled_arrival_min - cur.scheduled_departure_min
                frac = 0.0 if span <= 0 else (eff_time - cur.scheduled_departure_min) / span
                dist = nxt.distance_from_source_km - cur.distance_from_source_km
                state.status = TrainStatus.IN_TRANSIT
                state.from_station = cur.station_code
                state.to_station = nxt.station_code
                state.station_code = None
                state.position_km = cur.distance_from_source_km + frac * dist
                state.speed_kmph = 0.0 if span <= 0 else (dist / span) * 60.0
                state.stop_idx = i
                return state
        return state


# ==========================================================================
# 3. Track Occupancy Manager
# ==========================================================================
class TrackOccupancyManager:
    """
    Tracks which train occupies each track section (the segment of line
    between two corridor-adjacent stations) and releases sections once a
    train is no longer travelling through them.

    See module docstring for the "no dedicated block-section dataset"
    assumption behind `SECTION_CAPACITY`.
    """

    SECTION_CAPACITY = 2  # trains that may simultaneously occupy one section

    def __init__(self, stations: StationDB, section_capacity: Optional[int] = None):
        self.stations = stations
        self.default_capacity = section_capacity or self.SECTION_CAPACITY
        self.sections: List[Section] = list(
            zip(stations.ordered_codes[:-1], stations.ordered_codes[1:])
        )
        self.blocked_sections: Dict[str, float] = {}   # section_key -> expiry tick (fully closed)
        self.speed_restrictions: Dict[str, Tuple[float, float]] = {}  # key -> (factor, expiry)
        self._occupancy: Dict[str, List[int]] = {}      # section_key -> [train_numbers]

    def reset(self):
        self.blocked_sections.clear()
        self.speed_restrictions.clear()
        self._occupancy.clear()

    def capacity_of(self, key: str) -> int:
        return 0 if key in self.blocked_sections else self.default_capacity

    def block_section(self, station_a: str, station_b: str, expiry_min: float):
        self.blocked_sections[section_key(station_a, station_b)] = expiry_min

    def restrict_speed(self, station_a: str, station_b: str, factor: float, expiry_min: float):
        self.speed_restrictions[section_key(station_a, station_b)] = (factor, expiry_min)

    def release_expired(self, sim_time_min: float):
        self.blocked_sections = {
            k: exp for k, exp in self.blocked_sections.items() if sim_time_min < exp
        }
        self.speed_restrictions = {
            k: v for k, v in self.speed_restrictions.items() if sim_time_min < v[1]
        }

    def update(self, states: Dict[int, TrainState], sim_time_min: float) -> Dict[str, List[int]]:
        """Rebuild the occupancy table from current train positions (allocates
        trains newly in transit onto a section, and releases trains that have
        arrived / are no longer travelling that section)."""
        self.release_expired(sim_time_min)

        occupancy: Dict[str, List[int]] = {}
        for tn, st in states.items():
            if st.status != TrainStatus.IN_TRANSIT or not (st.from_station and st.to_station):
                continue
            key = section_key(st.from_station, st.to_station)
            occupancy.setdefault(key, []).append(tn)

            # apply any active speed restriction to the train's running speed
            factor, _ = self.speed_restrictions.get(key, (1.0, 0))
            if factor < 1.0:
                st.speed_kmph *= factor

        self._occupancy = occupancy
        return occupancy

    def occupancy_table(self) -> Dict[str, List[int]]:
        return self._occupancy


# ==========================================================================
# 4. Platform Occupancy Manager
# ==========================================================================
class PlatformOccupancyManager:
    """
    Allocates and releases individual station platforms according to train
    arrivals and departures. Trains that persist at a station keep the
    platform they were already given; new arrivals are granted free
    platforms in priority order. Trains that cannot be given a platform are
    returned as overflow for the Loop Line Manager to try to place.
    """

    def __init__(self, platform_info: PlatformInfo, priority_info: TrainPriorityInfo):
        self.platform_info = platform_info
        self.priority_info = priority_info
        self.blocked_platforms: Dict[str, int] = {}   # station -> platforms lost to disruption
        self._occupancy: Dict[str, Dict[int, int]] = {}  # station -> {platform_id: train_number}
        # One-tick agent preferences. A GRANT_PLATFORM request moves a waiting
        # train ahead of neutral requests. A GRANT_LOOP request deliberately
        # moves it behind neutral platform requests so it can overflow to a loop.
        self.agent_platform_requests: set[tuple[int, str]] = set()
        self.agent_loop_requests: set[tuple[int, str]] = set()

    def set_agent_requests(self, platform_requests=None, loop_requests=None):
        self.agent_platform_requests = set(platform_requests or ())
        self.agent_loop_requests = set(loop_requests or ())

    def reset(self):
        self.blocked_platforms.clear()
        self._occupancy.clear()
        self.agent_platform_requests.clear()
        self.agent_loop_requests.clear()

    def block_station(self, station_code: str, count: int = 1):
        self.blocked_platforms[station_code] = self.blocked_platforms.get(station_code, 0) + count

    def unblock_station(self, station_code: str, count: int = 1):
        if station_code in self.blocked_platforms:
            self.blocked_platforms[station_code] = max(0, self.blocked_platforms[station_code] - count)

    def capacity_of(self, station_code: str) -> int:
        base = self.platform_info.get(station_code).platforms if station_code in self.platform_info.capacity else 1
        return max(0, base - self.blocked_platforms.get(station_code, 0))

    def allocate(
        self, states: Dict[int, TrainState]
    ) -> Tuple[Dict[str, Dict[int, int]], Dict[str, List[int]]]:
        """
        Returns (platform_occupancy_table, overflow) where overflow maps
        station -> [train_numbers] that requested a platform but none was
        free (fed to the Loop Line Manager next).
        """
        # release platforms held by trains that have left the station
        for tn, st in states.items():
            if st.platform_id is not None and st.status not in (TrainStatus.HALTED,):
                st.platform_id = None

        requesting: Dict[str, List[int]] = {}
        for tn, st in states.items():
            if st.status in (TrainStatus.HALTED, TrainStatus.WAITING, TrainStatus.HELD_LOOP):
                requesting.setdefault(st.station_code, []).append(tn)

        occupancy: Dict[str, Dict[int, int]] = {}
        overflow: Dict[str, List[int]] = {}

        for station, train_numbers in requesting.items():
            cap = self.capacity_of(station)
            table: Dict[int, int] = {}

            # trains that already hold a platform at this station keep it
            # A GRANT_LOOP request on a train that is sitting on a platform makes it
            # YIELD: it gives the platform up (and is pushed to a loop line) so a
            # waiting / loop-held train can take the platform.
            for tn in train_numbers:
                if states[tn].platform_id is not None and (tn, station) in self.agent_loop_requests:
                    states[tn].platform_id = None
            already = [tn for tn in train_numbers if states[tn].platform_id is not None]
            rest = [tn for tn in train_numbers if states[tn].platform_id is None]
            # Agent requests are a priority hint, not a capacity bypass.
            # platform request -> first; neutral/HOLD -> middle; loop request -> last.
            def platform_order(tn: int):
                key = (tn, station)
                if key in self.agent_platform_requests:
                    agent_rank = 0
                elif key in self.agent_loop_requests:
                    agent_rank = 2
                else:
                    agent_rank = 1
                return (agent_rank, self.priority_info.get(tn))

            rest.sort(key=platform_order)

            for tn in already:
                pid = states[tn].platform_id
                if len(table) < cap:
                    table[pid] = tn
                    states[tn].status = TrainStatus.HALTED
                else:
                    states[tn].platform_id = None
                    rest.append(tn)

            used_ids = set(table.keys())
            next_id = 1
            for tn in rest:
                if len(table) >= cap:
                    break
                while next_id in used_ids:
                    next_id += 1
                table[next_id] = tn
                used_ids.add(next_id)
                states[tn].platform_id = next_id
                states[tn].status = TrainStatus.HALTED

            placed = set(table.values())
            leftover = [tn for tn in train_numbers if tn not in placed]
            if leftover:
                overflow[station] = leftover
            if table:
                occupancy[station] = table

        self._occupancy = occupancy
        return occupancy, overflow

    def occupancy_table(self) -> Dict[str, Dict[int, int]]:
        return self._occupancy


# ==========================================================================
# 5. Loop Line Manager
# ==========================================================================
class LoopLineManager:
    """
    Assigns loop lines to trains that could not get a platform -- used to
    hold or overtake slower/lower-priority trains when a loop line is
    available. Trains that get neither a platform nor a loop line are left
    WAITING (and accrue delay via the Delay Propagation Engine).
    """

    def __init__(self, platform_info: PlatformInfo, priority_info: TrainPriorityInfo):
        self.platform_info = platform_info
        self.priority_info = priority_info
        self.blocked_loops: Dict[str, int] = {}
        self._occupancy: Dict[str, Dict[int, int]] = {}
        self.agent_loop_requests: set[tuple[int, str]] = set()

    def set_agent_requests(self, loop_requests=None):
        self.agent_loop_requests = set(loop_requests or ())

    def reset(self):
        self.blocked_loops.clear()
        self._occupancy.clear()
        self.agent_loop_requests.clear()

    def block_station(self, station_code: str, count: int = 1):
        self.blocked_loops[station_code] = self.blocked_loops.get(station_code, 0) + count

    def unblock_station(self, station_code: str, count: int = 1):
        if station_code in self.blocked_loops:
            self.blocked_loops[station_code] = max(0, self.blocked_loops[station_code] - count)

    def capacity_of(self, station_code: str) -> int:
        cap = self.platform_info.get(station_code) if station_code in self.platform_info.capacity else None
        base = cap.loop_line_count if cap else 0
        return max(0, base - self.blocked_loops.get(station_code, 0))

    def assign(
        self, states: Dict[int, TrainState], overflow: Dict[str, List[int]]
    ) -> Tuple[Dict[str, Dict[int, int]], List[Tuple[int, str]]]:
        """
        Returns (loop_line_occupancy_table, still_waiting) where
        still_waiting is a list of (train_number, station_code) that got
        neither a platform nor a loop line this tick.
        """
        # release loop lines held by trains no longer parked in one
        for tn, st in states.items():
            if st.loop_line_id is not None and st.status != TrainStatus.HELD_LOOP:
                st.loop_line_id = None

        occupancy: Dict[str, Dict[int, int]] = {}
        still_waiting: List[Tuple[int, str]] = []

        for station, train_numbers in overflow.items():
            cap = self.capacity_of(station)
            table: Dict[int, int] = {}

            already = [tn for tn in train_numbers if states[tn].loop_line_id is not None]
            rest = [tn for tn in train_numbers if states[tn].loop_line_id is None]
            # Explicit GRANT_LOOP requests get first claim on loop capacity.
            # Otherwise retain the original behaviour: lower-priority trains
            # are more likely to be held in loops, freeing platforms for
            # higher-priority traffic.
            rest.sort(
                key=lambda tn: (
                    0 if (tn, station) in self.agent_loop_requests else 1,
                    -self.priority_info.get(tn),
                )
            )

            for tn in already:
                lid = states[tn].loop_line_id
                if len(table) < cap:
                    table[lid] = tn
                    states[tn].status = TrainStatus.HELD_LOOP
                else:
                    states[tn].loop_line_id = None
                    rest.append(tn)

            used_ids = set(table.keys())
            next_id = 1
            for tn in rest:
                if len(table) >= cap:
                    break
                while next_id in used_ids:
                    next_id += 1
                table[next_id] = tn
                used_ids.add(next_id)
                states[tn].loop_line_id = next_id
                states[tn].status = TrainStatus.HELD_LOOP

            placed = set(table.values())
            for tn in train_numbers:
                if tn not in placed:
                    states[tn].status = TrainStatus.WAITING
                    still_waiting.append((tn, station))

            if table:
                occupancy[station] = table

        self._occupancy = occupancy
        return occupancy, still_waiting

    def occupancy_table(self) -> Dict[str, Dict[int, int]]:
        return self._occupancy


# ==========================================================================
# 6. Disruption Injector
# ==========================================================================
class DisruptionInjector:
    """
    Pulls stochastic events (signal failures, maintenance blocks, speed
    restrictions, track closures) from the DisruptionGenerator each tick
    and applies their effects onto the infrastructure state -- i.e. it
    modifies the capacity/availability that the occupancy managers see,
    Direct train-delay events are intentionally excluded from this PPO-control
    environment; infrastructure disruptions are retained.
    """

    def __init__(
        self,
        generator: DisruptionGenerator,
        platform_manager: PlatformOccupancyManager,
        loop_manager: LoopLineManager,
        track_manager: TrackOccupancyManager,
    ):
        self.generator = generator
        self.platform_manager = platform_manager
        self.loop_manager = loop_manager
        self.track_manager = track_manager
        # (station_code, expiry_min, blocked_count) -- count so the right
        # number of slots gets released again on expiry, not just 1.
        self._active_platform_blocks: List[Tuple[str, float, int]] = []
        self._active_loop_blocks: List[Tuple[str, float, int]] = []
        self.log: List[DisruptionEvent] = []

    def reset(self):
        self._active_platform_blocks.clear()
        self._active_loop_blocks.clear()
        self.log.clear()

    @staticmethod
    def _compute_hot_stations(states: Dict[int, TrainState]) -> set:
        """Stations with a train actually present (HALTED/WAITING/HELD_LOOP)
        or about to arrive (IN_TRANSIT -> to_station) this tick. Built
        purely from existing live TrainState -- no new trains, no schedule
        changes -- so demo-mode disruptions can be targeted at genuine
        traffic instead of landing on an empty station."""
        hot: set = set()
        for st in states.values():
            if st.status in (TrainStatus.HALTED, TrainStatus.WAITING, TrainStatus.HELD_LOOP) and st.station_code:
                hot.add(st.station_code)
            elif st.status == TrainStatus.IN_TRANSIT and st.to_station:
                hot.add(st.to_station)
        return hot

    def inject(self, tick_min: float, states: Dict[int, TrainState]) -> List[DisruptionEvent]:
        # release expired platform/loop blocks from previous ticks
        still_active = []
        for station, expiry, count in self._active_platform_blocks:
            if tick_min >= expiry:
                self.platform_manager.unblock_station(station, count=count)
            else:
                still_active.append((station, expiry, count))
        self._active_platform_blocks = still_active

        still_active = []
        for station, expiry, count in self._active_loop_blocks:
            if tick_min >= expiry:
                self.loop_manager.unblock_station(station, count=count)
            else:
                still_active.append((station, expiry, count))
        self._active_loop_blocks = still_active

        hot_stations = self._compute_hot_stations(states)
        events = self.generator.generate(tick_min, hot_stations=hot_stations)
        for ev in events:
            # Direct injected train delays are intentionally ignored by the
            # PPO control simulation. Infrastructure disruptions remain, but
            # only delay that arises from dispatch decisions/resource
            # contention is accumulated in TrainState.delay_min.
            if ev.kind == "delay":
                continue

            if ev.kind == "platform_block" and ev.station_code:
                count = max(1, int(ev.magnitude) or 1)
                self.platform_manager.block_station(ev.station_code, count=count)
                self._active_platform_blocks.append((ev.station_code, tick_min + ev.duration_min, count))

            elif ev.kind == "loop_block" and ev.station_code:
                count = max(1, int(ev.magnitude) or 1)
                self.loop_manager.block_station(ev.station_code, count=count)
                self._active_loop_blocks.append((ev.station_code, tick_min + ev.duration_min, count))

            elif ev.kind == "track_block" and ev.section:
                self.track_manager.block_section(*ev.section, expiry_min=tick_min + ev.duration_min)

            elif ev.kind == "speed_restriction" and ev.section:
                self.track_manager.restrict_speed(
                    *ev.section, factor=ev.magnitude, expiry_min=tick_min + ev.duration_min
                )

        self.log.extend(events)
        return events

    def active_disruptions(self) -> List[dict]:
        """A lightweight view of what's currently in effect, for the Railway
        State snapshot (not the full historical log)."""
        active = []
        for station, expiry, count in self._active_platform_blocks:
            active.append({"kind": "platform_block", "station": station, "expires_min": expiry, "count": count})
        for station, expiry, count in self._active_loop_blocks:
            active.append({"kind": "loop_block", "station": station, "expires_min": expiry, "count": count})
        for key, expiry in self.track_manager.blocked_sections.items():
            active.append({"kind": "track_block", "section": key, "expires_min": expiry})
        for key, (factor, expiry) in self.track_manager.speed_restrictions.items():
            active.append({
                "kind": "speed_restriction", "section": key,
                "factor": factor, "expires_min": expiry,
            })
        return active


# ==========================================================================
# 7. Conflict Detector
# ==========================================================================
@dataclass
class Conflict:
    type: str            # "platform" | "loop_line" | "track" | "junction" | "headway"
    location: str        # station code or section key
    trains: List[int]

    def as_dict(self) -> dict:
        return {"type": self.type, "location": self.location, "trains": self.trains}


class ConflictDetector:
    """
    Detects conflicts caused by shared track sections, platforms, loop
    lines/junctions, or headway violations, from the occupancy data
    produced upstream this tick.

    A station whose name contains "Junction"/"Jn" is treated as a
    junction, so contention there is reported as a "junction" conflict
    rather than a plain "platform" conflict (no separate junction/points
    dataset was provided, so this is the closest available proxy).
    """

    HEADWAY_KM = 2.0  # minimum safe separation on a shared section

    def __init__(self, stations: StationDB):
        self.stations = stations

    def _is_junction(self, station_code: str) -> bool:
        name = self.stations.get(station_code).name if station_code in self.stations.stations else ""
        return "junction" in name.lower() or name.lower().endswith(" jn")

    def detect(
        self,
        states: Dict[int, TrainState],
        track_occupancy: Dict[str, List[int]],
        track_manager: TrackOccupancyManager,
        platform_occupancy: Dict[str, Dict[int, int]],
        loop_occupancy: Dict[str, Dict[int, int]],
        waiting: List[Tuple[int, str]],
    ) -> List[Conflict]:
        conflicts: List[Conflict] = []

        # -- shared platforms: any station where trains are WAITING despite
        #    a platform table already being full is a platform contention.
        waiting_by_station: Dict[str, List[int]] = {}
        for tn, station in waiting:
            waiting_by_station.setdefault(station, []).append(tn)

        for station, waiters in waiting_by_station.items():
            occupants = list(platform_occupancy.get(station, {}).values())
            trains = sorted(set(waiters) | set(occupants))
            if len(trains) < 2:
                continue
            ctype = "junction" if self._is_junction(station) else "platform"
            conflicts.append(Conflict(type=ctype, location=station, trains=trains))

        # -- shared loop lines: loop-line contention is already captured by
        #    the platform/junction conflict block above (a train only ends
        #    up in `waiting` after both platform *and* loop-line allocation
        #    have failed for it), so no separate pass is needed here.

        # -- shared track sections: more trains on a section than its
        #    capacity, or same-direction trains closer than the safe
        #    headway.
        for key, train_numbers in track_occupancy.items():
            cap = track_manager.capacity_of(key)
            if len(train_numbers) > cap:
                conflicts.append(Conflict(type="track", location=key, trains=sorted(train_numbers)))
                continue
            if len(train_numbers) < 2:
                continue
            for i in range(len(train_numbers)):
                for j in range(i + 1, len(train_numbers)):
                    a, b = states[train_numbers[i]], states[train_numbers[j]]
                    same_direction = a.from_station == b.from_station and a.to_station == b.to_station
                    if same_direction and abs(a.position_km - b.position_km) < self.HEADWAY_KM:
                        conflicts.append(
                            Conflict(type="headway", location=key, trains=sorted((a.train_number, b.train_number)))
                        )

        return conflicts


# ==========================================================================
# 8. Delay Propagation Engine
# ==========================================================================
class DelayPropagationEngine:
    """
    Propagates delays to subsequent stations and affected trains.

    Because `TrainPositionUpdater` drives every future stop from
    `eff_time = sim_time - delay_min`, any increase to a train's
    `delay_min` automatically pushes back *all* of that train's remaining
    stops -- this is how delay propagates "to subsequent stations" for the
    same train. This engine is responsible for:

      1. Accruing the knock-on delay a train picks up while it is
         WAITING or HELD_LOOP for a resource (so waiting itself, not just
         disruptions, becomes a delay).
      2. Propagating a small additional knock-on delay to the *other*
         train(s) involved in a detected conflict, biased towards the
         lower-priority train, to approximate real dispatch behaviour
         (the lower-priority train absorbs the hold).
    """

    WAITING_DELAY_PER_TICK = 1.0
    LOOP_HOLD_DELAY_PER_TICK = 0.5
    CONFLICT_KNOCKON_DELAY = 0.0
    # Blocking cost: while a headway/track conflict is left unresolved, every
    # higher-priority train in it is slowed by (priority gap x this) min/tick.
    # If the dispatcher HOLDs the lowest-priority train, that train yields and
    # the higher-priority trains are not penalised this tick.
    BLOCKING_DELAY_PER_PRIORITY_STEP = 0.25

    def __init__(self, tick_min: float, priority_info: TrainPriorityInfo):
        self.tick_min = tick_min
        self.priority_info = priority_info

    def propagate(
        self,
        states: Dict[int, TrainState],
        conflicts: List[Conflict],
        sim_time_min: float,
        held: Optional[set] = None,
    ) -> Dict[int, float]:
        for tn, st in states.items():
            if st.status == TrainStatus.WAITING:
                if st.waiting_since_min is None:
                    st.waiting_since_min = sim_time_min
                st.delay_min += self.WAITING_DELAY_PER_TICK
            elif st.status == TrainStatus.HELD_LOOP:
                st.delay_min += self.LOOP_HOLD_DELAY_PER_TICK
                st.waiting_since_min = None
            else:
                st.waiting_since_min = None

        held = held or set()
        # blocking delay: higher-priority trains stuck behind a lower-priority
        # one in an unresolved headway/track conflict
        for c in conflicts:
            if c.type not in ("headway", "track") or len(c.trains) < 2:
                continue
            ranked = sorted(c.trains, key=lambda tn: self.priority_info.get(tn))
            low = ranked[-1]
            if low in held:
                continue
            for tn in ranked[:-1]:
                gap = self.priority_info.get(low) - self.priority_info.get(tn)
                if gap > 0 and states[tn].status != TrainStatus.TERMINATED:
                    states[tn].delay_min += self.BLOCKING_DELAY_PER_PRIORITY_STEP * gap

        # knock-on delay for the lower-priority side of each conflict
        for c in conflicts:
            if len(c.trains) < 2:
                continue
            ranked = sorted(c.trains, key=lambda tn: self.priority_info.get(tn))
            lowest_priority_train = ranked[-1]
            if states[lowest_priority_train].status != TrainStatus.TERMINATED:
                states[lowest_priority_train].delay_min += self.CONFLICT_KNOCKON_DELAY

        return {tn: st.delay_min for tn, st in states.items()}


# ==========================================================================
# 9. Railway State Manager
# ==========================================================================
@dataclass
class RailwayState:
    simulation_time_min: float
    trains: List[dict]
    track_occupancy: Dict[str, List[int]]
    platform_occupancy: Dict[str, Dict[int, int]]
    loop_line_occupancy: Dict[str, Dict[int, int]]
    conflicts: List[dict]
    delays: Dict[int, float]
    waiting_trains: List[Tuple[int, str]]
    active_disruptions: List[dict]
    mode: str = "REALISTIC"   # "REALISTIC" | "STRESS_TEST" -- see DisruptionGenerator.demo_mode

    def as_dict(self) -> dict:
        return {
            "simulation_time_min": self.simulation_time_min,
            "trains": self.trains,
            "track_occupancy": self.track_occupancy,
            "platform_occupancy": self.platform_occupancy,
            "loop_line_occupancy": self.loop_line_occupancy,
            "conflicts": self.conflicts,
            "delays": self.delays,
            "waiting_trains": self.waiting_trains,
            "active_disruptions": self.active_disruptions,
            "mode": self.mode,
        }


class RailwayStateManager:
    """
    Aggregates the outputs of every other module into one complete
    Railway State snapshot per simulation tick. This snapshot -- and only
    this snapshot -- is what gets handed off to the DORG Builder.
    """

    def __init__(self, priority_info: TrainPriorityInfo):
        self.priority_info = priority_info

    def aggregate(
        self,
        sim_time_min: float,
        states: Dict[int, TrainState],
        track_occupancy: Dict[str, List[int]],
        platform_occupancy: Dict[str, Dict[int, int]],
        loop_line_occupancy: Dict[str, Dict[int, int]],
        conflicts: List[Conflict],
        waiting: List[Tuple[int, str]],
        disruption_injector: DisruptionInjector,
        mode: str = "REALISTIC",
    ) -> RailwayState:
        trains_snapshot = []
        for tn, st in states.items():
            if st.status == TrainStatus.NOT_STARTED:
                continue
            trains_snapshot.append({
                "train_number": tn,
                "status": st.status.value,
                "station": st.station_code,
                "from_station": st.from_station,
                "to_station": st.to_station,
                "position_km": round(st.position_km, 2),
                "speed_kmph": round(st.speed_kmph, 1),
                "delay_min": round(st.delay_min, 1),
                "platform_id": st.platform_id,
                "loop_line_id": st.loop_line_id,
                "priority": self.priority_info.get(tn),
            })

        return RailwayState(
            simulation_time_min=sim_time_min,
            trains=trains_snapshot,
            track_occupancy=track_occupancy,
            platform_occupancy=platform_occupancy,
            loop_line_occupancy=loop_line_occupancy,
            conflicts=[c.as_dict() for c in conflicts],
            delays={tn: round(st.delay_min, 1) for tn, st in states.items()},
            waiting_trains=waiting,
            active_disruptions=disruption_injector.active_disruptions(),
            mode=mode,
        )


# ==========================================================================
# Orchestrator
# ==========================================================================
class SimulationLayer:
    """
    Top-level Simulation Layer object. Owns the Simulation Clock and all
    train states, and drives every module above, tick by tick, in exactly
    the order of the "Simulation Flow" diagram. Produces a `RailwayState`
    each tick -- and nothing else (no graphs, no GraphSAGE/PPO).
    """

    def __init__(
        self,
        timetable: Optional[TimetableDB] = None,
        stations: Optional[StationDB] = None,
        platforms: Optional[PlatformInfo] = None,
        priority: Optional[TrainPriorityInfo] = None,
        disruptions: Optional[DisruptionGenerator] = None,
        tick_min: float = 1.0,
        section_capacity: Optional[int] = None,
    ):
        if timetable is None:
            timetable, stations, platforms, priority, disruptions = load_all()

        self.timetable = timetable
        self.stations = stations
        self.platforms = platforms
        self.priority = priority

        # 1. Simulation Clock
        self.clock = SimulationClock(tick_min=tick_min)
        # 2. Train Position Updater
        self.position_updater = TrainPositionUpdater(timetable, platforms)
        # 3. Track Occupancy Manager
        self.track_occupancy_manager = TrackOccupancyManager(stations, section_capacity)
        # 4. Platform Occupancy Manager
        self.platform_occupancy_manager = PlatformOccupancyManager(platforms, priority)
        # 5. Loop Line Manager
        self.loop_line_manager = LoopLineManager(platforms, priority)
        # 6. Disruption Injector
        self.disruption_injector = DisruptionInjector(
            disruptions, self.platform_occupancy_manager, self.loop_line_manager,
            self.track_occupancy_manager,
        )
        # 7. Conflict Detector
        self.conflict_detector = ConflictDetector(stations)
        # 8. Delay Propagation Engine
        self.delay_propagation_engine = DelayPropagationEngine(tick_min, priority)
        # 9. Railway State Manager
        self.railway_state_manager = RailwayStateManager(priority)

        self.states: Dict[int, TrainState] = {
            tn: TrainState(train_number=tn) for tn in timetable.all_train_numbers()
        }
        self._hold_requests: set[int] = set()
        self._pass_requests: set[int] = set()
        self._pass_until: Dict[int, float] = {}
        self._last_railway_state: Optional[RailwayState] = None

    @property
    def mode(self) -> str:
        """Live mode, read straight from the disruption generator's own
        `demo_mode` flag -- reflects the actual simulation state rather
        than a separate UI-only label."""
        return "STRESS_TEST" if self.disruption_injector.generator.demo_mode else "REALISTIC"

    def set_demo_mode(self, on: bool) -> None:
        self.disruption_injector.generator.set_demo_mode(on)

    # Loop-and-overtake: the train waits in a loop line while the faster train
    # passes. One-off cost, but the conflict is cleared for LOOP_PASS_MIN minutes.
    LOOP_PASS_COST_MIN = 1.5
    LOOP_PASS_MIN = 10.0

    def request_loop_pass(self, train_numbers):
        self._pass_requests = {int(t) for t in (train_numbers or ())}

    def set_agent_requests(self, platform_requests=None, loop_requests=None, hold_requests=None) -> None:
        """Apply one-tick PPO control requests.

        Platform/loop requests only reorder genuine resource contention.
        HOLD requests add one minute of controllable operational delay to
        the selected train before the next movement update.
        """
        self.platform_occupancy_manager.set_agent_requests(platform_requests, loop_requests)
        self.loop_line_manager.set_agent_requests(loop_requests)
        self._hold_requests = {int(tn) for tn in (hold_requests or set())}

    def clear_agent_requests(self) -> None:
        self.set_agent_requests(set(), set(), set())

    def reset(self, start_time_min: float = 0.0, end_time_min: Optional[float] = None) -> RailwayState:
        self.clock.reset(start_time_min, end_time_min)
        self.track_occupancy_manager.reset()
        self.platform_occupancy_manager.reset()
        self.loop_line_manager.reset()
        self.disruption_injector.reset()
        self.states = {
            tn: TrainState(train_number=tn) for tn in self.timetable.all_train_numbers()
        }
        self._hold_requests = set()
        self._pass_requests = set(); self._pass_until = {}
        self._last_railway_state = None
        return self.get_railway_state()

    def step(self) -> RailwayState:
        """Advance the simulation by exactly one tick and return the new
        Railway State snapshot, following the diagram's fixed module order."""
        sim_time = self.clock.tick()

        # PPO-controlled delay only: a HOLD is an explicit dispatcher
        # decision, so it is the only direct train-delay increment here.
        for tn in self._pass_requests:
            if tn in self.states and self.states[tn].status != TrainStatus.TERMINATED:
                self.states[tn].delay_min += self.LOOP_PASS_COST_MIN
                self._pass_until[tn] = sim_time + self.LOOP_PASS_MIN
        self._pass_requests = set()
        held_now = set(self._hold_requests) | {tn for tn, u in self._pass_until.items() if u > sim_time}
        for tn in list(self._hold_requests):
            if tn in self.states and self.states[tn].status != TrainStatus.TERMINATED:
                self.states[tn].delay_min += 1.0
        self._hold_requests.clear()

        # 2. Train Position Updater
        active = self.timetable.trains_active_between(sim_time - 600, sim_time + 1)
        for tn in active:
            self.position_updater.update(self.states[tn], sim_time)

        # 3. Track Occupancy Manager
        track_occupancy = self.track_occupancy_manager.update(self.states, sim_time)

        # 4. Platform Occupancy Manager
        platform_occupancy, overflow = self.platform_occupancy_manager.allocate(self.states)

        # 5. Loop Line Manager
        loop_occupancy, waiting = self.loop_line_manager.assign(self.states, overflow)

        # Agent preferences are one-tick commands, not persistent overrides.
        self.set_agent_requests(set(), set())

        # 6. Disruption Injector
        self.disruption_injector.inject(sim_time, self.states)

        # 7. Conflict Detector
        conflicts = self.conflict_detector.detect(
            self.states, track_occupancy, self.track_occupancy_manager,
            platform_occupancy, loop_occupancy, waiting,
        )

        # 8. Delay Propagation Engine
        # The engine adds only resource-waiting/loop-hold delay. Automatic
        # lower-priority conflict penalties are disabled in the rebuilt
        # control loop; PPO must choose the train to hold.
        self.delay_propagation_engine.propagate(self.states, conflicts, sim_time, held_now)

        # 9. Railway State Manager -> Current Railway State
        self._last_railway_state = self.railway_state_manager.aggregate(
            sim_time, self.states, track_occupancy, platform_occupancy,
            loop_occupancy, conflicts, waiting, self.disruption_injector,
            mode=self.mode,
        )
        return self._last_railway_state

    def run(self, horizon_min: float, start_time_min: float = 0.0) -> List[RailwayState]:
        """Runs the simulator from start_time_min to horizon_min, returns the
        full list of Railway State snapshots (one per tick)."""
        self.reset(start_time_min, horizon_min)
        history = []
        while self.clock.now() <= horizon_min:
            history.append(self.step())
        return history

    def get_railway_state(self) -> RailwayState:
        """Returns the current Railway State without advancing the clock
        (used right after reset())."""
        return self.railway_state_manager.aggregate(
            self.clock.now(), self.states,
            self.track_occupancy_manager.occupancy_table(),
            self.platform_occupancy_manager.occupancy_table(),
            self.loop_line_manager.occupancy_table(),
            [], [], self.disruption_injector,
            mode=self.mode,
        )


# Backwards-compatible alias: earlier versions of this module named the
# orchestrator `RailwayEnvironmentSimulator`.
RailwayEnvironmentSimulator = SimulationLayer
