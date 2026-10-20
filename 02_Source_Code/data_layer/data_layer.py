"""
Data Layer
==========
Loads and serves the four static/generative data sources shown in the
architecture diagram:

    Timetable DB        -> TimetableDB      (dataset.csv)
    Station DB           -> StationDB        (chennai_bangalore_stations.csv
                                               + stations_with_coordinates.csv)
    Platform Info        -> PlatformInfo     (loop_line_data.csv)
    Train Priority Info  -> TrainPriorityInfo (derived from dataset.csv)
    Disruption Generator -> DisruptionGenerator (synthetic, configurable)

Nothing here runs a simulation -- this layer only loads/derives data and
exposes clean lookup APIs that the Simulation Layer consumes.
"""

from __future__ import annotations
import random
from dataclasses import replace, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "01_Data" / "1_Simulator_Timetable"


# --------------------------------------------------------------------------
# Timetable DB
# --------------------------------------------------------------------------
@dataclass
class Stop:
    sequence: int
    station_code: str
    station_name: str
    is_halt: bool
    day: int
    scheduled_arrival_min: float
    scheduled_departure_min: float
    halt_duration_min: float
    distance_from_source_km: float


@dataclass
class TrainSchedule:
    train_number: int
    train_name: str
    category: str
    train_type: str
    source_code: str
    destination_code: str
    total_distance_km: float
    avg_speed_kmph: float
    stops: List[Stop] = field(default_factory=list)
    running_days: frozenset = frozenset(range(7))   # 0=Mon .. 6=Sun, days the train DEPARTS its origin

    @property
    def origin_min(self) -> float:
        return self.stops[0].scheduled_departure_min

    @property
    def terminus_min(self) -> float:
        return self.stops[-1].scheduled_arrival_min


class TimetableDB:
    """Loads dataset.csv and exposes per-train schedules."""

    def __init__(self, csv_path: Path = DATA_DIR / "dataset.csv"):
        df = pd.read_csv(csv_path)
        df = df.sort_values(["trainNumber", "sequence"])
        self.trains: Dict[int, TrainSchedule] = {}

        for train_number, g in df.groupby("trainNumber"):
            first = g.iloc[0]
            stops = [
                Stop(
                    sequence=int(r.sequence),
                    station_code=str(r.stationCode),
                    station_name=str(r.stationName),
                    is_halt=bool(r.isHalt),
                    day=int(r.day),
                    scheduled_arrival_min=float(r.scheduledArrivalMin),
                    scheduled_departure_min=float(r.scheduledDepartureMin),
                    halt_duration_min=float(r.haltDurationMinutes or 0),
                    distance_from_source_km=float(r.distanceFromSourceKm),
                )
                for r in g.itertuples(index=False)
            ]
            self.trains[int(train_number)] = TrainSchedule(
                train_number=int(train_number),
                train_name=str(first.trainName),
                category=str(first.category),
                train_type=str(first.type),
                source_code=str(first.sourceStationCode),
                destination_code=str(first.destinationStationCode),
                total_distance_km=float(first.distanceKm),
                avg_speed_kmph=float(first.avgSpeedKmph),
                stops=stops,
                running_days=frozenset(DAY_NAMES.index(x.strip()) for x in str(first.runningDaysList).split(",")),
            )
        self.weekday = None          # raw (unresolved) database
        self.duplicates = 0

    def resolve(self, weekday: int, start_min: float = 0.0, end_min: float = 1440.0) -> "TimetableDB":
        """Timetable for ONE real calendar day (0=Mon..6=Sun).

        * keeps only trains that run on that date: running days are origin-departure
          days, so a train whose corridor stops are on journey-day d (CSV `day`)
          started its journey d-1 days earlier and is checked against THAT weekday;
        * converts every stop time to minutes since 00:00 of the simulated date
          (negative = yesterday, >1440 = after midnight), so overnight windows work;
        * keeps only trains overlapping [start_min, end_min].
        """
        out = TimetableDB.__new__(TimetableDB)
        out.trains, out.weekday, out.duplicates = {}, int(weekday) % 7, 0
        for tn, sch in self.trains.items():
            best, n_ok = None, 0
            for k in range(-4, 3):                      # origin date = simulated date + k days
                if (out.weekday + k) % 7 not in sch.running_days:
                    continue
                stops = [replace(p,
                                 scheduled_arrival_min=p.scheduled_arrival_min + (k + p.day - 1) * 1440.0,
                                 scheduled_departure_min=p.scheduled_departure_min + (k + p.day - 1) * 1440.0)
                         for p in sch.stops]
                o, t = stops[0].scheduled_departure_min, stops[-1].scheduled_arrival_min
                if o <= end_min and t >= start_min:
                    n_ok += 1
                    ov = min(t, end_min) - max(o, start_min)
                    if best is None or ov > best[0]:
                        best = (ov, stops)
            if best:
                out.trains[tn] = replace(sch, stops=best[1])
                out.duplicates += n_ok - 1
        return out

    def get(self, train_number: int) -> TrainSchedule:
        return self.trains[train_number]

    def all_train_numbers(self) -> List[int]:
        return list(self.trains.keys())

    def trains_active_between(self, start_min: float, end_min: float) -> List[int]:
        """Trains whose journey overlaps a given simulation window."""
        out = []
        for tn, sched in self.trains.items():
            if sched.origin_min <= end_min and sched.terminus_min >= start_min:
                out.append(tn)
        return out


# --------------------------------------------------------------------------
# Station DB
# --------------------------------------------------------------------------
@dataclass
class Station:
    code: str
    name: str
    distance_km: float
    platforms: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class StationDB:
    """Loads station master list, enriched with coordinates where available."""

    def __init__(
        self,
        stations_csv: Path = DATA_DIR / "chennai_bangalore_stations.csv",
        coords_csv: Path = DATA_DIR / "stations_with_coordinates.csv",
    ):
        base = pd.read_csv(stations_csv)
        coords = pd.read_csv(coords_csv)[["stationCode", "latitude", "longitude"]]
        merged = base.merge(coords, on="stationCode", how="left")

        self.stations: Dict[str, Station] = {}
        for r in merged.itertuples(index=False):
            self.stations[r.stationCode] = Station(
                code=r.stationCode,
                name=r.stationName,
                distance_km=float(r.distanceKm),
                platforms=int(r.platforms),
                latitude=None if pd.isna(r.latitude) else float(r.latitude),
                longitude=None if pd.isna(r.longitude) else float(r.longitude),
            )
        # ordered station list by corridor distance (used for topology)
        self.ordered_codes: List[str] = [
            s.code for s in sorted(self.stations.values(), key=lambda s: s.distance_km)
        ]

    def get(self, code: str) -> Station:
        return self.stations[code]

    def distance_km(self, code: str) -> float:
        return self.stations[code].distance_km

    def next_station(self, code: str) -> Optional[str]:
        idx = self.ordered_codes.index(code)
        return self.ordered_codes[idx + 1] if idx + 1 < len(self.ordered_codes) else None

    def prev_station(self, code: str) -> Optional[str]:
        idx = self.ordered_codes.index(code)
        return self.ordered_codes[idx - 1] if idx > 0 else None


# --------------------------------------------------------------------------
# Platform Info (loop lines / capacity)
# --------------------------------------------------------------------------
@dataclass
class PlatformCapacity:
    station_code: str
    platforms: int
    tracks: int
    has_loop_line: bool
    loop_line_count: int

    @property
    def total_capacity(self) -> int:
        """Max trains a station can simultaneously hold (platforms + loop lines)."""
        return self.platforms + self.loop_line_count


class PlatformInfo:
    """Loads loop_line_data.csv -> per-station simultaneous-occupancy capacity."""

    def __init__(self, csv_path: Path = DATA_DIR / "loop_line_data.csv"):
        df = pd.read_csv(csv_path)
        self.capacity: Dict[str, PlatformCapacity] = {}
        for r in df.itertuples(index=False):
            self.capacity[r.stationCode] = PlatformCapacity(
                station_code=r.stationCode,
                platforms=int(r.platforms),
                tracks=int(r.tracks),
                has_loop_line=bool(r.has_loop_line),
                loop_line_count=int(r.loop_line_count),
            )

    def get(self, code: str) -> PlatformCapacity:
        return self.capacity[code]

    def capacity_of(self, code: str) -> int:
        c = self.capacity.get(code)
        return c.total_capacity if c else 1


# --------------------------------------------------------------------------
# Train Priority Info
# --------------------------------------------------------------------------
# Lower number = higher precedence, mirrors Indian Railways operational
# precedence: premium/high-speed > superfast > mail/express > MEMU/EMU/passenger.
PRIORITY_RANK = {
    "Vande Bharat Express": 1,
    "Shatabdi Express": 1,
    "Amrit Bharat Express": 2,
    "Superfast Express": 3,
    "Mail/Express": 4,
    "MEMU": 5,
    "EMU": 5,
    "Suburban": 5,
    "Passenger": 6,
    "Train on Demand": 6,
}


class TrainPriorityInfo:
    """Derives a numeric dispatch priority per train from its `type`."""

    def __init__(self, timetable: TimetableDB):
        self.priority: Dict[int, int] = {}
        for tn, sched in timetable.trains.items():
            self.priority[tn] = PRIORITY_RANK.get(sched.train_type, 6)

    def get(self, train_number: int) -> int:
        return self.priority.get(train_number, 6)

    def higher_priority(self, train_a: int, train_b: int) -> int:
        """Returns the train_number with dispatch precedence (lower rank wins)."""
        pa, pb = self.get(train_a), self.get(train_b)
        return train_a if pa <= pb else train_b


# --------------------------------------------------------------------------
# Disruption Generator
# --------------------------------------------------------------------------
@dataclass
class DisruptionEvent:
    tick_min: float
    kind: str            # "delay" | "platform_block" | "loop_block" |
                          # "track_block" | "speed_restriction"
    train_number: Optional[int]
    station_code: Optional[str]
    section: Optional[Tuple[str, str]] = None   # (station_a, station_b) for track-level events
    magnitude: float = 0.0     # minutes of delay, blocked-slot count, or speed factor (0-1)
    duration_min: float = 0.0


class DisruptionGenerator:
    """
    Synthetic disruption source. No ground-truth disruption dataset was
    provided, so this generates configurable random events representative
    of real corridor disruptions:

        - "delay"             -> signal failure / late-running style delay
                                  injected directly onto a train
        - "platform_block"    -> a platform at a station is taken out of
                                  service (maintenance / stabling)
        - "loop_block"        -> a loop line at a station is taken out of
                                  service
        - "track_block"       -> a track section between two adjacent
                                  stations is closed (maintenance block /
                                  track closure)
        - "speed_restriction" -> a temporary speed restriction (TSR) is
                                  imposed on a track section

    Swap `.generate()` out for a real feed if one becomes available; the
    rest of the Simulation Layer only depends on the `DisruptionEvent`
    contract above, not on how events are produced.

    Demo / stress-test mode
    ------------------------
    `demo_mode` does NOT introduce a new event kind, touch
    `loop_line_data.csv`, or change any headway/capacity rule -- it is the
    exact same "platform_block" / "loop_block" events used in normal
    operation, just:

      1. fired far more often (`demo_*_prob_per_tick` instead of the
         normal-mode probabilities), and
      2. sized (in `_block_magnitude`) to actually bind -- a maintenance
         block that only closes 1 of a station's 8 platforms rarely causes
         real contention, so in demo mode the magnitude is calibrated (from
         `PlatformInfo`, read-only) to leave just one platform/loop free at
         the targeted station for the duration of the event, and
      3. targeted (`_pick_station`) preferentially at a station where a
         train is actually present or about to arrive (`hot_stations`,
         computed by the injector from live train state each tick), so the
         temporary closure has a real train to contend with instead of
         landing on an empty station.

    Every event this produces still goes through the same
    `DisruptionInjector.inject()` -> `PlatformOccupancyManager` /
    `LoopLineManager` path as normal mode, still expires on its own, and
    still only ever *reduces available capacity for a while* -- the
    allocate() / assign() overflow logic is what turns that into a
    genuine WAITING train, exactly as in normal operation.
    """

    def __init__(
        self,
        station_codes: List[str],
        train_numbers: List[int],
        seed: int = 42,
        delay_prob_per_tick: float = 0.02,
        platform_block_prob_per_tick: float = 0.003,
        loop_block_prob_per_tick: float = 0.002,
        track_block_prob_per_tick: float = 0.002,
        speed_restriction_prob_per_tick: float = 0.003,
        sections: Optional[List[Tuple[str, str]]] = None,
        platform_info: Optional["PlatformInfo"] = None,
        demo_mode: bool = False,
        demo_platform_block_prob_per_tick: float = 0.10,
        demo_loop_block_prob_per_tick: float = 0.08,
        demo_hot_station_bias: float = 0.85,
    ):
        self.rng = random.Random(seed)
        self.station_codes = station_codes
        self.train_numbers = train_numbers
        self.sections = sections or list(zip(station_codes[:-1], station_codes[1:]))
        self.platform_info = platform_info  # read-only lookup, used only to size demo blocks

        self.delay_prob = delay_prob_per_tick
        self.platform_block_prob = platform_block_prob_per_tick
        self.loop_block_prob = loop_block_prob_per_tick
        self.track_block_prob = track_block_prob_per_tick
        self.speed_restriction_prob = speed_restriction_prob_per_tick

        self.demo_mode = demo_mode
        self.demo_platform_block_prob = demo_platform_block_prob_per_tick
        self.demo_loop_block_prob = demo_loop_block_prob_per_tick
        self.demo_hot_station_bias = demo_hot_station_bias

    def set_demo_mode(self, on: bool) -> None:
        """Flip stress-test mode on/off without rebuilding the generator."""
        self.demo_mode = bool(on)

    def _pick_station(self, hot_stations: Optional[set]) -> str:
        """Normal mode: uniform pick, unchanged. Demo mode: prefer a
        station that currently has a train present or arriving, so the
        temporary closure has real traffic to contend with."""
        if self.demo_mode and hot_stations and self.rng.random() < self.demo_hot_station_bias:
            return self.rng.choice(sorted(hot_stations))
        return self.rng.choice(self.station_codes)

    def _block_magnitude(self, station_code: str, kind: str) -> int:
        """Normal mode: always 1 (unchanged behaviour). Demo mode: close
        this station down to exactly one remaining platform/loop line for
        the duration of the event -- still temporary, still per-station,
        still leaves the underlying infrastructure data untouched."""
        if not self.demo_mode or self.platform_info is None:
            return 1
        cap = self.platform_info.capacity.get(station_code)
        if cap is None:
            return 1
        base = cap.platforms if kind == "platform" else cap.loop_line_count
        return max(1, base - 1)

    def generate(self, tick_min: float, hot_stations: Optional[set] = None) -> List[DisruptionEvent]:
        events: List[DisruptionEvent] = []

        if self.train_numbers and self.rng.random() < self.delay_prob:
            events.append(
                DisruptionEvent(
                    tick_min=tick_min,
                    kind="delay",
                    train_number=self.rng.choice(self.train_numbers),
                    station_code=None,
                    magnitude=self.rng.randint(5, 45),
                )
            )

        platform_prob = self.demo_platform_block_prob if self.demo_mode else self.platform_block_prob
        if self.station_codes and self.rng.random() < platform_prob:
            station = self._pick_station(hot_stations)
            events.append(
                DisruptionEvent(
                    tick_min=tick_min,
                    kind="platform_block",
                    train_number=None,
                    station_code=station,
                    magnitude=self._block_magnitude(station, "platform"),
                    duration_min=self.rng.randint(10, 60),
                )
            )

        loop_prob = self.demo_loop_block_prob if self.demo_mode else self.loop_block_prob
        if self.station_codes and self.rng.random() < loop_prob:
            station = self._pick_station(hot_stations)
            events.append(
                DisruptionEvent(
                    tick_min=tick_min,
                    kind="loop_block",
                    train_number=None,
                    station_code=station,
                    magnitude=self._block_magnitude(station, "loop"),
                    duration_min=self.rng.randint(10, 90),
                )
            )

        if self.sections and self.rng.random() < self.track_block_prob:
            events.append(
                DisruptionEvent(
                    tick_min=tick_min,
                    kind="track_block",
                    train_number=None,
                    station_code=None,
                    section=self.rng.choice(self.sections),
                    magnitude=1,
                    duration_min=self.rng.randint(15, 120),
                )
            )

        if self.sections and self.rng.random() < self.speed_restriction_prob:
            events.append(
                DisruptionEvent(
                    tick_min=tick_min,
                    kind="speed_restriction",
                    train_number=None,
                    station_code=None,
                    section=self.rng.choice(self.sections),
                    magnitude=round(self.rng.uniform(0.3, 0.7), 2),  # speed factor
                    duration_min=self.rng.randint(20, 180),
                )
            )

        return events


# --------------------------------------------------------------------------
# Convenience loader
# --------------------------------------------------------------------------
def load_all(demo_mode: bool = False, seed: int = 42):
    timetable = TimetableDB()
    stations = StationDB()
    platforms = PlatformInfo()
    priority = TrainPriorityInfo(timetable)
    sections = list(zip(stations.ordered_codes[:-1], stations.ordered_codes[1:]))
    disruptions = DisruptionGenerator(
        station_codes=stations.ordered_codes,
        train_numbers=timetable.all_train_numbers(),
        sections=sections,
        platform_info=platforms,
        demo_mode=demo_mode,
        seed=seed,
    )
    return timetable, stations, platforms, priority, disruptions
