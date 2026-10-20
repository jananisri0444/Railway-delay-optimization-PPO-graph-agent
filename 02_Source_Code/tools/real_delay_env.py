"""Environment that replays REAL historical delays instead of synthetic lateness."""
import json
import project_paths as P
import numpy as np
from decision_layer.environment import RailwayPPOEnv

class RealDelayEnv(RailwayPPOEnv):
    """reset(seed=i) -> scenario number (i mod N) of the chosen split; weekday and per-train entry delays come from real data."""
    def __init__(self, *args, scenario_file=str(P.REAL_PROC / 'real_day_scenarios.json'), split='test', **kw):
        kw['late_fraction'] = 0.0                      # no synthetic lateness
        super().__init__(*args, **kw)
        data = json.load(open(scenario_file))
        self.scenarios = [s for s in data['scenarios'] if split == 'all' or s['split'] == split]
        if not self.scenarios: raise SystemExit(f'No scenarios in split "{split}". Run build_real_delays.py first.')
        self.current = None

    def reset(self, seed=None, options=None):
        i = int(seed if seed is not None else 0) % len(self.scenarios)
        self.current = self.scenarios[i]; self.fixed_weekday = self.current['weekday']
        super().reset(seed=seed, options=options)
        self._initial_delay = {}
        for tn, st in self._sim.states.items():
            d = float(self.current['delays'].get(str(int(tn)), 0.0)); st.delay_min = d
            if d > 0: self._initial_delay[int(tn)] = d
        self._rs = self._sim.get_railway_state().as_dict(); self._prev_delay = self.total_delay(); self._snapshot()
        return self._obs.copy(), {}


class _FilteredTT:
    """Wraps the timetable so only trains that REALLY ran on the replayed day are simulated."""
    def __init__(self, base, owner): self._base, self._owner = base, owner
    def resolve(self, weekday, start, end):
        tt = self._base.resolve(weekday, start, end)
        if self._owner.allowed is not None: tt.trains = {k: v for k, v in tt.trains.items() if int(k) in self._owner.allowed}
        return tt
    def __getattr__(self, n): return getattr(self._base, n)


class RealDayEnv(RailwayPPOEnv):
    """Replays ONE real calendar day: real weekday, only the trains that really ran, their real entry delays.
    No synthetic lateness or disruptions unless stress_mode=True."""
    def __init__(self, *args, **kw):
        kw['late_fraction'] = 0.0; super().__init__(*args, **kw)
        self.allowed, self.day_weekday, self.day_delays, self.fill = None, 0, {}, None
        self._base_tt = _FilteredTT(self._base_tt, self)

    def set_day(self, weekday, trains, delays, fill=None):
        """trains=None -> simulate EVERY train the timetable runs that weekday (not only those observed). Trains without a real
        delay then draw one at random from `fill` (the real delays seen that day), so the day is as busy as the full timetable."""
        self.day_weekday, self.allowed = int(weekday), (None if trains is None else {int(t) for t in trains})
        self.day_delays, self.fill = {int(k): float(v) for k, v in delays.items()}, ([float(x) for x in fill] if fill is not None else None)

    def reset(self, seed=None, options=None):
        self.fixed_weekday = self.day_weekday
        super().reset(seed=seed, options=options); self._initial_delay = {}
        rng = np.random.default_rng(0 if seed is None else int(seed))
        for tn, st in self._sim.states.items():
            d = self.day_delays.get(int(tn))
            if d is None: d = float(rng.choice(self.fill)) if self.fill else 0.0
            st.delay_min = d
            if d > 0: self._initial_delay[int(tn)] = d
        self._rs = self._sim.get_railway_state().as_dict(); self._prev_delay = self.total_delay(); self._snapshot()
        return self._obs.copy(), {}
