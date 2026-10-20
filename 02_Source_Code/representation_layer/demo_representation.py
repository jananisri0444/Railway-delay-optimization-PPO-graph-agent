"""
Demo: Representation Layer
==========================
Runs 5 ticks of the simulation, builds the DORG each tick, and prints
the shape of the resulting state vector.

Run from project root:
    python -m representation_layer.demo_representation
or:
    cd railway_scheduler && python representation_layer/demo_representation.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_layer.data_layer import load_all
from simulation_layer.simulation_layer import SimulationLayer
from graph_construction_layer.dorg_builder import DORGBuilder
from representation_layer import RepresentationLayer


def main():
    print("Loading data …")
    timetable, stations, platforms, priority, disruptions = load_all()

    sim  = SimulationLayer(timetable, stations, platforms, priority, disruptions, tick_min=1.0)
    dorg = DORGBuilder(stations, platforms)
    rep  = RepresentationLayer(hidden_dim=64, num_layers=2, dropout=0.0)

    print(f"State vector dimension: {rep.state_dim}")
    print(f"Encoder architecture:\n  {rep.encoder}\n")

    sim.reset(start_time_min=480, end_time_min=480 + 10)   # 8:00 AM, 10 ticks
    for tick in range(10):
        rs = sim.step()
        g  = dorg.build(rs)
        rs_dict = rs.as_dict()
        state   = rep.encode(g, rs_dict)

        n_trains  = sum(1 for t in rs_dict["trains"]
                        if t["status"] not in ("NOT_STARTED", "TERMINATED"))
        n_waiting  = len(rs_dict["waiting_trains"])
        n_conflict = len(rs_dict["conflicts"])

        print(
            f"  tick {tick+1:3d} | t={rs_dict['simulation_time_min']:6.1f} min "
            f"| active={n_trains:3d} waiting={n_waiting} conflicts={n_conflict} "
            f"| state[0:5]={state[:5].round(3)}"
        )

    print("\nRepresentation layer demo complete ✓")


if __name__ == "__main__":
    main()
