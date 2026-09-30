#!/usr/bin/env python3
"""Plan one scenario and show it.

python parking_demo.py                  # B2: normal perpendicular reverse parking
python parking_demo.py --id C2 --save c2.png
python parking_demo.py --id B3 --vehicle suv
python parking_demo.py --id C2 --gif docs/parking.gif --search-gif docs/search.gif
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collision_checker import CollisionChecker  # noqa: E402
from metrics import IndependentValidator, trajectory_metrics  # noqa: E402
from parking_planner import PLANNERS  # noqa: E402
from state_lattice import PlannerConfig  # noqa: E402
from vehicle import VEHICLE_PRESETS  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="B2", help="scenario id (see scenarios.py)")
    ap.add_argument("--vehicle", default=None, choices=sorted(VEHICLE_PRESETS))
    ap.add_argument("--planner", default="bidirectional", choices=sorted(PLANNERS))
    ap.add_argument("--save", default=None, help="save the figure instead of showing it")
    ap.add_argument("--gif", default=None, help="save an animation of the vehicle driving the path")
    ap.add_argument("--search-gif", default=None, help="save an animation of the search (bidirectional planner)")
    a = ap.parse_args(argv)
    if a.save or a.gif or a.search_gif:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scenarios import build_scenario
    from visualization import animate_result, animate_search, plot_result

    sc = build_scenario(a.id)
    vehicle = VEHICLE_PRESETS[a.vehicle]() if a.vehicle else sc.make_vehicle()
    planner = PLANNERS[a.planner](vehicle, sc.make_config(PlannerConfig()))
    detailed = planner.plan_detailed(sc.grid, sc.start, sc.goal) if hasattr(planner, "plan_detailed") else None
    res = detailed.result if detailed is not None else planner.plan(sc.grid, sc.start, sc.goal)
    print(f"{sc.id}: {sc.name}\n{vehicle.summary()}\n{res.status.value} ({res.search}) "
          f"{res.planning_time * 1e3:.0f} ms, {res.expanded_nodes} expansions")
    m = {"success": res.success}
    if res.success:
        m.update(trajectory_metrics(res.trajectory, sc.goal, vehicle,
                                    CollisionChecker(sc.grid, vehicle, "rectangle", 0.0)))
        moves = [f"{'F' if p.points[-1].direction > 0 else 'R'} {p.points[-1].arc_length - p.points[0].arc_length:.2f} m"
                 for p in res.trajectory.split_by_direction()]
        print(f"  gear changes {res.trajectory.n_direction_changes}: {', '.join(moves)}")
        print(f"  length {m['trajectory_length']:.2f} m, min clearance {m['min_clearance']:.2f} m, "
              f"goal error {m['goal_position_error']:.3f} m / {m['goal_yaw_error_deg']:.2f} deg")
        ok = IndependentValidator(sc.grid, vehicle).is_collision_free(res.trajectory.poses())
        print(f"  independent validation: {'collision-free' if ok else 'COLLISION'}")
    if a.gif or a.search_gif:
        if not res.success:
            return 1
        title = f"{sc.id} {sc.name}"
        for out in (a.gif, a.search_gif):
            if out:
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        if a.gif:
            animate_result(sc.grid, vehicle, sc.start, sc.goal, res, sc.obstacles, a.gif, title=title)
        if a.search_gif:
            if detailed is None:
                ap.error("--search-gif needs the bidirectional planner")
            animate_search(sc.grid, vehicle, sc.start, sc.goal, detailed, sc.obstacles, a.search_gif, title=title)
        return 0
    plot_result(sc.grid, vehicle, sc.start, sc.goal, res, sc.obstacles,
                title=f"{sc.id} {sc.name} - {vehicle.summary()}", metrics=m)
    if a.save:
        plt.savefig(a.save, dpi=110)
    else:
        plt.show()
    return 0 if res.success else 1


if __name__ == "__main__":
    sys.exit(main())
