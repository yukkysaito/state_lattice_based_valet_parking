"""Production planner: bidirectional state-lattice search + post-processing.

Why two searches
----------------
Parking is hard at *one end* of the path. Where the goal is confined (tight
parallel slot, narrow perpendicular slot), a forward A* has to discover the
precise final manoeuvre at the far end of its search tree. Searching from the
goal to the start ("how would I leave this slot?") and time-reversing the
result finds it near the root, where the search is exhaustive and cheap. When
the *start* is confined (leaving a nose-in slot, dead ends), the forward search
is the easy direction. The kinematic model is time-reversible, so the reversed
exit path is feasible and has the same geometry.

``BidirectionalPlanner`` runs

* ``forward`` : nominal lattice (``config``), start -> goal
* ``exit``    : fine lattice (xy 0.05 m + ``micro`` primitives), goal -> start,
  forward/reverse weights swapped so that the reversed path is priced like a
  parking path, and the real start must be hit exactly (analytic shot)

alternately in fixed expansion chunks (deterministic) under one time budget.
When one search succeeds, the other continues for a bounded grace budget as
branch & bound (only solutions at least 3 % cheaper than the first one are
kept), and the cheaper result is returned (both use the same cost model). Proofs (start/goal in collision, 2D unreachability,
invalid input) from either search terminate immediately. If one search fails
without a proof (e.g. the exit search cannot hit the exact start pose), the
other one continues.

The fine lattice is used for the exit search only: the confined end is near
its root, and fine closed-set cells are what makes tight slots solvable
(0.1 m cells merge sub-cell positions that matter when the slack is ~0.1 m).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from meet import Meeting
from path_shortcut import path_cost, shortcut
from state_lattice import PROOF_STATUSES, PlannerConfig, PlanResult, PlanStatus, StateLatticePlanner
from trajectory import Trajectory
from vehicle import VehicleInfo

# status of the exit search (goal -> start) expressed for the original problem
_SWAP = {PlanStatus.START_IN_COLLISION: PlanStatus.GOAL_IN_COLLISION,
         PlanStatus.GOAL_IN_COLLISION: PlanStatus.START_IN_COLLISION}


def reverse_trajectory(traj: Trajectory, wheel_base: float) -> Trajectory:
    """Time-reverse a trajectory: driving the same path backwards with flipped
    direction and the same steering is kinematically feasible."""
    pts = traj.points
    n = len(pts)
    if n < 2:
        return traj
    poses = np.array([[p.x, p.y, p.yaw] for p in reversed(pts)])
    dirs = np.empty(n, dtype=int)
    steer = np.empty(n)
    for i in range(1, n):
        src = pts[n - i]           # motion p[n-i-1] -> p[n-i], traversed backwards
        dirs[i] = -src.direction
        steer[i] = src.steering
    dirs[0], steer[0] = dirs[1], steer[1]
    return Trajectory.from_arrays(poses, dirs, steer, wheel_base)


def exit_config(cfg: PlannerConfig, fine: bool = True) -> PlannerConfig:
    """Configuration of the goal -> start search.

    * forward/reverse weights swapped (a reverse move of the exit path is a
      forward move of the parking path),
    * the new goal (= real start) must be reached exactly (analytic shot),
    * ``fine``: 0.05 m closed-set cells and micro primitives for the confined end.
    """
    kw = dict(forward_weight=cfg.reverse_weight, reverse_weight=cfg.forward_weight,
              position_tolerance=1e-3, yaw_tolerance=1e-3)
    if fine:
        kw.update(xy_resolution=min(cfg.xy_resolution, 0.05),
                  yaw_resolution=min(cfg.yaw_resolution, math.radians(2.5)),
                  length_classes=tuple(dict.fromkeys(("micro",) + tuple(cfg.length_classes))))
    return cfg.with_changes(**kw)


def _finalize_exit(res: PlanResult, vehicle: VehicleInfo, start) -> PlanResult:
    """Map an exit-search result back to the original problem (never raises)."""
    try:
        return _finalize_exit_unsafe(res, vehicle, start)
    except Exception as exc:  # pragma: no cover - defensive
        return PlanResult(PlanStatus.ERROR, f"exit result conversion failed: {exc!r}", search="exit")


def _finalize_exit_unsafe(res: PlanResult, vehicle: VehicleInfo, start) -> PlanResult:
    res.search = "exit"
    if res.status in _SWAP:
        res.status = _SWAP[res.status]
    if res.success:
        res.trajectory = reverse_trajectory(res.trajectory, vehicle.wheel_base)
        p0 = res.trajectory.points[0]
        if math.hypot(p0.x - start[0], p0.y - start[1]) > 1e-2:
            res.status = PlanStatus.ERROR
            res.message = "exit search did not end at the start pose"
        else:
            res.message = "exit search (goal->start, reversed): " + res.message
    return res


def goal_region_seeds(goal, cfg: PlannerConfig, deviation_weight: float = 2.0):
    """Root poses for the exit search covering the goal tolerance region.

    A parked pose anywhere inside the tolerance is acceptable, so the exit search
    may start from any of them; aligning to the exact goal can otherwise cost
    several extra direction switches in a tight slot. Each seed has an initial
    cost ``deviation_weight * (|dp|/tol + |dyaw|/yaw_tol)`` (max ~2 x weight),
    well below one switch penalty, so the exact goal wins unless an offset pose
    saves a manoeuvre. Seeds are on a small grid: 7 longitudinal x 3 lateral x
    3 yaw values, restricted to the tolerance disc.
    """
    tol, ytol = cfg.position_tolerance, cfg.yaw_tolerance
    c, s = math.cos(goal[2]), math.sin(goal[2])
    seeds = []
    for dl in np.linspace(-tol, tol, 7):
        for dt in (-tol / 3, 0.0, tol / 3):
            if math.hypot(dl, dt) > tol + 1e-9:
                continue
            for dyaw in (-ytol, 0.0, ytol):
                if dl == 0.0 and dt == 0.0 and dyaw == 0.0:
                    continue
                pose = (goal[0] + c * dl - s * dt, goal[1] + s * dl + c * dt, goal[2] + dyaw)
                g0 = deviation_weight * (math.hypot(dl, dt) / max(tol, 1e-9) + abs(dyaw) / max(ytol, 1e-9))
                seeds.append((pose, g0))
    return seeds


def plan_by_exit(vehicle: VehicleInfo, cfg: PlannerConfig, grid, start, goal, fine: bool = True) -> PlanResult:
    """Exit search only (goal -> start), result reversed."""
    res = StateLatticePlanner(vehicle, exit_config(cfg, fine)).plan(grid, goal, start)
    return _finalize_exit(res, vehicle, start)


@dataclass
class BidirectionalResult:
    result: PlanResult
    forward: Optional[PlanResult] = None
    exit: Optional[PlanResult] = None


class BidirectionalPlanner:
    """Forward + exit search sharing one deterministic budget (see module doc)."""

    def __init__(self, vehicle: VehicleInfo, config: PlannerConfig = PlannerConfig(),
                 exit_fine: bool = True, chunk: int = 64,
                 grace_ratio: float = 0.5, min_grace_expansions: int = 300, improvement_ratio: float = 0.97,
                 shortcut: bool = True, sealed_goal_expansions: int = 20000, reduction_expansions: int = 40000,
                 meet: bool = True):
        self.vehicle = vehicle
        self.config = config
        self.chunk = chunk
        self.grace_ratio = grace_ratio
        self.improvement_ratio = improvement_ratio
        self.shortcut = shortcut
        self.sealed_goal_expansions = sealed_goal_expansions
        self.reduction_expansions = reduction_expansions
        self.min_grace_expansions = min_grace_expansions
        self.meet = meet
        self.forward = StateLatticePlanner(vehicle, config)
        self.exit = StateLatticePlanner(vehicle, exit_config(config, exit_fine))
        self.primitives = self.forward.primitives

    def plan(self, grid, start, goal) -> PlanResult:
        return self.plan_detailed(grid, start, goal).result

    def plan_detailed(self, grid, start, goal) -> BidirectionalResult:
        """Plan, then - only if the path needs 2 or more gear changes - try again
        with one gear change less (small budget) until that fails: the product
        prefers fewer manoeuvres over a shorter path. Finally RS shortcutting."""
        t0 = time.perf_counter()
        deadline = t0 + self.config.max_planning_time
        self._start, self._goal = start, goal
        budget = self.config.max_expanded_nodes      # total over all searches below
        best = self._plan_once(grid, start, goal, t0, deadline, None, budget)
        used = best.result.expanded_nodes
        while (best.result.success and best.result.trajectory.n_direction_changes >= 2
               and time.perf_counter() < deadline and used < budget):
            k = best.result.trajectory.n_direction_changes
            trial = self._plan_once(grid, start, goal, t0, deadline, k - 1,
                                    min(self.reduction_expansions, budget - used))
            used += trial.result.expanded_nodes
            if not (trial.result.success and trial.result.trajectory.n_direction_changes < k):
                break
            best = trial
        chosen = best.result
        chosen.expanded_nodes = used
        if chosen.success and self.shortcut:
            self._post_process(chosen, grid)
        if chosen.success:
            self._check_endpoints(chosen, start, goal)
        chosen.planning_time = time.perf_counter() - t0
        return best

    def _plan_once(self, grid, start, goal, t0, deadline, cap, max_exp) -> BidirectionalResult:
        """One bidirectional search; ``max_exp`` bounds the expansions of both together."""
        fwd = self.forward.start_search(grid, start, goal, "forward", max_gear_switches=cap, max_expansions=max_exp)
        # the exit search may start anywhere in the goal tolerance region
        ext = self.exit.start_search(grid, goal, start, "exit", goal_region_seeds(goal, self.config),
                                     max_gear_switches=cap, max_expansions=max_exp, slot_pose=goal)
        if self.meet:
            Meeting(fwd, ext)
        res = {"forward": fwd.result,
               "exit": _finalize_exit(ext.result, self.vehicle, start) if ext.result is not None else None}
        searches = {"forward": fwd, "exit": ext}
        self._max_exp = max_exp

        def step(name: str, n: int) -> None:
            n = min(n, max_exp - fwd.expanded - ext.expanded)
            r = searches[name].step(n, deadline) if n > 0 else searches[name].result
            if r is not None and res[name] is None:
                res[name] = _finalize_exit(r, self.vehicle, start) if name == "exit" else r

        while True:
            for r in res.values():
                if r is not None and r.status in PROOF_STATUSES:
                    return self._done(r, res, t0, grid)
            # sealed goal region: the exit search (started from the whole goal
            # tolerance region) exhausted a SMALL reachable set, i.e. the goal
            # cannot be left - hence not entered - within the manoeuvre limits
            # (kinematics are reversible). Not a strict proof (discrete seeds and
            # lattice), but a reliable, fast "slot too tight" answer.
            e = res["exit"]
            if (e is not None and e.status == PlanStatus.NO_PATH and res["forward"] is None
                    and searches["exit"].expanded < self.sealed_goal_expansions):
                r = searches["forward"].stop(PlanStatus.NO_PATH, "goal region sealed: it cannot be left with "
                                             f"<= {searches['exit'].cap} gear changes and manoeuvres "
                                             f">= {self.config.min_maneuver_length} m (exit search exhausted "
                                             f"after {searches['exit'].expanded} expansions)")
                res["forward"] = r
                return self._done(r, res, t0, grid)
            winner = next((k for k, r in res.items() if r is not None and r.success), None)
            if winner is not None:
                return self._refine(winner, res, searches, step, t0, grid)
            if all(r is not None for r in res.values()):
                # both failed without proof (an ERROR in one search does not stop the other)
                return self._done(res["forward"], res, t0, grid)
            if fwd.expanded + ext.expanded >= max_exp:
                return self._stop_all(res, searches, PlanStatus.MAX_EXPANSIONS,
                                      f"expanded {max_exp} nodes (both searches)", t0, grid)
            if time.perf_counter() > deadline:
                return self._stop_all(res, searches, PlanStatus.TIMEOUT,
                                      f"exceeded {self.config.max_planning_time}s", t0, grid)
            for name in ("forward", "exit"):
                if res[name] is None:
                    step(name, self.chunk)

    def _refine(self, winner, res, searches, step, t0, grid) -> BidirectionalResult:
        """Give the other direction a bounded grace budget to find a cheaper
        solution (branch & bound on the winner's cost), then pick the cheapest."""
        other = "exit" if winner == "forward" else "forward"
        if res[other] is None:
            # only a clearly better solution is worth the extra search
            searches[other].bound_cost(res[winner].cost * self.improvement_ratio)
            grace = max(self.min_grace_expansions,
                        int(self.grace_ratio * (searches[winner].expanded + searches[other].expanded)))
            limit = searches[other].expanded + grace
            deadline = t0 + self.config.max_planning_time
            limit = min(limit, searches[other].expanded + self._max_exp - sum(x.expanded for x in searches.values()))
            while res[other] is None and searches[other].expanded < limit and time.perf_counter() < deadline:
                step(other, min(self.chunk, limit - searches[other].expanded))
            if res[other] is None:
                r = searches[other].stop(PlanStatus.MAX_EXPANSIONS, "grace budget exhausted "
                                         f"(no cheaper solution than the {winner} search's)")
                res[other] = r if other == "forward" else _finalize_exit(r, self.vehicle, self._start)
        cands = [r for r in res.values() if r is not None and r.success]
        best = min(cands, key=lambda r: (r.cost, r.search != "forward"))
        return self._done(best, res, t0, grid)

    def _stop_all(self, res, searches, status, message, t0, grid) -> BidirectionalResult:
        if res["forward"] is None:
            res["forward"] = searches["forward"].stop(status, message)
        if res["exit"] is None:
            res["exit"] = _finalize_exit(searches["exit"].stop(status, message), self.vehicle, self._start)
        return self._done(res["forward"], res, t0, grid)

    def _check_endpoints(self, res: PlanResult, start, goal) -> None:
        """Last line of defence: the path must start at the start and end inside
        the goal tolerance, whatever search / post-processing produced it."""
        p0, pn = res.trajectory.points[0], res.trajectory.points[-1]
        cfg = self.config
        yaw_err = abs(math.atan2(math.sin(pn.yaw - goal[2]), math.cos(pn.yaw - goal[2])))
        if (math.hypot(p0.x - start[0], p0.y - start[1]) > 1e-2
                or math.hypot(pn.x - goal[0], pn.y - goal[1]) > cfg.position_tolerance + 1e-6
                or yaw_err > cfg.yaw_tolerance + 1e-6):
            res.status = PlanStatus.ERROR
            res.message = "internal inconsistency: path does not connect start and goal"

    def _post_process(self, chosen: PlanResult, grid) -> None:
        """RS shortcutting of the chosen path (removes needless direction switches);
        uses the same map ROI, margins and endpoint relaxation as the search."""
        try:
            pts = chosen.trajectory.points
            start = (pts[0].x, pts[0].y, pts[0].yaw)
            end = (pts[-1].x, pts[-1].y, pts[-1].yaw)
            roi = self.forward.region_of_interest(grid, start, self._goal)
            checker, _ = self.forward.make_zoned_checker(roi, (("start", start), ("goal", end)))
            before = chosen.trajectory.n_direction_changes
            new = shortcut(self.forward, checker, chosen.trajectory)
            if new is not chosen.trajectory:
                chosen.raw_trajectory = chosen.trajectory
                chosen.trajectory = new
                chosen.cost = path_cost(self.forward, checker, new.poses(), new.directions(), new.steerings())
                chosen.message += f" + post-processing (switches {before} -> {new.n_direction_changes})"
        except Exception as exc:  # the validated search output stays the result
            chosen.message += f" (shortcut skipped: {exc!r})"

    def _done(self, chosen: PlanResult, res, t0, grid) -> BidirectionalResult:
        fres, eres = res["forward"], res["exit"]
        both = [r for r in (fres, eres) if r is not None]
        chosen.expanded_nodes = sum(r.expanded_nodes for r in both)
        chosen.generated_nodes = sum(r.generated_nodes for r in both)
        chosen.collision_checks = sum(r.collision_checks for r in both)
        chosen.exact_collision_checks = sum(r.exact_collision_checks for r in both)
        chosen.analytic_attempts = sum(r.analytic_attempts for r in both)
        return BidirectionalResult(chosen, fres, eres)


#: planners selectable by name (run_tests.py --planner)
PLANNERS = {
    "bidirectional": BidirectionalPlanner,
    "forward": StateLatticePlanner,
}
