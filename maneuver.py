"""Manoeuvre rules: what a product-acceptable parking path looks like.

A path is a sequence of *manoeuvres* (maximal runs in one gear). The rules
are hard constraints of the search, not just costs:

* at most ``max_gear_switches`` gear changes;
* a gear change is only allowed after the current manoeuvre is long enough:
  ``min_maneuver_length`` in the aisle / on the road, ``min_slot_maneuver_length``
  close to the parking slot (within ``slot_zone_radius`` of the slot pose),
  where the free space itself is short (in-slot corrections);
* the same applies to the final manoeuvre.

Soft costs make the result natural: a gear change after a manoeuvre shorter
than ``preferred_maneuver_length`` and every steering reversal (left <-> right)
inside one manoeuvre are penalised.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Tuple


class ManeuverRules:
    def __init__(self, cfg, vehicle, slot_pose, cap: Optional[int]):
        self.cfg = cfg
        self.cap = cap
        self.slot = slot_pose
        self.zone = cfg.slot_zone_radius if cfg.slot_zone_radius is not None else 0.5 * vehicle.vehicle_length
        self.mn_road = cfg.min_maneuver_length
        self.mn_slot = min(cfg.min_slot_maneuver_length, cfg.min_maneuver_length)

    # ------------------------------------------------------------ basics
    def near_slot(self, x: float, y: float) -> bool:
        return math.hypot(x - self.slot[0], y - self.slot[1]) <= self.zone

    def min_run(self, near: bool) -> float:
        return self.mn_slot if near else self.mn_road

    def run_class(self, run: float) -> int:
        """0: shorter than any minimum, 1: enough near the slot, 2: enough anywhere."""
        if run >= self.mn_road - 1e-9:
            return 2
        return 1 if run >= self.mn_slot - 1e-9 else 0

    def switch_cost(self, run: float) -> float:
        pref = self.cfg.preferred_maneuver_length
        return self.cfg.short_maneuver_weight * max(0.0, 1.0 - run / pref) if pref > 0 else 0.0

    def can_switch(self, run: float, sw: int, near: bool) -> bool:
        return run >= self.min_run(near) - 1e-9 and (self.cap is None or sw < self.cap)

    def final_ok(self, d: int, run: float, sw: int, cusp_near: bool) -> Tuple[bool, float]:
        """The last manoeuvre (after the last gear change) must be long enough too."""
        if d == 0 or sw == 0:
            return True, 0.0
        if run < self.min_run(cusp_near) - 1e-9:
            return False, 0.0
        return True, self.switch_cost(run)

    # --------------------------------------------------- analytic pieces
    def evaluate(self, runs: Iterable, x0: float, y0: float, d: int, run: float, sw: int, sg: int,
                 cusp_near: bool, final: bool) -> Tuple[bool, float]:
        ok, soft, *_ = self.evaluate_full(runs, x0, y0, d, run, sw, sg, cusp_near, final)
        return ok, soft

    def evaluate_full(self, runs: Iterable, x0: float, y0: float, d: int, run: float, sw: int, sg: int,
                      cusp_near: bool, final: bool):
        """Apply the rules to a sequence of manoeuvre pieces continuing a path.

        ``runs``: [(direction, length, steer_signs, end_x, end_y), ...] where
        steer_signs is the ordered list of steering signs (-1/0/+1) of the piece.
        State at (x0, y0): gear d, current run length, gear changes sw, last
        non-zero steer sign sg of the current manoeuvre, whether its gear change
        was near the slot.
        Returns (ok, soft_cost, d, run, sw, cusp_near) - the last four are the end state.
        """
        soft = 0.0
        x, y = x0, y0
        for rd, rl, signs, ex, ey in runs:
            if d != 0 and rd != d:          # gear change at (x, y)
                cusp_near = self.near_slot(x, y)
                if not self.can_switch(run, sw, cusp_near):
                    return False, 0.0, d, run, sw, cusp_near
                soft += self.switch_cost(run)
                sw, run, sg = sw + 1, 0.0, 0
            if d == 0:                      # first manoeuvre of the path starts here
                run = 0.0
            d = rd
            run += rl
            for s in signs:
                if s != 0:
                    if sg != 0 and s != sg:
                        soft += self.cfg.steer_reversal_penalty
                    sg = s
            x, y = ex, ey
        if final:
            ok, c = self.final_ok(d, run, sw, cusp_near)
            if not ok:
                return False, 0.0, d, run, sw, cusp_near
            soft += c
        return True, soft, d, run, sw, cusp_near


def pieces_from_path(poses, dirs, steer):
    """Manoeuvre pieces of a sampled path (point i carries the motion arriving at it)."""
    pieces = []
    for i in range(1, len(poses)):
        ds = math.hypot(poses[i][0] - poses[i - 1][0], poses[i][1] - poses[i - 1][1])
        s = 1 if steer[i] > 1e-6 else (-1 if steer[i] < -1e-6 else 0)
        pieces.append((int(dirs[i]), ds, [s], poses[i][0], poses[i][1]))
    return pieces


def path_ok(rules: ManeuverRules, poses, dirs, steer) -> Tuple[bool, float]:
    """Check a complete path against the rules; returns (ok, soft_cost)."""
    if len(poses) < 2:
        return True, 0.0
    return rules.evaluate(pieces_from_path(poses, dirs, steer), poses[0][0], poses[0][1],
                          0, math.inf, 0, 0, True, final=True)
