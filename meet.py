"""Meet-in-the-middle for the bidirectional planner.

The forward tree (start -> goal) and the exit tree (goal region -> start,
time-reversed) usually get close to each other long before either can hit the
other's exact endpoint. Expanded nodes of both trees are binned by position
and heading; when a new node lands next to a node of the other tree, the two
are joined by a Reeds-Shepp curve. A joint path

    start --forward branch--> A --RS--> B --reversed exit branch--> goal region

is checked as a whole (collision with the forward checker, manoeuvre rules on
the complete path) and handed to the forward search as a goal candidate, so the
usual A* termination / branch & bound decides whether it is returned.
"""
from __future__ import annotations

import math

import numpy as np

from maneuver import path_ok
from reeds_shepp import rs_paths, sample_rs_path


class Meeting:
    def __init__(self, fwd, ext, cell: float = 1.0, yaw_cell: float = math.radians(10.0),
                 per_bin: int = 2, max_tries: int = 4, max_candidates: int = 3):
        self.f, self.e = fwd, ext
        self.cell, self.yaw_cell = cell, yaw_cell
        self.per_bin, self.max_tries, self.max_candidates = per_bin, max_tries, max_candidates
        self.bins = ({}, {})            # forward, exit: bin -> [node ids]
        self._tails = {}                # exit node -> reversed branch (or None if it collides)
        self.connections = 0
        fwd.on_expand = lambda n: self._on_expand(0, n)
        ext.on_expand = lambda n: self._on_expand(1, n)

    # ---------------------------------------------------------------- bins
    def _bin(self, nodes, n):
        return (math.floor(nodes.x[n] / self.cell), math.floor(nodes.y[n] / self.cell),
                math.floor((nodes.yaw[n] % (2 * math.pi)) / self.yaw_cell))

    def _on_expand(self, side: int, n: int) -> None:
        if self.f.result is not None:
            return
        search = (self.f, self.e)[side]
        nodes = search.nodes
        bx, by, bk = self._bin(nodes, n)
        mine = self.bins[side].setdefault((bx, by, bk), [])
        mine.append(n)
        mine.sort(key=lambda m: (nodes.sw[m], nodes.g[m]))
        del mine[self.per_bin:]
        other = self.bins[1 - side]
        on = (self.e, self.f)[side].nodes
        cands = [m for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m in other.get((bx + dx, by + dy, bk), ())]
        if not cands:
            return
        x, y, yaw = nodes.x[n], nodes.y[n], nodes.yaw[n]
        cands.sort(key=lambda m: math.hypot(on.x[m] - x, on.y[m] - y)
                   + abs(math.remainder(on.yaw[m] - yaw, 2 * math.pi)))
        for m in cands[:self.max_tries]:
            a, b = (n, m) if side == 0 else (m, n)
            self._connect(a, b)

    # ------------------------------------------------------------ joining
    def _tail(self, b: int):
        """Reversed exit branch root -> b: parking motion b -> goal region."""
        if b not in self._tails:
            poses, dirs, steer = self.e.branch(b)
            n = len(poses)
            rp = poses[::-1].copy()
            rd = np.empty(n, dtype=int)
            rs = np.empty(n)
            rd[1:] = -dirs[::-1][:-1]
            rs[1:] = steer[::-1][:-1]
            rd[0], rs[0] = (rd[1], rs[1]) if n > 1 else (0, 0.0)
            self._tails[b] = None if self.f.checker.any_collision(rp) else (rp, rd, rs)
        return self._tails[b]

    def _connect(self, a: int, b: int) -> None:
        f, e = self.f, self.e
        fn, en = f.nodes, e.nodes
        pl, cfg, rules = f.planner, f.planner.config, f.rules
        g0 = fn.g[a] + en.g[b]
        if g0 >= f.best_goal_cost:
            return
        if rules.cap is not None and fn.sw[a] + en.sw[b] > rules.cap:
            return
        pa = (fn.x[a], fn.y[a], fn.yaw[a])
        pb = (en.x[b], en.y[b], en.yaw[b])
        db = -en.d[b]                   # first parking gear after b (0: b is an exit root)
        scored = []
        for p in rs_paths(pa, pb, pl.vehicle.min_turning_radius):
            if p.shortest_segment() < cfg.min_rs_segment_length:
                continue
            ok, soft, d1, run1, sw1, _ = rules.evaluate_full(p.pieces(pa), pa[0], pa[1], fn.d[a], fn.run[a],
                                                           fn.sw[a], fn.sg[a], fn.cz[a], final=False)
            if not ok:
                continue
            if db != 0 and d1 != db:        # gear change at b
                if not rules.can_switch(run1, sw1 + en.sw[b], rules.near_slot(*pb[:2])):
                    continue
            c, pd, pst = g0 + soft, fn.d[a], fn.st[a]
            for t, l in p.segments():
                sd = 1 if l > 0 else -1
                sst = pl.max_steer if t == "L" else (-pl.max_steer if t == "R" else 0.0)
                c += pl.segment_cost(abs(l), sd, sst, pd, pst, math.inf)
                pd, pst = sd, sst
            if db != 0 and d1 != db:
                c += cfg.direction_switch_penalty + rules.switch_cost(run1)
            if c < f.best_goal_cost:
                scored.append((c, p))
        scored.sort(key=lambda t: t[0])
        for c, p in scored[:self.max_candidates]:
            coarse = sample_rs_path(p, pa, max(0.3, cfg.sample_ds))[0]
            if f.checker.any_collision(coarse):
                continue
            poses, sg, sd, _ = sample_rs_path(p, pa, cfg.sample_ds)
            if len(poses) == 0 or f.checker.any_collision(poses):
                continue
            tail = self._tail(b)
            if tail is None:
                return
            tp, td, ts = tail
            j_poses = np.vstack([poses, tp[1:]])
            j_dirs = np.concatenate([sd, td[1:]])
            j_steer = np.concatenate([sg * pl.max_steer, ts[1:]])
            bp, bd, bs = f.branch(a)
            ok, _ = path_ok(rules, np.vstack([bp, j_poses]), np.concatenate([bd, j_dirs]),
                            np.concatenate([bs, j_steer]))
            if not ok:
                continue
            self.connections += 1
            f.inject_goal(a, tuple(tp[-1]), c, (j_poses, j_steer, j_dirs))
            return
