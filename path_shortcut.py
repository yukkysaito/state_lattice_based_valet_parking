"""Reeds-Shepp shortcutting: remove unnecessary direction switches.

A lattice search produces paths made of fixed primitives. Near the goal (and in
tight spaces) this often leaves small shuffles ("reverse - 0.3 m forward -
reverse") that only fix sub-cell position errors. This post-processing step
replaces a piece of the path between two of its poses by a Reeds-Shepp curve
whenever that is

* cheaper under the planner's own cost function (length / reverse weight /
  steering / *direction switches* / clearance), without adding a switch, and
* collision-free with the planner's own checker (same margin, same continuous
  guarantee, same endpoint relaxation).

It is greedy and deterministic: starting at the first pose, it tries the
farthest reachable anchor first and splices the first improvement. Endpoints
(start, final pose) are never changed. The result is re-validated as a whole.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from maneuver import ManeuverRules, path_ok
from reeds_shepp import rs_paths, sample_rs_path
from trajectory import Trajectory


def _arrays(traj: Trajectory):
    return traj.poses(), traj.directions(), traj.steerings()


def path_cost(planner, checker, poses: np.ndarray, dirs: np.ndarray, steer: np.ndarray,
              prev_dir: int = 0, prev_steer: float = 0.0) -> float:
    """Planner cost of a sampled path (point i carries the motion arriving at it)."""
    cfg = planner.config
    if len(poses) < 2:
        return 0.0
    ds = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    d, st = dirs[1:], steer[1:]
    c = float(np.sum(ds * np.where(d > 0, cfg.forward_weight, cfg.reverse_weight)))
    c += cfg.steer_weight * float(np.sum(ds * np.abs(st))) / planner.max_steer
    pd = np.concatenate([[prev_dir], d[:-1]])
    pst = np.concatenate([[prev_steer], st[:-1]])
    moving = pd != 0
    c += cfg.direction_switch_penalty * float(np.sum(moving & (d != pd)))
    c += cfg.steer_change_weight * float(np.sum(np.abs(st - pst)[moving])) / planner.max_steer
    c += cfg.steer_reversal_penalty * _steer_reversals(d, st)
    if cfg.clearance_weight > 0 and cfg.clearance_ref > 0:
        clr = checker.clearance_estimate(poses[1:])
        c += cfg.clearance_weight * float(np.sum(ds * np.maximum(0.0, 1.0 - clr / cfg.clearance_ref)))
    return c


def _steer_reversals(d: np.ndarray, st: np.ndarray) -> int:
    """Number of left <-> right steering reversals within manoeuvres."""
    n, last, gear = 0, 0, None
    for di, si in zip(d, st):
        if di != gear:
            gear, last = di, 0
        s = 1 if si > 1e-6 else (-1 if si < -1e-6 else 0)
        if s:
            n += last != 0 and s != last
            last = s
    return n


def count_switches(dirs: np.ndarray, prev_dir: int = 0) -> int:
    d = dirs[1:]
    pd = np.concatenate([[prev_dir], d[:-1]])
    return int(np.sum((pd != 0) & (d != pd)))


def min_run_length(poses: np.ndarray, dirs: np.ndarray) -> float:
    """Shortest forward/reverse run of a sampled path (point i carries the motion arriving at it)."""
    if len(poses) < 2:
        return math.inf
    ds = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    d = dirs[1:]
    runs, cur = [], 0.0
    for k in range(len(d)):
        if k > 0 and d[k] != d[k - 1]:
            runs.append(cur)
            cur = 0.0
        cur += ds[k]
    runs.append(cur)
    return min(runs)


def min_steer_hold(poses: np.ndarray, dirs: np.ndarray, steer: np.ndarray) -> float:
    """Shortest distance a steering sign is held before it changes (within a manoeuvre)."""
    ds = np.hypot(np.diff(poses[:, 0]), np.diff(poses[:, 1]))
    sg = np.where(steer[1:] > 1e-6, 1, np.where(steer[1:] < -1e-6, -1, 0))
    d = dirs[1:]
    best, cur = math.inf, 0.0
    for k in range(len(ds)):
        if k > 0 and (sg[k] != sg[k - 1] or d[k] != d[k - 1]):
            if d[k] == d[k - 1]:
                best = min(best, cur)
            cur = 0.0
        cur += ds[k]
    return best


def _rs_lower_bound(planner, path, prev_dir: int, prev_steer: float) -> float:
    """Planner cost of an RS path without the (non-negative) clearance term."""
    c, pd, pst = 0.0, prev_dir, prev_steer
    for t, l in path.segments():
        sd = 1 if l > 0 else -1
        sst = planner.max_steer if t == "L" else (-planner.max_steer if t == "R" else 0.0)
        c += planner.segment_cost(abs(l), sd, sst, pd, pst, math.inf)
        pd, pst = sd, sst
    return c


def shortcut(planner, checker, traj: Trajectory, anchor_spacing: float = 0.4,
             max_anchors: int = 80, min_gain: float = 1e-3) -> Trajectory:
    """Return a trajectory with RS shortcuts applied (or the input if none helps)."""
    if traj is None or len(traj) < 3:
        return traj
    cfg = planner.config
    poses, dirs, steer = _arrays(traj)
    r = planner.vehicle.min_turning_radius
    # anchors: cusps, segment ends and points every ``anchor_spacing`` metres
    s = np.array([p.arc_length for p in traj.points])
    idx = set(np.searchsorted(s, np.arange(0.0, s[-1], anchor_spacing)).tolist())
    idx |= {i for i in range(1, len(poses)) if dirs[i] != dirs[i - 1] or steer[i] != steer[i - 1]}
    idx |= {0, len(poses) - 1}
    anchors = sorted(i for i in idx if i < len(poses))
    if len(anchors) > max_anchors:
        keep = np.linspace(0, len(anchors) - 1, max_anchors).round().astype(int)
        anchors = sorted({anchors[k] for k in keep} | {0, len(poses) - 1})

    rules = ManeuverRules(cfg, planner.vehicle, tuple(poses[-1]), cfg.max_gear_switches)
    base_hold = min_steer_hold(poses, dirs, steer)
    out_p: List[np.ndarray] = [poses[:1]]
    out_d: List[np.ndarray] = [dirs[:1]]
    out_s: List[np.ndarray] = [steer[:1]]
    a = 0
    while a < len(anchors) - 1:
        i = anchors[a]
        prev_dir = int(out_d[-1][-1]) if a > 0 else 0
        prev_steer = float(out_s[-1][-1]) if a > 0 else 0.0
        done = False
        for b in range(len(anchors) - 1, a + 1, -1):  # farthest first; a+1 is the original step
            j = anchors[b]
            orig = path_cost(planner, checker, poses[i:j + 1], dirs[i:j + 1], steer[i:j + 1], prev_dir, prev_steer)
            best = None
            for p in rs_paths(tuple(poses[i]), tuple(poses[j]), r)[:cfg.analytic_max_candidates]:
                if p.shortest_segment() < cfg.min_rs_segment_length:
                    continue
                if _rs_lower_bound(planner, p, prev_dir, prev_steer) >= orig - min_gain:
                    continue  # cannot beat the original piece (cheap test before sampling)
                sp, sg, sd, _ = sample_rs_path(p, tuple(poses[i]), cfg.sample_ds)
                if len(sp) == 0:
                    continue
                cand_p = np.vstack([poses[i:i + 1], sp])
                cand_d = np.concatenate([[sd[0]], sd])
                cand_s = np.concatenate([[sg[0]], sg]) * planner.max_steer
                c = path_cost(planner, checker, cand_p, cand_d, cand_s, prev_dir, prev_steer)
                if c >= orig - min_gain or (best is not None and c >= best[0]):
                    continue
                # the whole spliced path must still satisfy the manoeuvre rules
                full_p = np.vstack(out_p + [sp, poses[j + 1:]])
                full_d = np.concatenate(out_d + [sd, dirs[j + 1:]])
                full_s = np.concatenate(out_s + [sg * planner.max_steer, steer[j + 1:]])
                if not path_ok(rules, full_p, full_d, full_s)[0]:
                    continue
                # ... and must not leave a short steering fragment at a junction
                if min_steer_hold(full_p, full_d, full_s) < min(cfg.min_rs_segment_length, base_hold) - 1e-9:
                    continue
                # the purpose is to remove manoeuvres: never add a switch (junctions included)
                if count_switches(full_d) > count_switches(np.concatenate(out_d + [dirs[i + 1:]])):
                    continue
                if not checker.any_collision(sp):
                    best = (c, sp, sd, sg * planner.max_steer)
            if best is not None:
                _, sp, sd, sst = best
                out_p.append(sp)
                out_d.append(sd)
                out_s.append(sst)
                a = b
                done = True
                break
        if not done:
            j = anchors[a + 1]
            out_p.append(poses[i + 1:j + 1])
            out_d.append(dirs[i + 1:j + 1])
            out_s.append(steer[i + 1:j + 1])
            a += 1
    new_p = np.vstack(out_p)
    new_d = np.concatenate(out_d)
    new_s = np.concatenate(out_s)
    new_d[0], new_s[0] = new_d[1], new_s[1]
    # keep the exact final pose (RS endpoints match to ~1e-9, copy for bit-exactness)
    new_p[-1] = poses[-1]
    if checker.check_poses(new_p).any():  # defensive: never return a colliding path
        return traj
    return Trajectory.from_arrays(new_p, new_d, new_s, planner.vehicle.wheel_base)
