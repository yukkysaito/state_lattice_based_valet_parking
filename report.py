"""Reports of a benchmark run.

Two separate questions:

* **Tests** - does the planner show the *expected behaviour*?  (pass / fail)
  e.g. "goal in an obstacle -> GOAL_IN_COLLISION", "straight goal -> success
  with 0 gear changes", "slot too tight -> NO_PATH within 20 s".
* **Benchmark** - how *good* is it?  Metrics per difficulty (easy / normal /
  hard), with the number of gear changes as the headline quality metric.
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List

import numpy as np

DIFFICULTIES = ("easy", "normal", "hard")


def _num(v):
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _ok(r) -> bool:
    return r.get("success") in (True, "True")


# ------------------------------------------------------------------ tests
def test_report(rows: List[Dict]) -> List[str]:
    tests = [r for r in rows if r.get("is_test") in (True, "True")]
    passed = [r for r in tests if r.get("passed") in (True, "True")]
    out = ["", f"== Tests: {len(passed)}/{len(tests)} passed ==",
           f"  {'test':<30}{'expected':<44}{'result':<34}pass"]
    for r in tests:
        res = r.get("status", "")
        if _ok(r):
            res = f"success, {int(_num(r.get('direction_changes')) or 0)} gear change(s)"
        name = f"{r['id']} {r.get('name', '')}"[:29]
        out.append(f"  {name:<30}{str(r.get('expected', ''))[:43]:<44}{res[:33]:<34}"
                   f"{'ok' if r in passed else 'FAIL'}")
    for r in tests:
        if r not in passed:
            out.append(f"  FAIL {r['id']}: {r.get('fail_reason', '')}")
    return out


# -------------------------------------------------------------- benchmark
def _stats(rs: List[Dict]) -> Dict:
    ok = [r for r in rs if _ok(r)]
    sw = [int(_num(r.get("direction_changes")) or 0) for r in ok]
    t = [_num(r.get("planning_time_ms")) for r in rs if _num(r.get("planning_time_ms")) is not None]
    road = [_num(r.get("min_road_maneuver")) for r in ok if _num(r.get("min_road_maneuver")) is not None]
    slot = [_num(r.get("min_slot_maneuver")) for r in ok if _num(r.get("min_slot_maneuver")) is not None]
    rev = [int(_num(r.get("max_steer_reversals")) or 0) for r in ok]
    hold = [_num(r.get("min_steer_hold")) for r in ok if _num(r.get("min_steer_hold")) is not None]
    clr = [_num(r.get("min_clearance")) for r in ok if _num(r.get("min_clearance")) is not None]
    ge = [_num(r.get("goal_position_error")) for r in ok if _num(r.get("goal_position_error")) is not None]
    ln = [_num(r.get("trajectory_length")) for r in ok if _num(r.get("trajectory_length")) is not None]
    short = [int(_num(r.get("short_maneuvers")) or 0) for r in ok]
    hist = [sum(1 for s in sw if s == k) for k in range(4)] + [sum(1 for s in sw if s >= 4)]
    return dict(n=len(rs), solved=len(ok), sw=sw, hist=hist, t=t, road=road, slot=slot, rev=rev, hold=hold,
                clr=clr, ge=ge, ln=ln, short=short)


def spec_report(rows: List[Dict]) -> List[str]:
    """Product-specification compliance (the acceptance KPI)."""
    rs = [r for r in rows if r.get("spec_pass") is not None]
    if not rs:
        return []
    out = ["", "== Product specification compliance ==",
           f"  {'difficulty':<11}{'pass':>14}"]
    for d in DIFFICULTIES + ("all",):
        g = [r for r in rs if d == "all" or r.get("difficulty") == d]
        if g:
            n = sum(1 for r in g if r.get("spec_pass"))
            out.append(f"  {d:<11}{n:>6}/{len(g):<4} {100 * n / len(g):>3.0f}%")
    fails = [r for r in rs if not r.get("spec_pass")]
    for r in fails:
        out.append(f"  FAIL {r['id']:<14} ({r.get('difficulty')}, spec <= {r.get('spec_max_switches')} changes): "
                   f"{r.get('spec_fail')}")
    return out


def benchmark_report(rows: List[Dict]) -> List[str]:
    bench = [r for r in rows if r.get("difficulty") in DIFFICULTIES]
    if not bench:
        return []
    groups = [(d, [r for r in bench if r.get("difficulty") == d]) for d in DIFFICULTIES] + [("all", bench)]
    out = ["", f"== Benchmark ({len(bench)} scenarios; tests and benchmark overlap) ==", "",
           "  Quality (solved cases). road/slot manoeuvre: shortest move before a gear change in the",
           "  aisle / within half a vehicle length of the slot; steer hold: shortest distance a steering",
           "  direction is kept (a few cm = steering spike)",
           f"  {'difficulty':<11}{'solved':>12}{'<=1 change':>12}{'gear changes':>15}{'road move':>11}"
           f"{'slot move':>11}{'steer hold':>12}{'min clear':>11}{'goal err':>10}{'length':>9}"]
    for name, rs in groups:
        s = _stats(rs)
        if not s["n"]:
            continue
        sw = s["sw"]
        solved = f"{s['solved']}/{s['n']} {100 * s['solved'] / s['n']:.0f}%"
        changes = f"{statistics.fmean(sw):.2f} (max {max(sw)})" if sw else "-"
        le1 = f"{100 * sum(1 for x in sw if x <= 1) / len(sw):.0f}%" if sw else "-"
        road = f"{min(s['road']):.2f} m" if s["road"] else "-"
        slot = f"{min(s['slot']):.2f} m" if s["slot"] else "-"
        hold = f"{min(s['hold']):.2f} m" if s["hold"] else "-"
        clr = f"{min(s['clr']):.2f} m" if s["clr"] else "-"
        ge = f"{max(s['ge']):.3f}" if s["ge"] else "-"
        ln = f"{statistics.fmean(s['ln']):.1f}" if s["ln"] else "-"
        out.append(f"  {name:<11}{solved:>12}{le1:>12}{changes:>15}{road:>11}{slot:>11}{hold:>12}{clr:>11}{ge:>10}"
                   f"{ln:>9}")
    out += ["", "  Gear changes per solved case (number of cases)",
            f"  {'difficulty':<11}{'0':>6}{'1':>6}{'2':>6}{'3':>6}{'>=4':>6}   unsolved"]
    for name, rs in groups:
        s = _stats(rs)
        if s["n"]:
            out.append(f"  {name:<11}" + "".join(f"{h:>6}" for h in s["hist"]) + f"   {s['n'] - s['solved']:>6}")
    out += ["", "  Planning time (all cases)",
            f"  {'difficulty':<11}{'median':>10}{'p95':>10}{'max':>10}"]
    for name, rs in groups:
        s = _stats(rs)
        if s["t"]:
            out.append(f"  {name:<11}{statistics.median(s['t']):>8.0f}ms{np.percentile(s['t'], 95):>8.0f}ms"
                       f"{max(s['t']):>8.0f}ms")
    # gear changes by scene type
    def scene(r):
        if r.get("category") == "random":
            return "random-" + ("parallel" if "parallel" in str(r.get("name", "")) else "perpendicular")
        return r.get("category", "")
    scenes = sorted({scene(r) for r in bench})
    out += ["", "  Mean gear changes by scene type (solved / cases)",
            f"  {'scene':<24}" + "".join(f"{d:>16}" for d in DIFFICULTIES)]
    for sc in scenes:
        cells = []
        for d in DIFFICULTIES:
            rs = [r for r in bench if scene(r) == sc and r.get("difficulty") == d]
            s = _stats(rs)
            cells.append(f"{statistics.fmean(s['sw']):.2f} ({s['solved']}/{s['n']})" if s["sw"] else
                         (f"- (0/{s['n']})" if s["n"] else ""))
        out.append(f"  {sc:<24}" + "".join(f"{c:>16}" for c in cells))
    unsolved = [r for r in bench if not _ok(r)]
    if unsolved:
        out += ["", "  Unsolved: " + ", ".join(f"{r['id']}({r.get('difficulty')}, {r.get('status')})"
                                              for r in unsolved)]
    return out
