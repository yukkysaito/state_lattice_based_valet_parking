#!/usr/bin/env python3
"""Scenario runner: behaviour tests (pass/fail) and the performance benchmark.

python run_tests.py                          # all fixed scenarios
python run_tests.py --all --random 100       # fixed + 100 random scenarios
python run_tests.py --category parallel      # one category (name or letter)
python run_tests.py --ids B2 C3              # selected scenarios
python run_tests.py --regression [--update-baseline]
python run_tests.py --set min_maneuver_length=1.5 --planner forward

Results: <out>/summary.txt, <out>/benchmark.csv, plots in <out>/tests/(pass|fail)/
and <out>/benchmark/<difficulty>/(switch_<n>|unsolved)/.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from collision_checker import CollisionChecker  # noqa: E402
from metrics import IndependentValidator, trajectory_metrics  # noqa: E402
from state_lattice import PlannerConfig, PlanStatus, StateLatticePlanner  # noqa: E402
from vehicle import InvalidVehicleParameterError  # noqa: E402

# Product specification checked by the benchmark (per-scene gear-change limits: scenarios.spec_of)
SPEC = dict(min_road_maneuver=1.0, min_slot_maneuver=0.4, min_steer_hold=0.3, goal_position_error=0.10,
            goal_yaw_error_deg=2.0, min_clearance=0.095,
            max_expansions={"easy": 100_000, "normal": 100_000, "hard": 200_000})

REGRESSION_IDS = ["B2", "B3", "C2", "D3", "E5", "D2"]
REGRESSION_FILE = os.path.join(HERE, "regression_baseline.json")
REGRESSION_THRESHOLDS = {"extra_switches": 0, "length_ratio": 1.15, "nodes_ratio": 2.0, "nodes_abs": 500}

CSV_FIELDS = [
    "index", "id", "name", "category", "difficulty", "is_test", "expected", "passed", "fail_reason",
    "success", "status", "search", "planning_time_ms", "expanded_nodes", "direction_changes",
    "min_road_maneuver", "min_slot_maneuver", "min_steer_hold", "max_steer_reversals", "min_clearance",
    "goal_position_error", "goal_yaw_error_deg", "trajectory_length", "forward_length", "reverse_length",
    "max_curvature", "max_steering_deg", "validated_collision_free", "point_robot_success",
    "point_robot_collides", "spec_max_switches", "spec_pass", "spec_fail", "vehicle", "group", "plot", "message",
]


# ---------------------------------------------------------------- one case
def run_case(args) -> Dict:
    """Worker entry: rebuild the scenario by id (deterministic), plan, evaluate."""
    index, sid, seed, overrides, out_dir, save_plot, planner_name = args
    from parking_planner import PLANNERS
    from scenarios import build_scenario

    row: Dict = {"index": index, "id": sid}
    try:
        sc = build_scenario(sid, seed=seed)
        exp = sc.expectation
        row.update(name=sc.name, category=sc.category, group=sc.group, is_test=sc.is_test,
                   difficulty=sc.difficulty, expected=expected_text(exp))
        try:
            vehicle = sc.make_vehicle()
        except (InvalidVehicleParameterError, ValueError) as exc:  # G6: must raise
            row.update(success=False, status="invalid_vehicle", message=str(exc), planning_time_ms=0.0,
                       passed=exp.expect_exception,
                       fail_reason="" if exp.expect_exception else f"unexpected vehicle error: {exc}")
            return row
        if exp.expect_exception:
            row.update(success=False, status="no_exception", passed=False,
                       fail_reason="invalid vehicle parameters were accepted")
            return row
        cfg = sc.make_config(PlannerConfig().with_changes(**overrides))
        res = PLANNERS[planner_name](vehicle, cfg).plan(sc.grid, sc.start, sc.goal)
        row.update(success=res.success, status=res.status.value, search=res.search, vehicle=vehicle.name,
                   message=res.message.splitlines()[0] if res.message else "",
                   planning_time_ms=res.planning_time * 1e3, expanded_nodes=res.expanded_nodes)
        row.update(trajectory_metrics(res.trajectory if res.success else None, sc.goal, vehicle,
                                      CollisionChecker(sc.grid, vehicle, "rectangle", margin=0.0),
                                      short_maneuver=cfg.preferred_maneuver_length))
        validator = IndependentValidator(sc.grid, vehicle, margin=0.0)
        if res.success:
            row["validated_collision_free"] = validator.is_collision_free(res.trajectory.poses())
        if exp.point_robot_should_succeed:  # F4: the point model must be shown to be wrong
            pres = StateLatticePlanner(vehicle, cfg.with_changes(collision_method="point")).plan(
                sc.grid, sc.start, sc.goal)
            row["point_robot_success"] = pres.success
            row["point_robot_collides"] = (not validator.is_collision_free(pres.trajectory.poses())
                                           if pres.success else None)
        row["passed"], row["fail_reason"] = evaluate(row, exp, cfg)
        if sc.spec_max_switches is not None:
            row["spec_max_switches"] = sc.spec_max_switches
            row["spec_pass"], row["spec_fail"] = spec_check(row, sc.spec_max_switches, sc.spec_reject_ok, sc.difficulty)
        if save_plot:
            _save_plot(sc, vehicle, res, row, out_dir, index)
    except Exception as exc:  # the runner must never die on one case
        row.update(success=False, status="runner_error", passed=False,
                   fail_reason=f"runner exception {type(exc).__name__}: {exc}",
                   message=traceback.format_exc().splitlines()[-1])
    return row


def expected_text(exp) -> str:
    if exp.expect_exception:
        return "raise InvalidVehicleParameterError"
    if exp.expected_status:
        return f"reject: {exp.expected_status}" + (f" within {exp.max_time:g} s" if exp.max_time else "")
    if exp.success is False:
        return "fail safely" + (f" within {exp.max_time:g} s" if exp.max_time else "")
    if exp.success is None:
        return ""
    parts = ["success"]
    if exp.max_direction_changes is not None:
        parts.append(f"<= {exp.max_direction_changes} gear changes")
    if exp.min_direction_changes:
        parts.append(f">= {exp.min_direction_changes}")
    if exp.allowed_directions:
        parts.append(f"{exp.allowed_directions} only")
    return ", ".join(parts)


def evaluate(row: Dict, exp, cfg) -> tuple:
    """(passed, reason) of a case against its expected behaviour."""
    status = row.get("status")
    if status == PlanStatus.ERROR.value:
        return False, "planner error: " + str(row.get("message", ""))
    if row.get("success") and not row.get("validated_collision_free", False):
        return False, "independent validator found a collision"
    slow = exp.max_time is not None and row.get("planning_time_ms", 0) > exp.max_time * 1e3
    if exp.expected_status is not None:
        if status != exp.expected_status:
            return False, f"expected {exp.expected_status}, got {status}"
        return (False, "rejection too slow") if slow else (True, "")
    if exp.point_robot_should_succeed and row.get("point_robot_success") and not row.get("point_robot_collides"):
        return False, "point-robot path unexpectedly collision-free (F4 scene not discriminative)"
    if exp.success is False:
        if row.get("success"):
            return False, "expected failure but planner succeeded"
        return (False, f"failure took {row['planning_time_ms']:.0f} ms") if slow else (True, "")
    if exp.success is None:
        return True, ""
    if not row.get("success"):
        return False, f"planning failed ({status})"
    reasons = []
    sw = row["direction_changes"]
    if exp.max_direction_changes is not None and sw > exp.max_direction_changes:
        reasons.append(f"{sw} gear changes > {exp.max_direction_changes}")
    if exp.min_direction_changes is not None and sw < exp.min_direction_changes:
        reasons.append(f"{sw} gear changes < {exp.min_direction_changes}")
    if exp.allowed_directions == "forward" and row["reverse_length"] > 1e-6:
        reasons.append("used reverse")
    if exp.allowed_directions == "reverse" and row["forward_length"] > 1e-6:
        reasons.append("used forward")
    if row["goal_position_error"] > cfg.position_tolerance + 1e-6:
        reasons.append("goal position error")
    if row["goal_yaw_error_deg"] > math.degrees(cfg.yaw_tolerance) + 1e-6:
        reasons.append("goal yaw error")
    if slow:
        reasons.append("too slow")
    return (not reasons), "; ".join(reasons)


def spec_check(row: Dict, max_switches: int, reject_ok: bool, difficulty: str) -> tuple:
    """(passed, reasons) of a planned scenario against the product specification."""
    nodes = row.get("expanded_nodes") or 0
    budget = SPEC["max_expansions"][difficulty]
    if not row.get("success"):
        # a rejection is fine where allowed, if it is an answer within the budget
        if reject_ok and row.get("status") in ("no_path", "max_expansions") and nodes <= budget:
            return True, ""
        return False, f"not solved ({row.get('status')})" + (" [rejection allowed but too slow]" if reject_ok else "")
    bad = []
    if row["direction_changes"] > max_switches:
        bad.append(f"{row['direction_changes']} gear changes > {max_switches}")
    for k, lim in (("min_road_maneuver", SPEC["min_road_maneuver"]), ("min_slot_maneuver", SPEC["min_slot_maneuver"]),
                   ("min_steer_hold", SPEC["min_steer_hold"]), ("min_clearance", SPEC["min_clearance"])):
        v = row.get(k)
        if v is not None and v == v and v < lim - 0.005:  # lengths are measured on 2.5 cm samples
            if k == "min_clearance" and "relaxed" in str(row.get("message", "")):
                continue  # start/goal closer than the margin: accepted by design
            bad.append(f"{k} {v:.2f} < {lim}")
    for k in ("goal_position_error", "goal_yaw_error_deg"):
        if row.get(k, 0) > SPEC[k] + 1e-9:
            bad.append(f"{k} {row[k]:.3f} > {SPEC[k]}")
    if nodes > budget:
        bad.append(f"{nodes} expansions > {budget}")
    return (not bad), "; ".join(bad)


def _save_plot(sc, vehicle, res, row, out_dir, index):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from visualization import plot_result

    fig, _ = plot_result(sc.grid, vehicle, sc.start, sc.goal, res, sc.obstacles,
                         title=f"{sc.id} {sc.name} [{sc.difficulty}] - {vehicle.summary()}", metrics=row)
    dirs = []
    if row.get("is_test"):
        dirs.append(os.path.join("tests", "pass" if row.get("passed") else "fail"))
    if row.get("difficulty") in ("easy", "normal", "hard"):
        sub = f"switch_{int(row.get('direction_changes') or 0)}" if row.get("success") else "unsolved"
        dirs.append(os.path.join("benchmark", row["difficulty"], sub))
    paths = [os.path.join(out_dir, d, f"case_{index:03d}_{sc.id}.png") for d in dirs]
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=90)
    plt.close(fig)
    row["plot"] = ";".join(os.path.relpath(p, out_dir) for p in paths)


# ---------------------------------------------------------------- output
def _fmt(v, fmt="{:.1f}"):
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def sweep_tables(rows: List[Dict]) -> List[str]:
    """Parameter sweeps (I/J/K/L groups): one small table per group."""
    groups: Dict[str, List[Dict]] = {}
    for r in rows:
        if r.get("group") and r.get("category") != "random":
            groups.setdefault(r["group"], []).append(r)
    lines = ["", "== Parameter sweeps =="] if groups else []
    for g, rs in groups.items():
        lines.append(f"[{g}]  {'ok':>4}{'time ms':>9}{'nodes':>8}{'changes':>8}{'road move':>10}{'length':>8}")
        for r in rs:
            ok = r.get("success")
            lines.append(f"  {r['id']:<18}{'yes' if ok else 'no':>4}{_fmt(r.get('planning_time_ms'), '{:.0f}'):>9}"
                         f"{_fmt(r.get('expanded_nodes'), '{:d}'):>8}"
                         f"{_fmt(r.get('direction_changes') if ok else None, '{:d}'):>8}"
                         f"{_fmt(r.get('min_road_maneuver') if ok else None, '{:.2f}'):>10}"
                         f"{_fmt(r.get('trajectory_length') if ok else None):>8}")
    return lines


def write_csv(rows: List[Dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


# ---------------------------------------------------------------- regression
def regression_check(rows: List[Dict], update: bool) -> List[str]:
    by_id = {r["id"]: r for r in rows}
    if update:
        base = {}
        for sid in REGRESSION_IDS:
            r = by_id.get(sid)
            if not r or not r.get("success"):
                raise SystemExit(f"cannot record baseline: {sid} did not succeed")
            base[sid] = {k: r[k] for k in ("trajectory_length", "direction_changes", "expanded_nodes")}
        with open(REGRESSION_FILE, "w") as f:
            json.dump({"cases": base, "thresholds": REGRESSION_THRESHOLDS}, f, indent=2)
        return ["", f"regression baseline written to {REGRESSION_FILE}"]
    if not os.path.exists(REGRESSION_FILE):
        return ["", "no regression baseline (run with --regression --update-baseline)"]
    with open(REGRESSION_FILE) as f:
        data = json.load(f)
    lines = ["", "== Regression check =="]
    ok_all = True
    for sid, b in data["cases"].items():
        problems = regression_problems(by_id.get(sid), b, data.get("thresholds", REGRESSION_THRESHOLDS))
        ok_all &= not problems
        lines.append(f"  {sid:<5} {'OK' if not problems else 'FAIL ' + '; '.join(problems)}")
    lines.append("  REGRESSION " + ("PASSED" if ok_all else "FAILED"))
    return lines


def regression_problems(r: Optional[Dict], b: Dict, th: Dict) -> List[str]:
    """Deterministic checks only (no wall-clock time)."""
    if r is None:
        return ["case not run"]
    if not r.get("success"):
        return [f"not successful ({r.get('status')})"]
    p = []
    if not r.get("validated_collision_free"):
        p.append("collision")
    if r["direction_changes"] > b["direction_changes"] + th["extra_switches"]:
        p.append(f"{r['direction_changes']} gear changes > {b['direction_changes']}")
    if r["trajectory_length"] > b["trajectory_length"] * th["length_ratio"]:
        p.append(f"length {r['trajectory_length']:.1f} > {th['length_ratio']} x {b['trajectory_length']:.1f}")
    if r["expanded_nodes"] > max(b["expanded_nodes"] * th["nodes_ratio"], b["expanded_nodes"] + th["nodes_abs"]):
        p.append(f"expanded {r['expanded_nodes']} nodes > {th['nodes_ratio']} x {b['expanded_nodes']}")
    return p


# ---------------------------------------------------------------- warm-up
def _build_tables(job):
    """Build (and disk-cache) the heuristic tables one planner configuration needs."""
    sid, seed, overrides = job
    from parking_planner import BidirectionalPlanner
    from scenarios import build_scenario
    try:
        sc = build_scenario(sid, seed=seed)
        BidirectionalPlanner(sc.make_vehicle(), sc.make_config(PlannerConfig().with_changes(**overrides)))
    except Exception:
        pass
    return sid


def _warmup(scs, overrides, workers, seed):
    """Compile numba kernels and build the heuristic tables before timing starts
    (in parallel into the disk cache, then loaded in the parent so that forked
    workers inherit them): per-case times exclude this offline work."""
    uniq = {}
    for sc in scs:
        try:
            c = sc.make_config(PlannerConfig().with_changes(**overrides))
            key = (sc.make_vehicle().key(), c.primitive_config(), c.reverse_weight, c.steer_weight,
                   c.direction_switch_penalty, c.max_gear_switches, c.position_tolerance)
        except Exception:
            continue
        uniq.setdefault(key, sc.id)
    jobs = [(sid, seed, overrides) for sid in uniq.values()]
    t0 = time.perf_counter()
    if workers > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(min(workers, len(jobs))) as pool:
            list(pool.imap_unordered(_build_tables, jobs))
    for j in jobs:
        _build_tables(j)
    print(f"heuristic tables ready for {len(jobs)} configuration(s) in {time.perf_counter() - t0:.1f} s")


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", "-c", default=None, help="category name or letter (e.g. parallel, B)")
    ap.add_argument("--ids", nargs="*", default=None, help="explicit scenario ids")
    ap.add_argument("--random", type=int, default=0, help="number of random scenarios")
    ap.add_argument("--all", action="store_true", help="with --random: also run the fixed scenarios")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=max(1, min(14, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--time-limit", type=float, default=None, help="override max_planning_time [s]")
    ap.add_argument("--set", nargs="*", default=[], help="PlannerConfig overrides key=value (python literal)")
    ap.add_argument("--planner", default="bidirectional", choices=["bidirectional", "forward"])
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--regression", action="store_true", help="run the regression cases against the baseline")
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args(argv)

    from report import benchmark_report, spec_report, test_report
    from scenarios import get_scenarios
    overrides = {k: ast.literal_eval(v) for k, v in (kv.split("=", 1) for kv in a.set)}
    if a.time_limit is not None:
        overrides["max_planning_time"] = a.time_limit
    if a.regression:
        scs = get_scenarios(ids=REGRESSION_IDS)
    elif a.random and not (a.all or a.category or a.ids):
        scs = get_scenarios(category="random", random_n=a.random, seed=a.seed)
    else:
        scs = get_scenarios(category=a.category, ids=a.ids, random_n=a.random, seed=a.seed)
    if not scs:
        print("no scenarios selected")
        return 2
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i + 1, sc.id, a.seed, overrides, a.out, not a.no_plots, a.planner) for i, sc in enumerate(scs)]
    print(f"running {len(jobs)} scenarios with {a.workers} worker(s) ...")
    _warmup(scs, overrides, a.workers, a.seed)
    t0 = time.perf_counter()
    rows: List[Dict] = []
    if a.workers > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=20) as pool:
            for r in pool.imap(run_case, jobs):
                rows.append(r)
                _progress(r, len(rows), len(jobs))
    else:
        for j in jobs:
            rows.append(run_case(j))
            _progress(rows[-1], len(rows), len(jobs))

    lines = test_report(rows) + spec_report(rows) + benchmark_report(rows) + sweep_tables(rows)
    if a.regression:
        lines += regression_check(rows, a.update_baseline)
    lines.append(f"\nwall time {time.perf_counter() - t0:.1f} s ({a.workers} workers)")
    text = "\n".join(lines)
    print(text)
    write_csv(rows, os.path.join(a.out, "benchmark.csv"))
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    if a.regression and not a.update_baseline:
        return 0 if "REGRESSION PASSED" in text else 1
    return 0


def _progress(r, i, n):
    print(f"  [{i:>3}/{n}] {'ok  ' if r.get('passed') else 'FAIL'} {r['id']:<14} {r.get('status', ''):<18} "
          f"{_fmt(r.get('planning_time_ms'), '{:.0f}'):>7} ms  {r.get('fail_reason', '')}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
