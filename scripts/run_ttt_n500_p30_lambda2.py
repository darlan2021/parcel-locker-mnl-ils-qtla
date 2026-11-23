#!/usr/bin/env python3
from __future__ import annotations

"""
Time-to-Target (TTT) experiment for the FINAL frozen methods:
    - ILvAS1P
    - QTLA-TR+LP+NG

Design used here:
    N = 500, p = 30, lambda = 2.0
    instance-generation seeds = 21, 22, 23
    30 official algorithmic seeds for each method/instance

Target for each instance:
    C_target = (1 - TARGET_REL_GAP) * C_BK
with TARGET_REL_GAP = 2e-4 (0.02%) and C_BK taken from the completed
stress-test checkpoint (best C observed over both final methods and all
30 algorithmic seeds for that instance).

IMPORTANT
---------
This script does NOT modify mnl_ils.py or qtla_redesign.py.
It imports the already-frozen final configuration from
run_stress_n500_ils_qt_la.py and instruments the search only inside this
runner to record the first time a target-quality solution is reached.

Project layout:
    src/mnl_ils.py
    src/qtla_redesign.py
    scripts/run_stress_n500_ils_qt_la.py
    scripts/run_ttt_n500_p30_lambda2.py
    instances/instances_seed_21_22_23_N_500/
    results/stress_n500_ils_qt_la/checkpoint_stress_n500_30seeds.csv
    results/ttt_n500_p30_lambda2/

Run from the project root:
    python scripts/run_ttt_n500_p30_lambda2.py

The run is resume-safe. Re-running the same command continues from the
checkpoint instead of repeating completed executions.
"""

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Optional
import csv
import json
import math
import traceback
import sys

import numpy as np
import matplotlib.pyplot as plt

# Reusable algorithms live in src/; this runner lives in scripts/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import qtla_redesign as qtla
from mnl_ils import ILSConfig
from run_stress_n500_ils_qt_la import (
    ALGORITHM_SEEDS,
    FROZEN_ILVAS1P,
    FROZEN_QTLA_PARAMETERS,
    PR_TRIGGER_NO_IMPROVE,
    QTLA_FINAL_CONFIG,
    PRTriggeredILS,
    load_common_instance,
    make_ils_config,
    make_qtla_config,
    make_qtla_parameters,
)


# =============================================================================
# PATHS / EXPERIMENT
# =============================================================================

INSTANCE_DIR = ROOT / "instances" / "instances_seed_21_22_23_N_500"
STRESS_CHECKPOINT = (
    ROOT / "results" / "stress_n500_ils_qt_la"
    / "checkpoint_stress_n500_30seeds.csv"
)

RESULT_DIR = ROOT / "results" / "ttt_n500_p30_lambda2"
CHECKPOINT_CSV = RESULT_DIR / "checkpoint_ttt.csv"
TARGETS_CSV = RESULT_DIR / "targets_ttt.csv"
DESIGN_JSON = RESULT_DIR / "design_ttt.json"
REPORT_TXT = RESULT_DIR / "report_ttt.txt"
FIG_PNG = RESULT_DIR / "TTT_N500_p30_lambda20.png"
FIG_PDF = RESULT_DIR / "TTT_N500_p30_lambda20.pdf"

SELECT_N = 500
SELECT_P = 30
SELECT_ALPHA = 2.0
SELECT_INSTANCE_SEEDS = [21, 22, 23]
METHODS = ["ILvAS1P", "QTLA_TR_LP_NG"]

# 0.02% below the best-known observed objective for each selected instance.
TARGET_REL_GAP = 2.0e-4

# Partial figure refresh cadence; useful while the overnight run is progressing.
PLOT_EVERY_N_RUNS = 10


# =============================================================================
# TARGET-HIT EXCEPTION
# =============================================================================

class TargetReached(RuntimeError):
    def __init__(self, elapsed: float, objective: float):
        super().__init__(f"target reached at {elapsed:.9f}s with C={objective:.12f}")
        self.elapsed = float(elapsed)
        self.objective = float(objective)


# =============================================================================
# INSTANCE SELECTION
# =============================================================================

def load_selected_instances():
    files = sorted(INSTANCE_DIR.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ instances found in {INSTANCE_DIR}")

    selected = []
    for path in files:
        dt = load_common_instance(path)
        if (
            dt.N == SELECT_N
            and dt.p == SELECT_P
            and math.isclose(float(dt.alpha), SELECT_ALPHA, abs_tol=1e-12)
            and dt.instance_seed in SELECT_INSTANCE_SEEDS
        ):
            selected.append(dt)

    selected.sort(key=lambda d: d.instance_seed)

    found = [d.instance_seed for d in selected]
    if found != SELECT_INSTANCE_SEEDS:
        raise RuntimeError(
            "Expected exactly one N=500,p=30,lambda=2.0 instance for each "
            f"seed {SELECT_INSTANCE_SEEDS}; found seeds {found}."
        )
    return selected


# =============================================================================
# BEST-KNOWN VALUES AND TARGETS
# =============================================================================

def load_targets(instances) -> dict[int, dict[str, float]]:
    if not STRESS_CHECKPOINT.exists():
        raise FileNotFoundError(
            "Stress-test checkpoint not found:\n"
            f"  {STRESS_CHECKPOINT}\n"
            "The completed results of run_stress_n500_ils_qt_la.py are required, because C_BK is "
            "defined from the completed stress experiment."
        )

    best_by_seed = {seed: -math.inf for seed in SELECT_INSTANCE_SEEDS}

    with STRESS_CHECKPOINT.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for row in reader:
            if row.get("error") not in (None, ""):
                continue
            try:
                N = int(row["N"])
                p = int(row["p"])
                alpha = float(row["alpha"])
                iseed = int(row["instance_seed"])
                C = float(row["C"])
            except (KeyError, TypeError, ValueError):
                continue

            if (
                N == SELECT_N
                and p == SELECT_P
                and math.isclose(alpha, SELECT_ALPHA, abs_tol=1e-12)
                and iseed in best_by_seed
                and math.isfinite(C)
            ):
                best_by_seed[iseed] = max(best_by_seed[iseed], C)

    missing = [seed for seed, val in best_by_seed.items() if not math.isfinite(val)]
    if missing:
        raise RuntimeError(
            f"Could not recover C_BK from stress checkpoint for seeds {missing}."
        )

    targets: dict[int, dict[str, float]] = {}
    for dt in instances:
        bk = float(best_by_seed[dt.instance_seed])
        target = (1.0 - TARGET_REL_GAP) * bk
        targets[dt.instance_seed] = {
            "best_known": bk,
            "target": target,
        }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with TARGETS_CSV.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["instance_seed", "best_known_C", "target_C", "relative_gap"])
        for seed in SELECT_INSTANCE_SEEDS:
            writer.writerow([
                seed,
                f"{targets[seed]['best_known']:.15g}",
                f"{targets[seed]['target']:.15g}",
                f"{TARGET_REL_GAP:.15g}",
            ])

    return targets


# =============================================================================
# ILS TTT INSTRUMENTATION
# =============================================================================

class TTTILvAS1P(PRTriggeredILS):
    """Final calibrated ILvAS1P with first-passage target instrumentation."""

    def __init__(
        self,
        data,
        config: ILSConfig,
        pr_trigger_no_improve: int,
        target: float,
    ):
        super().__init__(data, config, pr_trigger_no_improve)
        self.ttt_target = float(target)
        self.ttt_start: Optional[float] = None

    def solve(self):
        # Set immediately before the exact frozen solve starts.
        self.ttt_start = perf_counter()
        return super().solve()

    def _register_best(self, state) -> bool:
        improved = super()._register_best(state)
        if (
            self.ttt_start is not None
            and self.best_feasible is not None
            and self.best_feasible.raw >= self.ttt_target - 1e-12
        ):
            raise TargetReached(
                perf_counter() - self.ttt_start,
                float(self.best_feasible.raw),
            )
        return improved


def run_ils_ttt(dt, algorithm_seed: int, target: float) -> dict[str, Any]:
    cfg = make_ils_config(int(algorithm_seed))
    solver = TTTILvAS1P(
        dt,
        cfg,
        PR_TRIGGER_NO_IMPROVE,
        target=target,
    )

    wall0 = perf_counter()
    try:
        result = solver.solve()
    except TargetReached as hit:
        wall = perf_counter() - wall0
        return {
            "success": 1,
            "ttt_s": hit.elapsed,
            "observed_C": hit.objective,
            "total_run_s": wall,
            "status": "TARGET_REACHED",
        }

    wall = perf_counter() - wall0
    return {
        "success": int(float(result.objective) >= target - 1e-12),
        "ttt_s": float(result.elapsed_seconds) if float(result.objective) >= target - 1e-12 else math.nan,
        "observed_C": float(result.objective),
        "total_run_s": wall,
        "status": "TARGET_REACHED_AT_END" if float(result.objective) >= target - 1e-12 else "TARGET_NOT_REACHED",
    }


# =============================================================================
# QTLA TTT INSTRUMENTATION
# =============================================================================

def make_ttt_local_polish(target: float, start_time: float):
    """
    Reproduce QTLA's final local-polishing routine, adding only a target check
    at the initial feasible surrogate and after every accepted polishing move.

    No algorithmic parameter or move rule is changed.
    """

    def check(z: np.ndarray, instance: dict) -> None:
        value = float(qtla.objective_value(z, instance))
        if value >= target - 1e-12:
            raise TargetReached(perf_counter() - start_time, value)

    def ttt_polish(instance: dict, initial_z: np.ndarray, max_steps: int) -> np.ndarray:
        z = np.asarray(initial_z, dtype=float).copy()
        if not qtla.is_feasible(z, instance):
            raise ValueError("local_polish requires a feasible initial solution.")

        # The surrogate solution itself is already an available feasible solution.
        check(z, instance)

        if max_steps <= 0:
            return z

        a = instance["a"]
        b = instance["b"]
        nk = instance["nk"]
        n = instance["n"]

        for _step in range(max_steps):
            current_num = a @ z
            current_den = b @ z
            current_value = float(np.sum(current_num / current_den))
            best_value = current_value
            best_move = None

            for index in range(n):
                if not qtla._single_flip_is_feasible(index, z, instance):
                    continue
                sign = -1.0 if z[index] > 0.5 else 1.0
                new_den = current_den + sign * b[:, index]
                if np.any(new_den <= 0.0):
                    continue
                new_num = current_num + sign * a[:, index]
                value = float(np.sum(new_num / new_den))
                if value > best_value + qtla.IMPROVEMENT_TOLERANCE:
                    best_value = value
                    best_move = (index,)

            for start, end in ((0, nk), (nk, n)):
                opened = np.flatnonzero(z[start:end] > 0.5) + start
                closed = np.flatnonzero(z[start:end] < 0.5) + start
                if opened.size == 0 or closed.size == 0:
                    continue

                for out_index in opened:
                    num_without = current_num - a[:, out_index]
                    den_without = current_den - b[:, out_index]
                    cand_num = num_without[:, None] + a[:, closed]
                    cand_den = den_without[:, None] + b[:, closed]
                    valid = np.all(cand_den > 0.0, axis=0)
                    if not np.any(valid):
                        continue
                    values = np.full(closed.size, -np.inf, dtype=float)
                    values[valid] = np.sum(
                        cand_num[:, valid] / cand_den[:, valid], axis=0
                    )
                    pos = int(np.argmax(values))
                    value = float(values[pos])
                    if value > best_value + qtla.IMPROVEMENT_TOLERANCE:
                        best_value = value
                        best_move = (int(out_index), int(closed[pos]))

            if best_move is None:
                break

            z = qtla.apply_move(z, best_move)
            check(z, instance)

        return z

    return ttt_polish


def run_qtla_ttt(dt, algorithm_seed: int, target: float) -> dict[str, Any]:
    instance = qtla.load_instance(dt.path)
    config = make_qtla_config()
    parameters = make_qtla_parameters()

    original_polish = qtla.best_improvement_local_search
    start = perf_counter()
    qtla.best_improvement_local_search = make_ttt_local_polish(target, start)

    try:
        try:
            result = qtla.solve_qtla(
                instance=instance,
                config=config,
                parameters=parameters,
                seed=int(algorithm_seed),
                verbose=False,
            )
        except TargetReached as hit:
            wall = perf_counter() - start
            return {
                "success": 1,
                "ttt_s": hit.elapsed,
                "observed_C": hit.objective,
                "total_run_s": wall,
                "status": "TARGET_REACHED",
            }

        wall = perf_counter() - start
        best_observed = float(result.best_seen_objective)
        success = best_observed >= target - 1e-12
        return {
            "success": int(success),
            "ttt_s": float(result.runtime) if success else math.nan,
            "observed_C": best_observed,
            "total_run_s": wall,
            "status": "TARGET_REACHED_AT_END" if success else "TARGET_NOT_REACHED",
        }
    finally:
        qtla.best_improvement_local_search = original_polish


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

COLUMNS = [
    "Method",
    "Instance",
    "N",
    "p",
    "alpha",
    "instance_seed",
    "algorithm_seed",
    "best_known_C",
    "target_C",
    "target_relative_gap",
    "success",
    "ttt_s",
    "observed_C",
    "total_run_s",
    "status",
    "error",
]


def key_of(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["Method"]),
        int(row["instance_seed"]),
        int(row["algorithm_seed"]),
    )


def load_checkpoint() -> dict[tuple[str, int, int], dict[str, Any]]:
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    if not CHECKPOINT_CSV.exists():
        return out

    with CHECKPOINT_CSV.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for raw in reader:
            if not raw.get("Method"):
                continue
            row = dict(raw)
            row["instance_seed"] = int(row["instance_seed"])
            row["algorithm_seed"] = int(row["algorithm_seed"])
            row["success"] = int(row["success"])
            for k in (
                "alpha",
                "best_known_C",
                "target_C",
                "target_relative_gap",
                "ttt_s",
                "observed_C",
                "total_run_s",
            ):
                if row.get(k) not in (None, "", "nan", "NaN"):
                    row[k] = float(row[k])
                else:
                    row[k] = math.nan
            out[key_of(row)] = row
    return out


def save_checkpoint(results: dict[tuple[str, int, int], dict[str, Any]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    method_order = {m: i for i, m in enumerate(METHODS)}
    seed_order = {s: i for i, s in enumerate(ALGORITHM_SEEDS)}

    ordered = sorted(
        results.values(),
        key=lambda r: (
            int(r["instance_seed"]),
            seed_order.get(int(r["algorithm_seed"]), 10**9),
            method_order.get(str(r["Method"]), 10**9),
        ),
    )

    temp = CHECKPOINT_CSV.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        for row in ordered:
            writer.writerow({c: row.get(c, "") for c in COLUMNS})
    temp.replace(CHECKPOINT_CSV)


# =============================================================================
# TTT FIGURE
# =============================================================================

def empirical_ttt(times: list[float], total_runs: int):
    arr = np.sort(np.asarray(times, dtype=float))
    if arr.size == 0:
        return arr, arr
    # Median-rank plotting positions; failures remain as probability mass < 1.
    probs = (np.arange(1, arr.size + 1) - 0.5) / float(total_runs)
    return arr, probs


def make_figure(results, targets, final: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.0), sharey=True)

    styles = {
        "ILvAS1P": dict(color="black", linestyle="-", marker="o"),
        "QTLA_TR_LP_NG": dict(color="0.55", linestyle="--", marker="s"),
    }
    labels = {
        "ILvAS1P": "ILvAS1P",
        "QTLA_TR_LP_NG": "QTLA-TR+LP+NG",
    }

    for ax, iseed in zip(axes, SELECT_INSTANCE_SEEDS):
        for method in METHODS:
            rows = [
                r for r in results.values()
                if r.get("error") in (None, "")
                and int(r["instance_seed"]) == iseed
                and str(r["Method"]) == method
                and int(r.get("success", 0)) == 1
                and math.isfinite(float(r.get("ttt_s", math.nan)))
            ]
            times = [float(r["ttt_s"]) for r in rows]
            x, y = empirical_ttt(times, len(ALGORITHM_SEEDS))
            if x.size:
                st = styles[method]
                ax.step(
                    x,
                    y,
                    where="post",
                    color=st["color"],
                    linestyle=st["linestyle"],
                    linewidth=1.6,
                    marker=st["marker"],
                    markersize=3.5,
                    markerfacecolor="white",
                    label=labels[method],
                )

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.02)
        ax.grid(True, which="both", alpha=0.22)
        target = targets[iseed]["target"]
        ax.set_title(f"instance seed {iseed}\n$C_{{target}}$={target:.3f}")
        ax.set_xlabel("Time to target (s, log scale)")

    axes[0].set_ylabel(r"$P(T \leq t)$")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_legend,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=2,
            frameon=True,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(FIG_PDF, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# REPORT / DESIGN
# =============================================================================

def write_design(instances, targets) -> None:
    design = {
        "experiment": "TTT final ILvAS1P vs QTLA-TR+LP+NG",
        "selected_design": {
            "N": SELECT_N,
            "p": SELECT_P,
            "alpha_lambda": SELECT_ALPHA,
            "instance_seeds": SELECT_INSTANCE_SEEDS,
        },
        "algorithm_seeds": ALGORITHM_SEEDS,
        "target_relative_gap": TARGET_REL_GAP,
        "targets": targets,
        "frozen_ilvas1p": FROZEN_ILVAS1P,
        "pr_trigger_no_improve": PR_TRIGGER_NO_IMPROVE,
        "qtla_final_config": QTLA_FINAL_CONFIG,
        "frozen_qtla_parameters": FROZEN_QTLA_PARAMETERS,
        "instances": [dt.name for dt in instances],
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    DESIGN_JSON.write_text(
        json.dumps(design, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_report(results, targets) -> None:
    lines = []
    lines.append("=" * 90)
    lines.append("TIME-TO-TARGET: ILvAS1P vs QTLA-TR+LP+NG")
    lines.append("=" * 90)
    lines.append(f"Design: N={SELECT_N}, p={SELECT_P}, lambda={SELECT_ALPHA}")
    lines.append(f"Instance seeds: {SELECT_INSTANCE_SEEDS}")
    lines.append(f"Algorithm seeds: {len(ALGORITHM_SEEDS)}")
    lines.append(f"Target relative gap: {100*TARGET_REL_GAP:.5f}%")
    lines.append("")

    for iseed in SELECT_INSTANCE_SEEDS:
        lines.append(
            f"seed {iseed}: C_BK={targets[iseed]['best_known']:.10f}, "
            f"C_target={targets[iseed]['target']:.10f}"
        )
        for method in METHODS:
            rows = [
                r for r in results.values()
                if int(r["instance_seed"]) == iseed
                and str(r["Method"]) == method
                and r.get("error") in (None, "")
            ]
            succ = [r for r in rows if int(r.get("success", 0)) == 1]
            times = np.asarray([
                float(r["ttt_s"]) for r in succ
                if math.isfinite(float(r.get("ttt_s", math.nan)))
            ])
            if times.size:
                lines.append(
                    f"  {method:16s}: completed={len(rows):2d}/30, "
                    f"hits={len(succ):2d}/30, "
                    f"median={np.median(times):.6f}s, "
                    f"mean={np.mean(times):.6f}s, max={np.max(times):.6f}s"
                )
            else:
                lines.append(
                    f"  {method:16s}: completed={len(rows):2d}/30, hits=0/30"
                )
        lines.append("")

    REPORT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =============================================================================
# SCHEDULE / MAIN
# =============================================================================

def build_schedule(instances):
    jobs = []
    for dt in instances:
        for pos, aseed in enumerate(ALGORITHM_SEEDS):
            # paired and order-balanced, as in the previous experiments
            order = METHODS if pos % 2 == 0 else list(reversed(METHODS))
            for method in order:
                jobs.append((method, dt, int(aseed)))
    return jobs


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    instances = load_selected_instances()
    targets = load_targets(instances)
    write_design(instances, targets)

    results = load_checkpoint()
    schedule = build_schedule(instances)
    total = len(schedule)

    print("=" * 100)
    print("TTT - FINAL ILvAS1P vs QTLA-TR+LP+NG")
    print("=" * 100)
    print(f"Design          : N={SELECT_N}, p={SELECT_P}, lambda={SELECT_ALPHA}")
    print(f"Instance seeds  : {SELECT_INSTANCE_SEEDS}")
    print(f"Algorithm seeds : {len(ALGORITHM_SEEDS)}")
    print(f"Runs expected   : {total}")
    print(f"Target gap      : {100*TARGET_REL_GAP:.5f}% below C_BK")
    for seed in SELECT_INSTANCE_SEEDS:
        print(
            f"  seed {seed}: C_BK={targets[seed]['best_known']:.8f} | "
            f"target={targets[seed]['target']:.8f}"
        )
    print(f"Results         : {RESULT_DIR}")
    print()

    completed_since_plot = 0

    for position, (method, dt, aseed) in enumerate(schedule, start=1):
        key = (method, dt.instance_seed, aseed)
        old = results.get(key)
        if old is not None and old.get("error") in (None, ""):
            continue

        target_info = targets[dt.instance_seed]
        target = target_info["target"]

        print(
            f"[{position:3d}/{total}] {method:16s} | iseed={dt.instance_seed:2d} | "
            f"aseed={aseed:6d} | target={target:.6f} | ",
            end="",
            flush=True,
        )

        try:
            if method == "ILvAS1P":
                out = run_ils_ttt(dt, aseed, target)
            elif method == "QTLA_TR_LP_NG":
                out = run_qtla_ttt(dt, aseed, target)
            else:
                raise KeyError(method)

            row = {
                "Method": method,
                "Instance": dt.name,
                "N": dt.N,
                "p": dt.p,
                "alpha": dt.alpha,
                "instance_seed": dt.instance_seed,
                "algorithm_seed": aseed,
                "best_known_C": target_info["best_known"],
                "target_C": target,
                "target_relative_gap": TARGET_REL_GAP,
                "success": int(out["success"]),
                "ttt_s": out["ttt_s"],
                "observed_C": out["observed_C"],
                "total_run_s": out["total_run_s"],
                "status": out["status"],
                "error": "",
            }
            results[key] = row

            if row["success"]:
                print(
                    f"HIT | t={float(row['ttt_s']):.6f}s | "
                    f"C={float(row['observed_C']):.8f}"
                )
            else:
                print(
                    f"MISS | Cmax={float(row['observed_C']):.8f} | "
                    f"run={float(row['total_run_s']):.4f}s"
                )

        except Exception as exc:
            row = {
                "Method": method,
                "Instance": dt.name,
                "N": dt.N,
                "p": dt.p,
                "alpha": dt.alpha,
                "instance_seed": dt.instance_seed,
                "algorithm_seed": aseed,
                "best_known_C": target_info["best_known"],
                "target_C": target,
                "target_relative_gap": TARGET_REL_GAP,
                "success": 0,
                "ttt_s": math.nan,
                "observed_C": math.nan,
                "total_run_s": math.nan,
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
            }
            results[key] = row
            print(f"ERROR | {type(exc).__name__}: {exc}")
            traceback.print_exc()

        save_checkpoint(results)
        write_report(results, targets)
        completed_since_plot += 1

        if completed_since_plot >= PLOT_EVERY_N_RUNS:
            make_figure(results, targets)
            completed_since_plot = 0

    save_checkpoint(results)
    write_report(results, targets)
    make_figure(results, targets, final=True)

    valid = [r for r in results.values() if r.get("error") in (None, "")]
    errors = [r for r in results.values() if r.get("error") not in (None, "")]

    print()
    print("=" * 100)
    print("TTT FINISHED")
    print("=" * 100)
    print(f"Valid runs : {len(valid)}/{total}")
    print(f"Errors     : {len(errors)}")
    print(f"Checkpoint : {CHECKPOINT_CSV}")
    print(f"Targets    : {TARGETS_CSV}")
    print(f"Report     : {REPORT_TXT}")
    print(f"Figure PDF : {FIG_PDF}")
    print(f"Figure PNG : {FIG_PNG}")

    if len(valid) != total or errors:
        raise RuntimeError(
            "TTT experiment ended with missing/failed runs. Re-run the same "
            "script to resume from the checkpoint."
        )


if __name__ == "__main__":
    main()
