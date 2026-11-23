#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SRC_DIR = PROJECT_ROOT / "src"
INSTANCE_DIR = PROJECT_ROOT / "instances" / "instances_seed_11_12_13_N60_100_200_400"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mnl_ils import ILSConfig, MNLParcelLockerILS, SearchProfile, SearchResult


# =====================================================================
# FROZEN ILvAS1 CONFIGURATION
# =====================================================================
# IMPORTANT: these values are passed explicitly. The calibration does not
# rely on the defaults currently written in mnl_ils.py.
FROZEN = {
    "grasp_starts": 8,
    "grasp_alpha": 0.0823,
    "max_vnd_passes": 250,
    "tau_tabu": 13,
    "max_cardinality_violation": 1,
    "s2_sample_size": 350,
    "first_improvement": False,  # BI

    "enable_A": True,
    "enable_S1": True,
    "enable_S2": False,
    "enable_path_relink": True,

    "penalty_weight": None,
    "penalty_multiplier": 2.2255,
    "penalty_quadratic": 0.6576,

    "max_iterations": 10,
    "time_limit": None,
    "initial_perturbation_strength": 2,
    "min_perturbation_strength": 1,
    "max_perturbation_strength": 10,
    "stagnation_window": 22,
    "acceptance_tolerance": 0.0074,

    "perturb_prob_A": 0.15,
    "perturb_prob_S1": 0.55,
    "perturb_prob_S2": 0.30,

    "elite_size": 6,
    "elite_min_hamming": 6,

    # Old periodic trigger is disabled. The subclass below triggers PR by
    # consecutive iterations without global improvement.
    "path_relink_interval": 0,

    # S2 must not enter indirectly through Path Relinking.
    "path_relink_s2_samples": 0,

    "verbose": False,
}


@dataclass
class InstanceData:
    name: str
    path: Path
    n_i: int
    n_j: int
    n_k: int
    p: int
    alpha: float
    seed: int
    a: np.ndarray
    b: np.ndarray


def _scalar(z: Any, key: str, cast):
    value = z[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_instance(path: Path) -> InstanceData:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as z:
        n_i = _scalar(z, "n_i", int)
        n_j = _scalar(z, "n_j", int)
        n_k = _scalar(z, "n_k", int)
        p = _scalar(z, "p", int)
        alpha = _scalar(z, "alpha", float)
        seed = _scalar(z, "seed", int)
        a = np.asarray(z["a"], dtype=float)
        b = np.asarray(z["b"], dtype=float)

    n = n_k + n_j
    if a.shape != (n_i, n):
        raise ValueError(f"{path.name}: a.shape={a.shape}, expected {(n_i, n)}")
    if b.shape != (n_i, n):
        raise ValueError(f"{path.name}: b.shape={b.shape}, expected {(n_i, n)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{path.name}: a contains non-finite values")
    if np.any(b <= 0.0) or not np.all(np.isfinite(b)):
        raise ValueError(f"{path.name}: b must be finite and strictly positive")

    return InstanceData(
        name=path.stem,
        path=path,
        n_i=n_i,
        n_j=n_j,
        n_k=n_k,
        p=p,
        alpha=alpha,
        seed=seed,
        a=a,
        b=b,
    )


class PRTriggeredILS(MNLParcelLockerILS):
    """
    Same ILS implementation, with only one control-flow change:
    Path Relinking is activated by stagnation instead of a periodic interval.

    For trigger q, PR is attempted at q, 2q, 3q, ... consecutive iterations
    without a global improvement. Any global improvement resets stagnation.
    """

    def __init__(self, data, config: ILSConfig, pr_trigger_no_improve: int):
        super().__init__(data, config)
        self.pr_trigger_no_improve = int(pr_trigger_no_improve)
        if self.pr_trigger_no_improve <= 0:
            raise ValueError("pr_trigger_no_improve must be positive")

    def solve(self) -> SearchResult:
        start_time = time.perf_counter()
        self.profile = SearchProfile()
        self.best_feasible = None
        self.elite = []
        self.history = []
        self.move_clock = 0
        self.tabu_drop_until.fill(-1)
        self.tabu_add_until.fill(-1)

        # GRASP multi-start + local improvement: unchanged.
        current = None
        for _ in range(self.cfg.grasp_starts):
            st = self.grasp_construct()
            if self.rho is None:
                self._scale_penalty_if_needed(st)
                st.penalized = self._penalized(st.raw, st.y)

            self._register_best(st)
            st = self.local_search(st)
            self._add_elite(st)
            self._register_best(st)

            if current is None or st.raw > current.raw:
                current = st.copy()

            if self._time_exceeded(start_time):
                break

        if current is None or self.best_feasible is None:
            raise RuntimeError("No feasible solution was generated.")

        # Iterated Local Search.
        strength = int(
            np.clip(
                self.cfg.initial_perturbation_strength,
                self.cfg.min_perturbation_strength,
                self.cfg.max_perturbation_strength,
            )
        )
        stagnation = 0
        iterations_done = 0

        for it in range(1, self.cfg.max_iterations + 1):
            if self._time_exceeded(start_time):
                break

            iterations_done = it
            before_best = self.best_feasible.raw

            shaken = self.perturb(current, strength, stagnation)
            candidate = self.local_search(shaken)
            self._add_elite(candidate)
            self._register_best(candidate)

            # Record-to-record acceptance: unchanged.
            threshold = self.best_feasible.raw - self.cfg.acceptance_tolerance * (
                1.0 + abs(self.best_feasible.raw)
            )
            if candidate.raw >= current.raw - 1e-12 or candidate.raw >= threshold:
                current = candidate.copy()
            elif self.elite:
                current = self.elite[int(self.rng.integers(len(self.elite)))].copy()

            # Include the current iteration in the stagnation count before
            # deciding whether PR should be triggered.
            pre_pr_improved = self.best_feasible.raw > before_best + 1e-12
            stagnation_for_pr = 0 if pre_pr_improved else stagnation + 1

            # Stagnation-triggered Path Relinking.
            if (
                self.cfg.enable_path_relink
                and stagnation_for_pr > 0
                and stagnation_for_pr % self.pr_trigger_no_improve == 0
                and len(self.elite) >= 2
            ):
                distances = np.asarray(
                    [self.hamming(current.y, e.y) for e in self.elite],
                    dtype=int,
                )

                # Original guide-selection rule: choose among the most
                # distant half of the elite pool.
                order = np.argsort(-distances)
                half = order[: max(1, len(order) // 2)]
                guide_idx = int(self.rng.choice(half))
                guide = self.elite[guide_idx]

                if not np.array_equal(current.y, guide.y):
                    pr = self.path_relink(current, guide)
                    self._add_elite(pr)
                    self._register_best(pr)
                    if pr.raw > current.raw + 1e-12:
                        current = pr.copy()

            after_best = self.best_feasible.raw
            if after_best > before_best + 1e-12:
                stagnation = 0
                strength = max(
                    self.cfg.min_perturbation_strength,
                    strength - 1,
                )
            else:
                stagnation = stagnation_for_pr
                if stagnation % self.cfg.stagnation_window == 0:
                    strength = min(
                        self.cfg.max_perturbation_strength,
                        strength + 1,
                    )

            self.history.append(
                {
                    "iteration": it,
                    "best": self.best_feasible.raw,
                    "current": current.raw,
                    "strength": strength,
                    "stagnation": stagnation,
                    "elite_size": len(self.elite),
                    "pr_trigger_no_improve": self.pr_trigger_no_improve,
                }
            )

        best = self.best_feasible
        elapsed = time.perf_counter() - start_time
        self.profile.total_seconds = elapsed

        open_existing = np.flatnonzero(best.y[: self.n_k] == 1).astype(int).tolist()
        closed_existing = np.flatnonzero(best.y[: self.n_k] == 0).astype(int).tolist()
        open_candidates = (
            np.flatnonzero(best.y[self.n_k :] == 1).astype(int).tolist()
        )

        return SearchResult(
            objective=float(best.raw),
            penalized_objective=float(best.penalized),
            feasible=self.feasible(best.y),
            y=best.y.copy(),
            open_existing=open_existing,
            closed_existing=closed_existing,
            open_candidates=open_candidates,
            iterations=iterations_done,
            elapsed_seconds=elapsed,
            perturbation_strength=strength,
            profile=self.profile.copy(),
            history=self.history.copy(),
        )


def append_run_log(row: dict[str, Any]) -> None:
    out = HERE / "run_level_results.csv"
    fields = [
        "configuration_id",
        "instance_id",
        "algorithm_seed",
        "instance",
        "N",
        "p",
        "alpha",
        "instance_seed",
        "pr_trigger_no_improve",
        "pr_bidirectional",
        "objective_C",
        "cpu_seconds",
        "path_relink_calls",
        "path_relink_seconds",
        "PR_move_evaluations",
        "PR_steps",
        "PR_A_steps",
        "PR_S1_steps",
        "PR_S2_steps",
        "feasible",
        "cost_returned",
        "error",
    ]

    new_file = not out.exists()
    with out.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("configuration_id", type=int)
    p.add_argument("instance_id", type=int)
    p.add_argument("algorithm_seed", type=int)
    p.add_argument("instance", type=str)
    p.add_argument("--pr-trigger-no-improve", type=int, required=True)
    p.add_argument("--pr-bidirectional", type=int, choices=[0, 1], required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Support historical instance-list entries while always preferring the
    # official calibration folder for relative paths. Do not alter instance order.
    raw_instance = args.instance.strip()
    filename = Path(raw_instance.replace("\\", "/")).name
    if not filename.lower().endswith(".npz"):
        raise ValueError(f"Expected .npz instance, got: {raw_instance}")
    supplied = Path(raw_instance)
    pth = supplied if supplied.is_file() else INSTANCE_DIR / filename
    pth = pth.resolve()

    row = {
        "configuration_id": args.configuration_id,
        "instance_id": args.instance_id,
        "algorithm_seed": args.algorithm_seed,
        "instance": str(pth),
        "pr_trigger_no_improve": args.pr_trigger_no_improve,
        "pr_bidirectional": args.pr_bidirectional,
        "error": "",
    }

    try:
        dt = load_instance(pth)

        cfg_kwargs = dict(FROZEN)
        cfg_kwargs["seed"] = int(args.algorithm_seed)
        cfg_kwargs["path_relink_bidirectional"] = bool(args.pr_bidirectional)

        cfg = ILSConfig(**cfg_kwargs)
        solver = PRTriggeredILS(
            dt,
            cfg,
            pr_trigger_no_improve=args.pr_trigger_no_improve,
        )

        t0 = time.perf_counter()
        result = solver.solve()
        wall = time.perf_counter() - t0

        if not result.feasible:
            raise RuntimeError("ILS returned an infeasible final solution")

        # Independent final objective audit.
        y = np.asarray(result.y, dtype=float)
        den = dt.b @ y
        if np.any(den <= 0.0):
            raise FloatingPointError("Non-positive denominator in final audit")
        direct_C = float(np.sum((dt.a @ y) / den))

        if abs(direct_C - result.objective) > 1e-8 * (1.0 + abs(direct_C)):
            raise FloatingPointError(
                f"Objective audit failed: solver={result.objective}, direct={direct_C}"
            )

        # irace minimizes; the MNL objective C is maximized.
        cost = -float(result.objective)

        row.update(
            {
                "N": dt.n_i + dt.n_k + dt.n_j,
                "p": dt.p,
                "alpha": dt.alpha,
                "instance_seed": dt.seed,
                "objective_C": result.objective,
                "cpu_seconds": wall,
                "path_relink_calls": result.profile.path_relink_calls,
                "path_relink_seconds": result.profile.path_relink_seconds,
                "PR_move_evaluations": result.profile.PR_move_evaluations,
                "PR_steps": result.profile.PR_steps,
                "PR_A_steps": result.profile.PR_A_steps,
                "PR_S1_steps": result.profile.PR_S1_steps,
                "PR_S2_steps": result.profile.PR_S2_steps,
                "feasible": result.feasible,
                "cost_returned": cost,
            }
        )
        append_run_log(row)

        # stdout must contain only the numerical cost for irace.
        print(f"{cost:.15g}")
        return 0

    except Exception as exc:
        penalty = 1.0e100
        row["cost_returned"] = penalty
        row["feasible"] = False
        row["error"] = f"{type(exc).__name__}: {exc}"
        try:
            append_run_log(row)
        except Exception:
            pass

        print(f"{penalty:.15g}")
        print(
            f"target_runner_pr.py error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
