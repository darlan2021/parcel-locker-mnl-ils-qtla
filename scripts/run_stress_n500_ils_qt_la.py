#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from xml.sax.saxutils import escape as xml_escape
import csv
import json
import math
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mnl_ils import ILSConfig, MNLParcelLockerILS, SearchProfile, SearchResult
from qtla_redesign import (
    QTLAConfig,
    QTLAParameters,
    load_instance as load_qtla_instance,
    objective_value as qtla_objective_value,
    solve_qtla,
)


# =============================================================================
# PATHS
# =============================================================================

INSTANCE_DIR = ROOT / "instances" / "instances_seed_21_22_23_N_500"

RESULT_DIR = ROOT / "results" / "stress_n500_ils_qt_la"
CHECKPOINT_CSV = RESULT_DIR / "checkpoint_stress_n500_30seeds.csv"
REPORT_TXT = RESULT_DIR / "relatorio_stress_n500_30seeds.txt"
DESIGN_JSON = RESULT_DIR / "desenho_stress_n500_30seeds.json"
XLSX_FILE = RESULT_DIR / "resultados_stress_n500_30seeds.xlsx"


# =============================================================================
# OFFICIAL 30 ALGORITHM SEEDS
# Same seeds used in the previous ILS/QTLA experiments.
# =============================================================================

ALGORITHM_SEEDS = [
    183601,
    122241,
    347126,
    420036,
    22114,
    384850,
    435983,
    134804,
    172693,
    210450,
    171434,
    187001,
    133041,
    328721,
    497477,
    91030,
    87198,
    205899,
    400804,
    237596,
    1346,
    411979,
    295898,
    269046,
    154485,
    171750,
    231105,
    317139,
    409973,
    89335,
]


# =============================================================================
# STRESS-TEST DESIGN
# =============================================================================

EXPECTED_INSTANCE_SEEDS = [21, 22, 23]
EXPECTED_DESIGN = {
    500: [20, 25, 30],
}
EXPECTED_ALPHAS = [0.5, 1.0, 2.0]
EXPECTED_INSTANCES_PER_SEED = 9
EXPECTED_INSTANCE_COUNT = 27

METHODS = ["ILvAS1P", "QTLA_TR_LP_NG"]


# =============================================================================
# FROZEN FINAL ILS: ILvAS1P
# Core parameters from the selected irace configuration + calibrated PR.
# =============================================================================

FROZEN_ILVAS1P = {
    "grasp_starts": 8,
    "grasp_alpha": 0.0823,
    "max_vnd_passes": 250,
    "tau_tabu": 13,
    "max_cardinality_violation": 1,
    "s2_sample_size": 350,
    "first_improvement": False,

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

    # Old periodic PR trigger disabled; calibrated stagnation trigger used below.
    "path_relink_interval": 0,
    "path_relink_bidirectional": False,
    "path_relink_s2_samples": 0,
}

PR_TRIGGER_NO_IMPROVE = 5
ILS_VERBOSE = False


# =============================================================================
# FROZEN FINAL QTLA: TR + LP + NG
# Frozen irace elite ID 68 parameters; NOA is OFF.
# =============================================================================

QTLA_FINAL_CONFIG = {
    "trust_region": True,
    "neighbor_oa": False,
    "local_polish": True,
    "no_good": True,
}

FROZEN_QTLA_PARAMETERS = {
    "max_iterations": 30,
    "subproblem_time_limit": 2.0,
    "damping_gamma": 0.9171,
    "trust_radius_factor": 0.4186,
    "neighbor_cuts": 1,
    "polish_max_steps": 20,
    "mip_gap": 1.0e-6,
    "oa_epsilon": 1.0e-4,
    "overall_time_limit": 3600.0,
}


# =============================================================================
# INSTANCE DATA
# =============================================================================

@dataclass(frozen=True)
class InstanceData:
    name: str
    path: Path
    n_i: int
    n_j: int
    n_k: int
    p: int
    alpha: float
    instance_seed: int
    a: np.ndarray
    b: np.ndarray

    @property
    def N(self) -> int:
        return self.n_i + self.n_k + self.n_j


def _scalar(npz: Any, key: str, cast):
    value = npz[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_common_instance(path: Path) -> InstanceData:
    with np.load(path, allow_pickle=False) as z:
        n_i = _scalar(z, "n_i", int)
        n_j = _scalar(z, "n_j", int)
        n_k = _scalar(z, "n_k", int)
        p = _scalar(z, "p", int)

        if "alpha" in z.files:
            alpha = _scalar(z, "alpha", float)
        elif "lambda_" in z.files:
            alpha = _scalar(z, "lambda_", float)
        elif "lambda" in z.files:
            alpha = _scalar(z, "lambda", float)
        else:
            raise KeyError(f"{path.name}: missing alpha/lambda field")

        if "seed" in z.files:
            instance_seed = _scalar(z, "seed", int)
        elif "instance_seed" in z.files:
            instance_seed = _scalar(z, "instance_seed", int)
        else:
            raise KeyError(f"{path.name}: missing seed field")

        a = np.asarray(z["a"], dtype=float)
        b = np.asarray(z["b"], dtype=float)

    n = n_k + n_j
    if a.shape != (n_i, n):
        raise ValueError(
            f"{path.name}: a.shape={a.shape}, expected {(n_i, n)}."
        )
    if b.shape != (n_i, n):
        raise ValueError(
            f"{path.name}: b.shape={b.shape}, expected {(n_i, n)}."
        )
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{path.name}: matrix a contains non-finite values.")
    if np.any(b <= 0.0) or not np.all(np.isfinite(b)):
        raise ValueError(
            f"{path.name}: all b coefficients must be finite and strictly positive."
        )

    return InstanceData(
        name=path.stem,
        path=path,
        n_i=n_i,
        n_j=n_j,
        n_k=n_k,
        p=p,
        alpha=alpha,
        instance_seed=instance_seed,
        a=a,
        b=b,
    )


def validate_algorithm_seeds() -> None:
    if len(ALGORITHM_SEEDS) != 30:
        raise RuntimeError(
            f"Expected exactly 30 algorithm seeds, found {len(ALGORITHM_SEEDS)}."
        )
    if len(set(ALGORITHM_SEEDS)) != 30:
        raise RuntimeError("Duplicate algorithm seeds detected.")


def load_stress_instances() -> list[InstanceData]:
    if not INSTANCE_DIR.exists():
        raise FileNotFoundError(
            f"Stress-test instance folder not found: {INSTANCE_DIR.resolve()}"
        )

    files = sorted(INSTANCE_DIR.glob("*.npz"))
    if len(files) != EXPECTED_INSTANCE_COUNT:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_INSTANCE_COUNT} .npz files in "
            f"{INSTANCE_DIR}, found {len(files)}."
        )

    instances = [load_common_instance(path) for path in files]

    found_seeds = sorted({dt.instance_seed for dt in instances})
    if found_seeds != EXPECTED_INSTANCE_SEEDS:
        raise RuntimeError(
            f"Stress-test instance seeds differ from expected. "
            f"Expected {EXPECTED_INSTANCE_SEEDS}, found {found_seeds}."
        )

    expected_points = {
        (N, p, round(alpha, 10))
        for N, p_values in EXPECTED_DESIGN.items()
        for p in p_values
        for alpha in EXPECTED_ALPHAS
    }

    for instance_seed in EXPECTED_INSTANCE_SEEDS:
        subset = [dt for dt in instances if dt.instance_seed == instance_seed]
        if len(subset) != EXPECTED_INSTANCES_PER_SEED:
            raise RuntimeError(
                f"Seed {instance_seed}: expected {EXPECTED_INSTANCES_PER_SEED} "
                f"instances, found {len(subset)}."
            )

        observed = {
            (dt.N, dt.p, round(dt.alpha, 10))
            for dt in subset
        }
        missing = sorted(expected_points - observed)
        extra = sorted(observed - expected_points)
        if missing or extra:
            raise RuntimeError(
                f"Seed {instance_seed}: stress-test design mismatch.\n"
                f"Missing: {missing}\nExtra: {extra}"
            )
        if len(observed) != EXPECTED_INSTANCES_PER_SEED:
            raise RuntimeError(
                f"Seed {instance_seed}: duplicate N/p/alpha design points detected."
            )

    instances.sort(
        key=lambda d: (d.instance_seed, d.N, d.p, d.alpha, d.name)
    )
    return instances


# =============================================================================
# STAGNATION-TRIGGERED PATH RELINKING
# Same final ILvAS1P rule selected after PR calibration.
# =============================================================================

class PRTriggeredILS(MNLParcelLockerILS):
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

            threshold = self.best_feasible.raw - self.cfg.acceptance_tolerance * (
                1.0 + abs(self.best_feasible.raw)
            )
            if candidate.raw >= current.raw - 1e-12 or candidate.raw >= threshold:
                current = candidate.copy()
            elif self.elite:
                current = self.elite[int(self.rng.integers(len(self.elite)))].copy()

            pre_pr_improved = self.best_feasible.raw > before_best + 1e-12
            stagnation_for_pr = 0 if pre_pr_improved else stagnation + 1

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


# =============================================================================
# METHOD RUNNERS
# =============================================================================

def make_ils_config(algorithm_seed: int) -> ILSConfig:
    params = dict(FROZEN_ILVAS1P)
    params["seed"] = int(algorithm_seed)
    params["verbose"] = ILS_VERBOSE
    return ILSConfig(**params)


def run_ilvas1p(dt: InstanceData, algorithm_seed: int) -> dict[str, Any]:
    cfg = make_ils_config(algorithm_seed)
    solver = PRTriggeredILS(dt, cfg, PR_TRIGGER_NO_IMPROVE)

    wall0 = perf_counter()
    result = solver.solve()
    wall_time = perf_counter() - wall0

    if not result.feasible:
        raise RuntimeError(
            f"ILvAS1P returned infeasible solution: {dt.name}, seed={algorithm_seed}"
        )

    y = np.asarray(result.y, dtype=float)
    num = dt.a @ y
    den = dt.b @ y
    if np.any(den <= 0.0):
        raise FloatingPointError("Non-positive denominator in ILvAS1P audit.")

    audited_C = float(np.sum(num / den))
    audit_abs_diff = abs(float(result.objective) - audited_C)

    if not math.isclose(
        float(result.objective), audited_C, rel_tol=1e-9, abs_tol=1e-8
    ):
        raise RuntimeError(
            "ILvAS1P objective audit mismatch: "
            f"reported={result.objective:.15g}, audited={audited_C:.15g}, "
            f"diff={audit_abs_diff:.3e}"
        )

    if int(result.profile.PR_S2_steps) != 0:
        raise RuntimeError(
            f"PR-S2 contamination detected: PR_S2_steps={result.profile.PR_S2_steps}"
        )

    return {
        "Method": "ILvAS1P",
        "Instance": dt.name,
        "N": dt.N,
        "ni": dt.n_i,
        "nk": dt.n_k,
        "nj": dt.n_j,
        "p": dt.p,
        "alpha": dt.alpha,
        "instance_seed": dt.instance_seed,
        "algorithm_seed": int(algorithm_seed),
        "C": audited_C,
        "time_s": float(result.elapsed_seconds),
        "wall_time_s": float(wall_time),
        "iterations": int(result.iterations),
        "status": "FEASIBLE",
        "audit_abs_diff": audit_abs_diff,
        "path_relink_calls": int(result.profile.path_relink_calls),
        "path_relink_seconds": float(result.profile.path_relink_seconds),
        "PR_steps": int(result.profile.PR_steps),
        "PR_A_steps": int(result.profile.PR_A_steps),
        "PR_S1_steps": int(result.profile.PR_S1_steps),
        "PR_S2_steps": int(result.profile.PR_S2_steps),
        "feasible": 1,
        "error": "",
    }


def make_qtla_config() -> QTLAConfig:
    return QTLAConfig(**QTLA_FINAL_CONFIG)


def make_qtla_parameters() -> QTLAParameters:
    return QTLAParameters(**FROZEN_QTLA_PARAMETERS)


def run_qtla_final(dt: InstanceData, algorithm_seed: int) -> dict[str, Any]:
    instance = load_qtla_instance(dt.path)
    config = make_qtla_config()
    parameters = make_qtla_parameters()

    wall0 = perf_counter()
    result = solve_qtla(
        instance=instance,
        config=config,
        parameters=parameters,
        seed=int(algorithm_seed),
        verbose=False,
    )
    wall_time = perf_counter() - wall0

    if not result.feasible:
        raise RuntimeError(
            f"QTLA_TR_LP_NG returned infeasible result: "
            f"{dt.name}, seed={algorithm_seed}"
        )

    C = float(result.objective)
    runtime = float(result.runtime)
    if not math.isfinite(C) or not math.isfinite(runtime):
        raise RuntimeError(
            f"Non-finite QTLA result: C={C}, runtime={runtime}"
        )

    z = np.zeros(instance["n"], dtype=float)
    for idx in result.combined_open_0based:
        z[int(idx)] = 1.0

    audited_C = float(qtla_objective_value(z, instance))
    audit_abs_diff = abs(C - audited_C)

    if not math.isclose(C, audited_C, rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError(
            "QTLA objective audit mismatch: "
            f"reported={C:.15g}, audited={audited_C:.15g}, "
            f"diff={audit_abs_diff:.3e}"
        )

    return {
        "Method": "QTLA_TR_LP_NG",
        "Instance": dt.name,
        "N": dt.N,
        "ni": dt.n_i,
        "nk": dt.n_k,
        "nj": dt.n_j,
        "p": dt.p,
        "alpha": dt.alpha,
        "instance_seed": dt.instance_seed,
        "algorithm_seed": int(algorithm_seed),
        "C": C,
        "time_s": runtime,
        "wall_time_s": float(wall_time),
        "iterations": int(result.iterations),
        "status": str(result.status),
        "audit_abs_diff": audit_abs_diff,
        "path_relink_calls": "",
        "path_relink_seconds": "",
        "PR_steps": "",
        "PR_A_steps": "",
        "PR_S1_steps": "",
        "PR_S2_steps": "",
        "feasible": 1,
        "error": "",
    }


def run_method(
    method: str,
    dt: InstanceData,
    algorithm_seed: int,
) -> dict[str, Any]:
    if method == "ILvAS1P":
        return run_ilvas1p(dt, algorithm_seed)
    if method == "QTLA_TR_LP_NG":
        return run_qtla_final(dt, algorithm_seed)
    raise KeyError(f"Unknown method: {method}")


# =============================================================================
# EXPERIMENT DESIGN / RESUME SAFETY
# =============================================================================

def experiment_design_dict(instances: list[InstanceData]) -> dict[str, Any]:
    return {
        "experiment": "stress test N=500: frozen ILvAS1P vs frozen QTLA TR+LP+NG",
        "instance_directory": INSTANCE_DIR.name,
        "instance_count": len(instances),
        "instance_seeds": EXPECTED_INSTANCE_SEEDS,
        "instances": [dt.name for dt in instances],
        "algorithm_seeds": ALGORITHM_SEEDS,
        "methods": METHODS,
        "frozen_ilvas1p": FROZEN_ILVAS1P,
        "pr_trigger_no_improve": PR_TRIGGER_NO_IMPROVE,
        "qtla_final_config": QTLA_FINAL_CONFIG,
        "frozen_qtla_parameters": FROZEN_QTLA_PARAMETERS,
        "expected_runs": len(instances) * len(ALGORITHM_SEEDS) * len(METHODS),
    }


def verify_or_write_design(instances: list[InstanceData]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    design = experiment_design_dict(instances)
    canonical = json.dumps(
        design,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )

    if DESIGN_JSON.exists():
        old = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
        # Only the location label changed; all experimental settings remain frozen.
        # Accept the original directory label without rewriting the existing JSON.
        old_normalized = dict(old)
        if old_normalized.get("instance_directory") == "instances_stress_n500":
            old_normalized["instance_directory"] = INSTANCE_DIR.name
        if old_normalized != design:
            raise RuntimeError(
                "The existing result folder belongs to a different experiment "
                "design. Preserve the original files and review the configuration."
            )
    else:
        DESIGN_JSON.write_text(canonical + "\n", encoding="utf-8")


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

CHECKPOINT_COLUMNS = [
    "Method",
    "Instance",
    "N",
    "ni",
    "nk",
    "nj",
    "p",
    "alpha",
    "instance_seed",
    "algorithm_seed",
    "C",
    "time_s",
    "wall_time_s",
    "iterations",
    "status",
    "audit_abs_diff",
    "path_relink_calls",
    "path_relink_seconds",
    "PR_steps",
    "PR_A_steps",
    "PR_S1_steps",
    "PR_S2_steps",
    "feasible",
    "error",
]


def result_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["Method"]),
        str(row["Instance"]),
        int(row["algorithm_seed"]),
    )


def load_checkpoint() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not CHECKPOINT_CSV.exists():
        return rows

    with CHECKPOINT_CSV.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter=";")
        for raw in reader:
            if not raw.get("Method") or not raw.get("Instance"):
                continue

            row = dict(raw)
            for k in ("N", "ni", "nk", "nj", "p", "instance_seed", "algorithm_seed"):
                row[k] = int(row[k])
            row["alpha"] = float(row["alpha"])

            for k in ("C", "time_s", "wall_time_s", "audit_abs_diff"):
                row[k] = float(row[k]) if row.get(k) not in ("", None) else None

            for k in (
                "iterations",
                "path_relink_calls",
                "PR_steps",
                "PR_A_steps",
                "PR_S1_steps",
                "PR_S2_steps",
                "feasible",
            ):
                if row.get(k) not in ("", None):
                    row[k] = int(row[k])

            if row.get("path_relink_seconds") not in ("", None):
                row["path_relink_seconds"] = float(row["path_relink_seconds"])

            rows[result_key(row)] = row

    return rows


def save_checkpoint(
    results: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    method_order = {name: idx for idx, name in enumerate(METHODS)}
    seed_order = {seed: idx for idx, seed in enumerate(ALGORITHM_SEEDS)}

    ordered = sorted(
        results.values(),
        key=lambda r: (
            method_order.get(str(r["Method"]), 10**9),
            int(r["instance_seed"]),
            int(r["N"]),
            int(r["p"]),
            float(r["alpha"]),
            str(r["Instance"]),
            seed_order.get(int(r["algorithm_seed"]), 10**9),
        ),
    )

    temp = CHECKPOINT_CSV.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=CHECKPOINT_COLUMNS,
            delimiter=";",
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow({key: row.get(key, "") for key in CHECKPOINT_COLUMNS})

    try:
        temp.replace(CHECKPOINT_CSV)
    except PermissionError:
        backup = CHECKPOINT_CSV.with_name(CHECKPOINT_CSV.stem + "_backup.csv")
        if backup.exists():
            backup.unlink()
        temp.replace(backup)
        print(f"[AVISO] Checkpoint bloqueado. Backup salvo em {backup.name}")


# =============================================================================
# TEXT REPORT
# =============================================================================

def initialize_report(instances: list[InstanceData]) -> None:
    if REPORT_TXT.exists():
        return

    expected_runs = len(instances) * len(METHODS) * len(ALGORITHM_SEEDS)

    with REPORT_TXT.open("w", encoding="utf-8") as fh:
        fh.write("=" * 118 + "\n")
        fh.write("STRESS TEST N=500 - ILvAS1P vs QTLA TR+LP+NG - 30 SEEDS\n")
        fh.write("=" * 118 + "\n")
        fh.write(f"Instances              : {len(instances)}\n")
        fh.write(f"Instance seeds          : {EXPECTED_INSTANCE_SEEDS}\n")
        fh.write(f"Methods                 : {METHODS}\n")
        fh.write(f"Algorithm seeds         : {ALGORITHM_SEEDS}\n")
        fh.write(f"Expected runs           : {expected_runs}\n")

        fh.write("\nFROZEN ILvAS1P PARAMETERS\n")
        for key, value in FROZEN_ILVAS1P.items():
            fh.write(f"  {key:<30s}: {value}\n")
        fh.write(f"  {'pr_trigger_no_improve':<30s}: {PR_TRIGGER_NO_IMPROVE}\n")

        fh.write("\nFROZEN QTLA FINAL CONFIGURATION\n")
        for key, value in QTLA_FINAL_CONFIG.items():
            fh.write(f"  {key:<30s}: {value}\n")

        fh.write("\nFROZEN QTLA PARAMETERS\n")
        for key, value in FROZEN_QTLA_PARAMETERS.items():
            fh.write(f"  {key:<30s}: {value}\n")

        fh.write("\nINSTANCES\n")
        for dt in instances:
            fh.write(
                f"  {dt.name} | N={dt.N} | I={dt.n_i} | K={dt.n_k} | "
                f"J={dt.n_j} | p={dt.p} | alpha={dt.alpha} | "
                f"instance_seed={dt.instance_seed}\n"
            )

        fh.write("\n" + "=" * 118 + "\n\n")


def append_report(row: dict[str, Any]) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as fh:
        fh.write(
            f"{row['Method']:15s} | {row['Instance']} | "
            f"iseed={int(row['instance_seed']):2d} | "
            f"aseed={int(row['algorithm_seed']):6d} | "
            f"C={float(row['C']):.10f} | "
            f"time={float(row['time_s']):.6f}s | "
            f"wall={float(row['wall_time_s']):.6f}s | "
            f"iter={int(row['iterations']):3d} | "
            f"status={row['status']} | "
            f"audit={float(row['audit_abs_diff']):.3e}\n"
        )


def append_error(
    method: str,
    dt: InstanceData,
    algorithm_seed: int,
    exc: Exception,
) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as fh:
        fh.write(
            f"ERROR | {method:15s} | {dt.name} | "
            f"iseed={dt.instance_seed:2d} | aseed={algorithm_seed:6d} | "
            f"{type(exc).__name__}: {exc}\n"
        )


def append_final_summary(
    instances: list[InstanceData],
    results: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    marker = "FINAL MEAN SUMMARY"
    existing = REPORT_TXT.read_text(encoding="utf-8") if REPORT_TXT.exists() else ""
    if marker in existing:
        return

    with REPORT_TXT.open("a", encoding="utf-8") as fh:
        fh.write("\n" + "=" * 118 + "\n")
        fh.write(marker + "\n")
        fh.write("=" * 118 + "\n")
        for method in METHODS:
            fh.write(f"\n[{method}]\n")
            for dt in instances:
                rows = [
                    results.get((method, dt.name, seed))
                    for seed in ALGORITHM_SEEDS
                ]
                valid = [r for r in rows if r is not None and not r.get("error")]
                if not valid:
                    fh.write(f"{dt.name} | no valid runs\n")
                    continue

                mean_c = float(np.mean([float(r["C"]) for r in valid]))
                mean_t = float(np.mean([float(r["time_s"]) for r in valid]))
                fh.write(
                    f"{dt.name} | n={len(valid):2d} | "
                    f"Mean_C={mean_c:.10f} | Mean_Time_s={mean_t:.6f}\n"
                )


# =============================================================================
# XLSX WRITER - standard library only
# One worksheet per final method, 30 seed C/time pairs + means.
# =============================================================================

def excel_col(n: int) -> str:
    out = ""
    while n:
        n, rem = divmod(n - 1, 26)
        out = chr(65 + rem) + out
    return out


def xml_text(value: Any) -> str:
    return xml_escape(
        "" if value is None else str(value),
        {'"': "&quot;", "'": "&apos;"},
    )


def xlsx_cell(
    ref: str,
    value: Any = None,
    style: int = 0,
    formula: str | None = None,
) -> str:
    s_attr = f' s="{style}"' if style else ""

    if formula is not None:
        return (
            f'<c r="{ref}"{s_attr}>'
            f'<f>{xml_text(formula)}</f>'
            f'</c>'
        )

    if value is None or value == "":
        return f'<c r="{ref}"{s_attr}/>'

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f'<c r="{ref}"{s_attr}><v>{int(value)}</v></c>'

    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return f'<c r="{ref}"{s_attr}><v>{float(value):.15g}</v></c>'

    return (
        f'<c r="{ref}" t="inlineStr"{s_attr}>'
        f'<is><t>{xml_text(value)}</t></is></c>'
    )


def build_sheet_xml(
    method: str,
    instances: list[InstanceData],
    results: dict[tuple[str, str, int], dict[str, Any]],
) -> str:
    headers = [
        "Instance",
        "N",
        "ni",
        "nk",
        "nj",
        "p",
        "alpha",
        "instance_seed",
    ]

    for seed in ALGORITHM_SEEDS:
        headers.extend([f"Seed{seed}_C", f"Seed{seed}_Time_s"])

    headers.extend(["Mean_C", "Mean_Time_s"])

    rows_xml: list[str] = []

    header_cells = [
        xlsx_cell(f"{excel_col(j)}1", h, style=1)
        for j, h in enumerate(headers, start=1)
    ]
    rows_xml.append(
        f'<row r="1" ht="28" customHeight="1">{"".join(header_cells)}</row>'
    )

    first_seed_col = 9
    mean_c_col = first_seed_col + 2 * len(ALGORITHM_SEEDS)
    mean_t_col = mean_c_col + 1

    for r_idx, dt in enumerate(instances, start=2):
        cells = [
            xlsx_cell(f"A{r_idx}", dt.name),
            xlsx_cell(f"B{r_idx}", dt.N, style=5),
            xlsx_cell(f"C{r_idx}", dt.n_i, style=5),
            xlsx_cell(f"D{r_idx}", dt.n_k, style=5),
            xlsx_cell(f"E{r_idx}", dt.n_j, style=5),
            xlsx_cell(f"F{r_idx}", dt.p, style=5),
            xlsx_cell(f"G{r_idx}", dt.alpha, style=2),
            xlsx_cell(f"H{r_idx}", dt.instance_seed, style=5),
        ]

        c_refs: list[str] = []
        t_refs: list[str] = []

        for s_pos, seed in enumerate(ALGORITHM_SEEDS):
            c_col_num = first_seed_col + 2 * s_pos
            t_col_num = c_col_num + 1
            c_ref = f"{excel_col(c_col_num)}{r_idx}"
            t_ref = f"{excel_col(t_col_num)}{r_idx}"
            c_refs.append(c_ref)
            t_refs.append(t_ref)

            row = results.get((method, dt.name, seed))
            if row is not None and not row.get("error"):
                cells.append(xlsx_cell(c_ref, row.get("C"), style=3))
                cells.append(xlsx_cell(t_ref, row.get("time_s"), style=4))
            else:
                cells.append(xlsx_cell(c_ref, None, style=3))
                cells.append(xlsx_cell(t_ref, None, style=4))

        mean_c_ref = f"{excel_col(mean_c_col)}{r_idx}"
        mean_t_ref = f"{excel_col(mean_t_col)}{r_idx}"

        cells.append(
            xlsx_cell(
                mean_c_ref,
                style=6,
                formula=f"AVERAGE({','.join(c_refs)})",
            )
        )
        cells.append(
            xlsx_cell(
                mean_t_ref,
                style=7,
                formula=f"AVERAGE({','.join(t_refs)})",
            )
        )

        rows_xml.append(f'<row r="{r_idx}">{"".join(cells)}</row>')

    last_row = len(instances) + 1
    last_col = excel_col(len(headers))

    col_widths = {
        1: 38,
        2: 8,
        3: 8,
        4: 8,
        5: 8,
        6: 8,
        7: 10,
        8: 14,
    }
    for col_num in range(9, mean_c_col):
        col_widths[col_num] = 15 if (col_num - 9) % 2 == 0 else 14
    col_widths[mean_c_col] = 16
    col_widths[mean_t_col] = 16

    cols_xml = "".join(
        f'<col min="{i}" max="{i}" width="{w}" customWidth="1"/>'
        for i, w in col_widths.items()
    )

    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane xSplit="8" ySplit="1" topLeftCell="I2" activePane="bottomRight" state="frozen"/>
    </sheetView>
  </sheetViews>
  <cols>{cols_xml}</cols>
  <sheetData>{"".join(rows_xml)}</sheetData>
  <autoFilter ref="A1:{last_col}{last_row}"/>
</worksheet>'''


def write_xlsx(
    instances: list[InstanceData],
    results: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_names = METHODS

    workbook_sheets = []
    workbook_rels = []
    content_overrides = []

    for idx, name in enumerate(sheet_names, start=1):
        workbook_sheets.append(
            f'<sheet name="{xml_text(name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
        content_overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            f'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )

    styles_rid = len(sheet_names) + 1
    workbook_rels.append(
        f'<Relationship Id="rId{styles_rid}" '
        f'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" '
        f'Target="styles.xml"/>'
    )

    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>{"".join(workbook_sheets)}</sheets>
  <calcPr calcId="191029" calcMode="auto" fullCalcOnLoad="1" forceFullCalc="1"/>
</workbook>'''

    workbook_rels_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {"".join(workbook_rels)}
</Relationships>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <numFmts count="4">
    <numFmt numFmtId="164" formatCode="0.0"/>
    <numFmt numFmtId="165" formatCode="0.0000000000"/>
    <numFmt numFmtId="166" formatCode="0.000000"/>
    <numFmt numFmtId="167" formatCode="0.0000000000"/>
  </numFmts>
  <fonts count="3">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FF1F1F1F"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="4">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border>
      <left style="thin"><color rgb="FFD9E2F3"/></left>
      <right style="thin"><color rgb="FFD9E2F3"/></right>
      <top style="thin"><color rgb="FFD9E2F3"/></top>
      <bottom style="thin"><color rgb="FFD9E2F3"/></bottom>
      <diagonal/>
    </border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="8">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="1" xfId="0" applyFill="1" applyFont="1">
      <alignment horizontal="center" vertical="center" wrapText="1"/>
    </xf>
    <xf numFmtId="164" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="165" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="166" fontId="0" fillId="0" borderId="1" xfId="0" applyNumberFormat="1"/>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"/>
    <xf numFmtId="167" fontId="2" fillId="3" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyFont="1"/>
    <xf numFmtId="166" fontId="2" fillId="3" borderId="1" xfId="0" applyNumberFormat="1" applyFill="1" applyFont="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>'''

    content_types = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  {"".join(content_overrides)}
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>'''

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="xl/workbook.xml"/>
  <Relationship Id="rId2"
    Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties"
    Target="docProps/core.xml"/>
  <Relationship Id="rId3"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties"
    Target="docProps/app.xml"/>
</Relationships>'''

    core_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties
 xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>Stress test N=500 - ILvAS1P vs QTLA TR+LP+NG - 30 seeds</dc:title>
 <dc:creator>Stress-test runner</dc:creator>
</cp:coreProperties>'''

    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Python</Application>
</Properties>'''

    temp_xlsx = XLSX_FILE.with_name(XLSX_FILE.stem + "_tmp.xlsx")

    with zipfile.ZipFile(temp_xlsx, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        z.writestr("xl/styles.xml", styles_xml)

        for idx, method in enumerate(sheet_names, start=1):
            z.writestr(
                f"xl/worksheets/sheet{idx}.xml",
                build_sheet_xml(method, instances, results),
            )

    try:
        temp_xlsx.replace(XLSX_FILE)
    except PermissionError:
        backup = XLSX_FILE.with_name(XLSX_FILE.stem + "_backup.xlsx")
        if backup.exists():
            backup.unlink()
        temp_xlsx.replace(backup)
        print(
            f"[AVISO] Excel principal esta aberto/bloqueado. "
            f"Resultado parcial salvo em: {backup.name}"
        )


# =============================================================================
# EXECUTION SCHEDULE
# =============================================================================

def build_run_schedule(
    instances: list[InstanceData],
) -> list[tuple[str, InstanceData, int]]:
    jobs: list[tuple[str, InstanceData, int]] = []

    # Paired, order-balanced execution:
    # same instance + same algorithm seed for both methods;
    # method order alternates across the 30 seeds.
    for dt in instances:
        for seed_pos, algorithm_seed in enumerate(ALGORITHM_SEEDS):
            order = METHODS if seed_pos % 2 == 0 else list(reversed(METHODS))
            for method in order:
                jobs.append((method, dt, algorithm_seed))

    return jobs


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    validate_algorithm_seeds()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    instances = load_stress_instances()
    verify_or_write_design(instances)
    initialize_report(instances)

    results = load_checkpoint()
    schedule = build_run_schedule(instances)

    total_expected = len(schedule)
    valid_before = sum(
        1 for row in results.values()
        if not row.get("error")
    )

    print("=" * 118)
    print("STRESS TEST N=500 - ILvAS1P vs QTLA TR+LP+NG - 30 SEEDS")
    print("=" * 118)
    print(f"Instancias encontradas : {len(instances)}")
    print(f"Seeds das instancias   : {EXPECTED_INSTANCE_SEEDS}")
    print(f"Metodos                : {METHODS}")
    print(f"Seeds dos algoritmos   : {ALGORITHM_SEEDS}")
    print(f"Execucoes esperadas    : {total_expected}")
    print(f"Ja concluidas          : {valid_before}")
    print(f"Resultados             : {RESULT_DIR}")
    print(f"Excel                  : {XLSX_FILE.name}")
    print()

    # Valid partial workbook exists from the beginning.
    write_xlsx(instances, results)

    runs_since_excel = 0

    for position, (method, dt, algorithm_seed) in enumerate(schedule, start=1):
        key = (method, dt.name, algorithm_seed)
        old = results.get(key)

        if old is not None and not old.get("error"):
            continue

        print(
            f"[{position:5d}/{total_expected}] "
            f"{method:15s} | {dt.name} | "
            f"iseed={dt.instance_seed:2d} | "
            f"aseed={algorithm_seed:6d} | rodando...",
            end="",
            flush=True,
        )

        try:
            row = run_method(method, dt, algorithm_seed)
            results[key] = row

            print(
                f" OK | C={row['C']:.8f} | "
                f"time={row['time_s']:.4f}s | "
                f"audit={row['audit_abs_diff']:.2e}"
            )
            append_report(row)

        except Exception as exc:
            row = {
                "Method": method,
                "Instance": dt.name,
                "N": dt.N,
                "ni": dt.n_i,
                "nk": dt.n_k,
                "nj": dt.n_j,
                "p": dt.p,
                "alpha": dt.alpha,
                "instance_seed": dt.instance_seed,
                "algorithm_seed": algorithm_seed,
                "C": None,
                "time_s": None,
                "wall_time_s": None,
                "iterations": None,
                "status": "ERROR",
                "audit_abs_diff": None,
                "path_relink_calls": "",
                "path_relink_seconds": "",
                "PR_steps": "",
                "PR_A_steps": "",
                "PR_S1_steps": "",
                "PR_S2_steps": "",
                "feasible": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            results[key] = row

            print(f" ERRO | {type(exc).__name__}: {exc}")
            append_error(method, dt, algorithm_seed, exc)
            traceback.print_exc()

        # Resume-safe checkpoint after every execution.
        save_checkpoint(results)

        runs_since_excel += 1
        if runs_since_excel >= 10:
            write_xlsx(instances, results)
            runs_since_excel = 0

    # Final outputs.
    save_checkpoint(results)
    write_xlsx(instances, results)

    valid_final = sum(
        1 for row in results.values()
        if not row.get("error")
    )
    errors_final = sum(
        1 for row in results.values()
        if row.get("error")
    )

    if valid_final == total_expected and errors_final == 0:
        append_final_summary(instances, results)

    print()
    print("=" * 118)
    print("STRESS TEST N=500 FINISHED")
    print("=" * 118)
    print(f"Expected runs : {total_expected}")
    print(f"Valid runs    : {valid_final}")
    print(f"Errors        : {errors_final}")
    print(f"Checkpoint    : {CHECKPOINT_CSV}")
    print(f"Report        : {REPORT_TXT}")
    print(f"Design        : {DESIGN_JSON}")
    print(f"Excel         : {XLSX_FILE}")

    if valid_final != total_expected or errors_final != 0:
        raise RuntimeError(
            "Stress-test experiment ended with missing/failed runs. "
            "Rerun the same script to resume from the checkpoint."
        )


if __name__ == "__main__":
    main()
