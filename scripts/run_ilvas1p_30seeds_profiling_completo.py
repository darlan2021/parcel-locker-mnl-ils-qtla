from __future__ import annotations

from dataclasses import dataclass
import time
from collections import defaultdict
from hashlib import sha256
import json
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any
from xml.sax.saxutils import escape as xml_escape
import csv
import math
import traceback
import zipfile

import numpy as np



# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mnl_ils import ILSConfig, MNLParcelLockerILS, SearchProfile, SearchResult

INSTANCE_DIR = ROOT / "instances" / "instances_seed_1_N_60_100_200_400"

RESULT_DIR = ROOT / "results" / "ils" / "ils_ilvas1p_30seeds_PROFILING_COMPLETO"
CHECKPOINT_CSV = RESULT_DIR / "checkpoint_ilvas1p_30seeds.csv"
REPORT_TXT = RESULT_DIR / "relatorio_ilvas1p_30seeds.txt"
XLSX_FILE = RESULT_DIR / "resultados_ilvas1p_30seeds.xlsx"
PROFILE_XLSX = RESULT_DIR / "profiling_completo.xlsx"
PROFILE_CSV = RESULT_DIR / "profiling_por_execucao.csv"
HISTORY_CSV = RESULT_DIR / "historico_iteracoes_pr.csv"
SUMMARY_5A_CSV = RESULT_DIR / "table_X_PR_time_distribution.csv"
SUMMARY_5B_CSV = RESULT_DIR / "table_Y_PR_local_search.csv"
GROUP_CSV = RESULT_DIR / "profiling_por_grupo_N_p.csv"
INSTANCE_CSV = RESULT_DIR / "profiling_por_instancia.csv"
MANIFEST_JSON = RESULT_DIR / "experiment_manifest.json"
PROFILE_FIELDS = list(SearchProfile.__dataclass_fields__)
PROFILE_DERIVED = ["other_seconds", "local_search_other_seconds"]
PROFILE_COLUMNS = ["profile_" + k for k in PROFILE_FIELDS + PROFILE_DERIVED]
ADDITIONAL_COLUMNS = [
    "iterations", "perturbation_strength_final", "y_bits",
    "open_existing", "closed_existing", "open_candidates", "history_json",
    "A_pct_VND", "S1_pct_VND", "S2_pct_VND", "repair_pct_VND",
    "other_pct_VND", "grasp_pct_total", "VND_pct_total",
    "perturb_pct_total", "PR_pct_total", "other_pct_total",
]



# =============================================================================
# EXPERIMENT: FINAL FROZEN ILS[A+S1]+PR
# =============================================================================
# This is the original calibrated PR implementation copied verbatim from
# run_ilvas1p_30seeds(1).py (PRTriggeredILS), plus full profiling.
# 24 instances * 30 heuristic seeds = 720 runs. PR is ON.
# PR activation: no improvement for 5, 10, ... consecutive iterations.
# PR is unidirectional; no S2 in VND or inside PR.
# Core irace ID 7 and all original ILvAS1P parameters are unchanged.
# The solver is limited by max_iterations=10; time_limit=None (as original).
# =============================================================================

HEURISTIC_SEEDS = [
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

EXPECTED_NPZ_INSTANCES = 24

EXPECTED_DESIGN = {
    60: [3, 6],
    100: [5, 10],
    200: [10, 20],
    400: [20, 30],
}

EXPECTED_ALPHAS = [0.5, 1.0, 2.0]

# Exact original frozen core and PR calibration. No respecification.
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

    # The old periodic PR trigger is disabled.
    "path_relink_interval": 0,
    "path_relink_bidirectional": False,
    "path_relink_s2_samples": 0,
}

PR_TRIGGER_NO_IMPROVE = 5

CONFIGURATIONS = {
    "ILvAS1P": {}
}

# Suppress internal ILS progress printing.
ILS_VERBOSE = False




# =============================================================================
# LIGHTWEIGHT INSTANCE OBJECT
# =============================================================================

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

    @property
    def total_N(self) -> int:
        return self.n_i + self.n_k + self.n_j


def _scalar(npz: Any, key: str, cast):
    value = npz[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_instance(path: Path) -> InstanceData:
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
        seed = _scalar(z, "seed", int)
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
        seed=seed,
        a=a,
        b=b,
    )


def validate_heuristic_seeds() -> None:
    if len(HEURISTIC_SEEDS) != 30:
        raise RuntimeError(
            f"Expected exactly 30 heuristic seeds, found {len(HEURISTIC_SEEDS)}."
        )

    if len(set(HEURISTIC_SEEDS)) != 30:
        raise RuntimeError("Duplicate heuristic seeds detected.")

    if any(seed < 0 or seed > 500000 for seed in HEURISTIC_SEEDS):
        raise RuntimeError("All heuristic seeds must be in [0, 500000].")

    ordered = sorted(HEURISTIC_SEEDS)
    consecutive = [
        (a, b)
        for a, b in zip(ordered, ordered[1:])
        if b - a == 1
    ]
    if consecutive:
        raise RuntimeError(
            f"Consecutive heuristic seed values detected: {consecutive}"
        )


def load_all_instances() -> list[InstanceData]:
    if not INSTANCE_DIR.exists():
        raise FileNotFoundError(
            f"Folder not found: {INSTANCE_DIR}\n"
            "Place this run file inside scripts/ and verify the official "
            "seed-1 instances directory under instances/."
        )

    files = sorted(INSTANCE_DIR.glob("*.npz"))

    if len(files) != EXPECTED_NPZ_INSTANCES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_NPZ_INSTANCES} .npz instances in "
            f"{INSTANCE_DIR}, but found {len(files)}. "
            "The pilot folder must contain the original 18 instances plus "
            "the 6 added N=400 instances."
        )

    instances = [load_instance(p) for p in files]

    # All pilot instances must use one common instance seed.
    instance_seeds = sorted({dt.seed for dt in instances})
    if instance_seeds != [1]:
        raise RuntimeError(
            "The pilot folder must contain exactly one instance seed. "
            f"Found: {instance_seeds}"
        )

    # Validate exact design: 4 N groups x 2 p levels x 3 alpha levels = 24.
    observed = {
        (dt.total_N, dt.p, round(float(dt.alpha), 10))
        for dt in instances
    }
    expected = {
        (N, p, round(float(alpha), 10))
        for N, p_values in EXPECTED_DESIGN.items()
        for p in p_values
        for alpha in EXPECTED_ALPHAS
    }

    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    if missing or extra:
        raise RuntimeError(
            "Pilot instance design is not the expected 24-point design.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    if len(observed) != EXPECTED_NPZ_INSTANCES:
        raise RuntimeError(
            "Duplicate N/p/alpha design points detected in the pilot folder."
        )

    instances.sort(
        key=lambda d: (d.total_N, d.p, d.alpha, d.seed, d.name)
    )
    return instances


# =============================================================================
# STAGNATION-TRIGGERED PATH RELINKING (UNCHANGED FROM ORIGINAL RUN)
# =============================================================================

# The calibrated PR activation behavior is intentionally kept in the subclass.
class PRTriggeredILS(MNLParcelLockerILS):
    """
    Same MNLParcelLockerILS implementation, except for the PR activation rule.

    For q = PR_TRIGGER_NO_IMPROVE, Path Relinking is attempted at
    q, 2q, 3q, ... consecutive ILS iterations without global improvement.
    Any global improvement resets the stagnation counter.
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




def make_config(configuration_name: str, heuristic_seed: int) -> ILSConfig:
    if configuration_name != "ILvAS1P":
        raise KeyError(f"Unexpected configuration: {configuration_name}")
    params = dict(FROZEN_ILVAS1P)
    params["seed"] = int(heuristic_seed)
    params["verbose"] = ILS_VERBOSE
    return ILSConfig(**params)

def run_one(
    dt: InstanceData,
    component_name: str,
    heuristic_seed: int,
) -> dict[str, Any]:
    cfg = make_config(component_name, heuristic_seed)
    solver = PRTriggeredILS(dt, cfg, PR_TRIGGER_NO_IMPROVE)

    wall0 = perf_counter()
    result = solver.solve()
    wall_time = perf_counter() - wall0

    if not result.feasible:
        raise RuntimeError(
            f"ILS returned an infeasible solution for {dt.name}, "
            f"{component_name}, seed={heuristic_seed}."
        )

    # Independent direct-objective audit.
    y = np.asarray(result.y, dtype=float)
    num = dt.a @ y
    den = dt.b @ y
    if np.any(den <= 0.0):
        raise FloatingPointError("Non-positive denominator in final solution audit.")

    direct_C = float(np.sum(num / den))
    audit_diff = abs(float(result.objective) - direct_C)

    if not np.isclose(float(result.objective), direct_C, rtol=1e-9, atol=1e-7):
        raise ArithmeticError(
            f"Independent C audit failed: reported={result.objective}, "
            f"recomputed={direct_C} ({dt.name}, {component_name}, {heuristic_seed})"
        )
    if dt.seed != 1:
        raise RuntimeError(f"Wrong instance seed: {dt.seed}")
    if len(y) != dt.n_k + dt.n_j or np.any((y != 0) & (y != 1)):
        raise ArithmeticError("Invalid binary solution vector")
    if int(np.sum(y[dt.n_k:])) > dt.p or int(np.sum(y[:dt.n_k])) < dt.n_k-dt.p:
        raise ArithmeticError("Final solution violates the redesign constraints")

    if int(result.profile.PR_S2_steps) != 0:
        raise RuntimeError("PR-S2 contamination: expected no S2 steps in Path Relinking")
    if (int(result.profile.S2_calls) != 0 or
        int(result.profile.S2_move_evaluations) != 0 or
        int(result.profile.S2_accepted) != 0):
        raise RuntimeError("S2 contamination: S2 must be OFF in the selected A+S1 configuration")
    profile = result.profile.to_dict()
    for key in PROFILE_FIELDS + PROFILE_DERIVED:
        if key not in profile:
            raise KeyError(f"Missing profiling field: {key}")
    total = max(float(profile["total_seconds"]), 1e-12)
    vnd = float(profile["local_search_seconds"])
    pcts = {
        "A_pct_VND": 100*float(profile["A_seconds"])/vnd if vnd else 0.,
        "S1_pct_VND": 100*float(profile["S1_seconds"])/vnd if vnd else 0.,
        "S2_pct_VND": 100*float(profile["S2_seconds"])/vnd if vnd else 0.,
        "repair_pct_VND": 100*float(profile["repair_seconds"])/vnd if vnd else 0.,
        "other_pct_VND": 100*float(profile["local_search_other_seconds"])/vnd if vnd else 0.,
        "grasp_pct_total": 100*float(profile["grasp_seconds"])/total,
        "VND_pct_total": 100*vnd/total,
        "perturb_pct_total": 100*float(profile["perturbation_seconds"])/total,
        "PR_pct_total": 100*float(profile["path_relink_seconds"])/total,
        "other_pct_total": 100*float(profile["other_seconds"])/total,
    }
    if not np.isclose(sum(pcts[k] for k in (
        "grasp_pct_total","VND_pct_total","perturb_pct_total",
        "PR_pct_total","other_pct_total")), 100, atol=.05):
        raise ArithmeticError("Main timing profile does not sum to 100%")

    return {
        "Configuration": component_name,
        "Instance": dt.name,
        "N": dt.total_N, "ni": dt.n_i, "nk": dt.n_k,
        "nj": dt.n_j, "p": dt.p, "alpha": dt.alpha,
        "instance_seed": dt.seed, "heuristic_seed": heuristic_seed,
        "C": direct_C,
        "time_s": float(result.elapsed_seconds),
        "wall_time_s": float(wall_time),
        "audit_abs_diff": audit_diff,
        "feasible": "SIM", "error": "",
        "pr_trigger_no_improve": PR_TRIGGER_NO_IMPROVE,
        "pr_bidirectional": False,
        "iterations": int(result.iterations),
        "perturbation_strength_final": int(result.perturbation_strength),
        "y_bits": "".join(str(int(v)) for v in y),
        "open_existing": json.dumps(result.open_existing),
        "closed_existing": json.dumps(result.closed_existing),
        "open_candidates": json.dumps(result.open_candidates),
        "history_json": json.dumps(result.history, separators=(",",":")),
        **{"profile_"+key: profile[key] for key in PROFILE_FIELDS+PROFILE_DERIVED},
        **pcts,
    }


# =============================================================================
# CHECKPOINT / RESUME
# =============================================================================

CHECKPOINT_COLUMNS = [
    "Configuration",
    "Instance",
    "N",
    "ni",
    "nk",
    "nj",
    "p",
    "alpha",
    "instance_seed",
    "heuristic_seed",
    "C",
    "time_s",
    "wall_time_s",
    "audit_abs_diff",
    "feasible",
    "error",
] + ["pr_trigger_no_improve", "pr_bidirectional"] + ADDITIONAL_COLUMNS + PROFILE_COLUMNS


def result_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["Configuration"]),
        str(row["Instance"]),
        int(row["heuristic_seed"]),
    )


def load_checkpoint() -> dict[tuple[str, str, int], dict[str, Any]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}

    if not CHECKPOINT_CSV.exists():
        return rows

    with CHECKPOINT_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        missing_cols = set(CHECKPOINT_COLUMNS) - set(reader.fieldnames or [])
        if missing_cols:
            raise RuntimeError(f"Existing checkpoint lacks profiling columns: {sorted(missing_cols)}")
        for raw in reader:
            if not raw.get("Configuration") or not raw.get("Instance"):
                continue

            row = dict(raw)
            row["N"] = int(row["N"])
            row["ni"] = int(row["ni"])
            row["nk"] = int(row["nk"])
            row["nj"] = int(row["nj"])
            row["p"] = int(row["p"])
            row["alpha"] = float(row["alpha"])
            row["instance_seed"] = int(row["instance_seed"])
            row["heuristic_seed"] = int(row["heuristic_seed"])

            for k in ("C", "time_s", "wall_time_s", "audit_abs_diff"):
                row[k] = float(row[k]) if row.get(k) not in ("", None) else None
            for k in PROFILE_COLUMNS + [
                "A_pct_VND","S1_pct_VND","S2_pct_VND","repair_pct_VND",
                "other_pct_VND","grasp_pct_total","VND_pct_total",
                "perturb_pct_total","PR_pct_total","other_pct_total",
            ]:
                row[k] = float(row[k]) if row.get(k) not in ("", None) else None
            for k in ("iterations", "perturbation_strength_final"):
                row[k] = int(row[k]) if row.get(k) not in ("", None) else None

            rows[result_key(row)] = row

    return rows


def save_checkpoint(
    results: dict[tuple[str, str, int], dict[str, Any]]
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    temp = CHECKPOINT_CSV.with_suffix(".tmp")

    config_order = {name: i for i, name in enumerate(CONFIGURATIONS)}

    ordered = sorted(
        results.values(),
        key=lambda r: (
            config_order[str(r["Configuration"])],
            int(r["N"]),
            int(r["p"]),
            float(r["alpha"]),
            str(r["Instance"]),
            int(r["heuristic_seed"]),
        ),
    )

    with temp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CHECKPOINT_COLUMNS,
            delimiter=";",
        )
        writer.writeheader()
        for row in ordered:
            writer.writerow({c: row.get(c, "") for c in CHECKPOINT_COLUMNS})

    try:
        temp.replace(CHECKPOINT_CSV)
    except PermissionError:
        backup = CHECKPOINT_CSV.with_name(
            CHECKPOINT_CSV.stem + "_backup.csv"
        )
        if backup.exists():
            backup.unlink()
        temp.replace(backup)
        print(
            f"[AVISO] checkpoint principal bloqueado. "
            f"Salvo em {backup.name}"
        )


# =============================================================================
# TEXT REPORT
# =============================================================================

def initialize_report(instances: list[InstanceData]) -> None:
    if REPORT_TXT.exists():
        return

    total_expected = (
        len(instances) * len(CONFIGURATIONS) * len(HEURISTIC_SEEDS)
    )

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write("=" * 110 + "\n")
        f.write("ILS[A+S1]+PR - FULL PROFILING - 30 HEURISTIC SEEDS\n")
        f.write("=" * 110 + "\n")
        f.write(f"Instances              : {len(instances)}\n")
        f.write(f"Instance seed          : {instances[0].seed}\n")
        f.write(f"Configurations         : {', '.join(CONFIGURATIONS)}\n")
        f.write(f"Heuristic seeds        : {HEURISTIC_SEEDS}\n")
        f.write("Improvement strategy   : BI (first_improvement=False)\n")
        f.write("Path Relinking         : ON | trigger=5 | bidirectional=False | PR-S2=0\n")
        f.write(f"Expected runs          : {total_expected}\n")

        f.write("\nFROZEN IRACE ID 7 PARAMETERS\n")
        for key, value in FROZEN_ILVAS1P.items():
            f.write(f"  {key} = {value}\n")

        f.write("\nCOMPONENT FLAGS\n")
        for name, flags in CONFIGURATIONS.items():
            f.write(f"  {name}: {flags}\n")

        f.write("\nINSTANCES\n")
        for dt in instances:
            f.write(
                f"  {dt.name} | N={dt.total_N} | p={dt.p} | "
                f"alpha={dt.alpha} | instance_seed={dt.seed}\n"
            )

        f.write("\n" + "=" * 110 + "\n\n")


def append_report(row: dict[str, Any]) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write(
            f"{row['Configuration']:8s} | "
            f"{row['Instance']} | "
            f"seed={int(row['heuristic_seed']):6d} | "
            f"C={float(row['C']):.10f} | "
            f"time={float(row['time_s']):.6f}s | "
            f"PRcalls={int(row['profile_path_relink_calls'])} | "
            f"PRtime={float(row['profile_path_relink_seconds']):.6f}s | "
            f"A_evals={int(row['profile_A_move_evaluations'])} | "
            f"S1_evals={int(row['profile_S1_move_evaluations'])} | "
            f"A_accepted={int(row['profile_A_accepted'])} | "
            f"S1_accepted={int(row['profile_S1_accepted'])} | "
            f"audit={float(row['audit_abs_diff']):.3e}\n"
        )


def append_error(
    dt: InstanceData,
    component_name: str,
    heuristic_seed: int,
    exc: Exception,
) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write(
            f"ERROR | {component_name:8s} | {dt.name} | "
            f"seed={heuristic_seed:6d} | "
            f"{type(exc).__name__}: {exc}\n"
        )


# =============================================================================
# XLSX WRITER - standard library only
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
    sheet_name: str,
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

    for seed in HEURISTIC_SEEDS:
        headers.extend([f"Seed{seed}_C", f"Seed{seed}_Time_s"])

    headers.extend(["Mean_C", "Mean_Time_s"])

    rows_xml: list[str] = []

    cells = []
    for j, h in enumerate(headers, start=1):
        cells.append(xlsx_cell(f"{excel_col(j)}1", h, style=1))
    rows_xml.append(
        f'<row r="1" ht="28" customHeight="1">{"".join(cells)}</row>'
    )

    first_seed_col = 9
    mean_c_col = first_seed_col + 2 * len(HEURISTIC_SEEDS)
    mean_t_col = mean_c_col + 1

    for r_idx, dt in enumerate(instances, start=2):
        cells = [
            xlsx_cell(f"A{r_idx}", dt.name),
            xlsx_cell(f"B{r_idx}", dt.total_N, style=5),
            xlsx_cell(f"C{r_idx}", dt.n_i, style=5),
            xlsx_cell(f"D{r_idx}", dt.n_k, style=5),
            xlsx_cell(f"E{r_idx}", dt.n_j, style=5),
            xlsx_cell(f"F{r_idx}", dt.p, style=5),
            xlsx_cell(f"G{r_idx}", dt.alpha, style=2),
            xlsx_cell(f"H{r_idx}", dt.seed, style=5),
        ]

        c_refs = []
        t_refs = []

        for s_pos, seed in enumerate(HEURISTIC_SEEDS):
            c_col_num = first_seed_col + 2 * s_pos
            t_col_num = c_col_num + 1

            c_ref = f"{excel_col(c_col_num)}{r_idx}"
            t_ref = f"{excel_col(t_col_num)}{r_idx}"
            c_refs.append(c_ref)
            t_refs.append(t_ref)

            row = results.get((sheet_name, dt.name, seed))
            if row is not None and not row.get("error"):
                cells.append(
                    xlsx_cell(c_ref, row.get("C"), style=3)
                )
                cells.append(
                    xlsx_cell(t_ref, row.get("time_s"), style=4)
                )
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

    sheet_names = list(CONFIGURATIONS.keys())

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
 <dc:title>ILS[A+S1]+PR calibrated - instance seed 1 - full profiling</dc:title>
 <dc:creator>ILS comparison runner</dc:creator>
</cp:coreProperties>'''

    app_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
 <Application>Python</Application>
</Properties>'''

    temp_xlsx = XLSX_FILE.with_name(XLSX_FILE.stem + "_tmp.xlsx")

    with zipfile.ZipFile(
        temp_xlsx,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core_xml)
        z.writestr("docProps/app.xml", app_xml)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        z.writestr("xl/styles.xml", styles_xml)

        for idx, name in enumerate(sheet_names, start=1):
            z.writestr(
                f"xl/worksheets/sheet{idx}.xml",
                build_sheet_xml(name, instances, results),
            )

    try:
        temp_xlsx.replace(XLSX_FILE)
    except PermissionError:
        backup = XLSX_FILE.with_name(
            XLSX_FILE.stem + "_backup.xlsx"
        )
        if backup.exists():
            backup.unlink()
        temp_xlsx.replace(backup)
        print(
            f"[AVISO] Excel principal esta aberto/bloqueado. "
            f"Resultado parcial salvo em: {backup.name}"
        )


# =============================================================================
# COMPLETE PROFILING, AUDIT MANIFEST AND SUPPLEMENTARY TABLES
# =============================================================================

def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_or_create_manifest(instances: list[InstanceData]) -> None:
    """Prevent silently resuming with different seeds, inputs or frozen settings."""
    source = SRC_DIR / "mnl_ils.py"
    if not source.exists():
        raise FileNotFoundError(f"ILS implementation not found: {source}")
    manifest = {
        "experiment": "Selected ILS[A+S1]+PR full profiling / irace ID 7 / calibrated stagnation PR",
        "instance_dir": "instances_pilot",
        "instance_seed": 1,
        "instance_files_sha256": {d.path.name: _sha256_file(d.path) for d in instances},
        "source_mnl_ils_sha256": _sha256_file(source),
        "frozen_ilvas1p": FROZEN_ILVAS1P,
        "pr_trigger_no_improve": PR_TRIGGER_NO_IMPROVE,
        "pr_impl": "PRTriggeredILS copied unchanged from run_ilvas1p_30seeds(1).py",
        "component_flags": CONFIGURATIONS,
        "heuristic_seeds": HEURISTIC_SEEDS,
        "expected_runs": EXPECTED_NPZ_INSTANCES * len(CONFIGURATIONS) * len(HEURISTIC_SEEDS),
        "timing_unit": "wall-clock seconds measured via time.perf_counter",
        "note": "A/S1/repair times are nested within VND; PR is unidirectional, no PR-S2; PR final VND is counted under VND.",
    }
    if MANIFEST_JSON.exists():
        old = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
        if old != manifest:
            diffs = [k for k in manifest if old.get(k) != manifest[k]]
            raise RuntimeError(
                "Experiment inputs, code or frozen parameters changed during a resumed run. "
                f"Differing manifest keys: {diffs}. Preserve the existing result folder "
                "and use a new output folder for a changed experiment."
            )
    else:
        if CHECKPOINT_CSV.exists():
            raise RuntimeError("Checkpoint exists without provenance manifest; refusing unsafe resume")
        MANIFEST_JSON.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def valid_profile_rows(results):
    ordered = sorted(
        (r for r in results.values() if not r.get("error")
         and r.get("profile_total_seconds") is not None),
        key=lambda r: (
            str(r["Configuration"]), int(r["N"]), int(r["p"]),
            float(r["alpha"]), int(r["heuristic_seed"])
        )
    )
    return ordered


def _mean(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals)/len(vals) if vals else None


def aggregate_row(rows, label):
    values = {
        "Configuration": label,
        "Runs": len(rows),
        "Mean_C": _mean(rows, "C"),
        "Time_s": _mean(rows, "time_s"),
        "GRASP_pct": _mean(rows, "grasp_pct_total"),
        "VND_pct": _mean(rows, "VND_pct_total"),
        "Perturbation_pct": _mean(rows, "perturb_pct_total"),
        "PR_pct": _mean(rows, "PR_pct_total"),
        "Other_pct": _mean(rows, "other_pct_total"),
        "GRASP_s": _mean(rows, "profile_grasp_seconds"),
        "VND_s": _mean(rows, "profile_local_search_seconds"),
        "Perturbation_s": _mean(rows, "profile_perturbation_seconds"),
        "PR_s": _mean(rows, "profile_path_relink_seconds"),
        "Other_s": _mean(rows, "profile_other_seconds"),
        "A_pct_VND": _mean(rows, "A_pct_VND"),
        "S1_pct_VND": _mean(rows, "S1_pct_VND"),
        "S2_pct_VND": _mean(rows, "S2_pct_VND"),
        "Repair_pct_VND": _mean(rows, "repair_pct_VND"),
        "Other_pct_VND": _mean(rows, "other_pct_VND"),
        "A_evaluations": _mean(rows, "profile_A_move_evaluations"),
        "A_accepted": _mean(rows, "profile_A_accepted"),
        "S1_evaluations": _mean(rows, "profile_S1_move_evaluations"),
        "S1_accepted": _mean(rows, "profile_S1_accepted"),
        "S2_evaluations": _mean(rows, "profile_S2_move_evaluations"),
        "S2_accepted": _mean(rows, "profile_S2_accepted"),
        "VND_calls": _mean(rows, "profile_local_search_calls"),
        "A_calls": _mean(rows, "profile_A_calls"),
        "S1_calls": _mean(rows, "profile_S1_calls"),
        "S2_calls": _mean(rows, "profile_S2_calls"),
        "GRASP_evaluations": _mean(rows, "profile_grasp_move_evaluations"),
        "Repair_calls": _mean(rows, "profile_repair_calls"),
        "Repair_moves": _mean(rows, "profile_repair_moves"),
        "Perturbation_calls": _mean(rows, "profile_perturbation_calls"),
        "Perturb_A_moves": _mean(rows, "profile_perturb_A_moves"),
        "Perturb_S1_moves": _mean(rows, "profile_perturb_S1_moves"),
        "Perturb_S2_moves": _mean(rows, "profile_perturb_S2_moves"),
        "PR_evaluations": _mean(rows, "profile_PR_move_evaluations"),
        "PR_steps": _mean(rows, "profile_PR_steps"),
    }
    return values


def build_summary_tables(results):
    rows = valid_profile_rows(results)
    by_np = defaultdict(list)
    by_instance = defaultdict(list)
    for r in rows:
        by_np[(int(r["N"]),int(r["p"]))].append(r)
        by_instance[r["Instance"]].append(r)
    groups = []
    table_x = []
    table_y = []
    for N, p in [(60,3),(60,6),(100,5),(100,10),(200,10),(200,20),(400,20),(400,30)]:
        agg = aggregate_row(by_np[(N,p)], "ILS[A+S1]+PR")
        groups.append({"N":N,"p":p,**agg})
        table_x.append({
            "N":N, "p":p, "Runs":agg["Runs"],
            "GRASP_pct":agg["GRASP_pct"], "VND_pct":agg["VND_pct"],
            "Perturbation_pct":agg["Perturbation_pct"],
            "PR_pct":agg["PR_pct"], "Other_pct":agg["Other_pct"],
            "Mean_Time_s":agg["Time_s"],
        })
        table_y.append({
            "N":N, "p":p, "Runs":agg["Runs"],
            "A_pct_VND":agg["A_pct_VND"], "S1_pct_VND":agg["S1_pct_VND"],
            "Repair_pct_VND":agg["Repair_pct_VND"],
            "Other_pct_VND":agg["Other_pct_VND"],
            "A_evaluations":agg["A_evaluations"],
            "S1_evaluations":agg["S1_evaluations"],
            "A_accepted":agg["A_accepted"],
            "S1_accepted":agg["S1_accepted"],
            "VND_calls":agg["VND_calls"],
        })
    overall = aggregate_row(rows, "ILS[A+S1]+PR")
    table_x.append({"N":"Overall", "p":"", "Runs":overall["Runs"],
        "GRASP_pct":overall["GRASP_pct"],"VND_pct":overall["VND_pct"],
        "Perturbation_pct":overall["Perturbation_pct"],
        "PR_pct":overall["PR_pct"],"Other_pct":overall["Other_pct"],
        "Mean_Time_s":overall["Time_s"]})
    table_y.append({"N":"Overall", "p":"", "Runs":overall["Runs"],
        "A_pct_VND":overall["A_pct_VND"],"S1_pct_VND":overall["S1_pct_VND"],
        "Repair_pct_VND":overall["Repair_pct_VND"],
        "Other_pct_VND":overall["Other_pct_VND"],
        "A_evaluations":overall["A_evaluations"],
        "S1_evaluations":overall["S1_evaluations"],
        "A_accepted":overall["A_accepted"],
        "S1_accepted":overall["S1_accepted"],
        "VND_calls":overall["VND_calls"]})
    instances = [{"Instance":name,**aggregate_row(by_instance[name],"ILS[A+S1]+PR")}
                 for name in sorted(by_instance)]
    params = ([{"Parameter":k,"Value":json.dumps(v,ensure_ascii=False)}
               for k,v in FROZEN_ILVAS1P.items()] + [
        {"Parameter":"pr_trigger_no_improve","Value":PR_TRIGGER_NO_IMPROVE},
        {"Parameter":"instance_seed","Value":1},
        {"Parameter":"instance_dir","Value":"instances_pilot"},
        {"Parameter":"heuristic_seeds","Value":json.dumps(HEURISTIC_SEEDS)},
        {"Parameter":"NOTE","Value":"A, S1 and repair time are nested within VND time"},
        {"Parameter":"PR core","Value":"PR post-search VND is included only in VND; do not double count"},
        {"Parameter":"AGGREGATION","Value":"Percentages calculated per run and then averaged"},
        {"Parameter":"DATA","Value":"Partial groups while running are preliminary"},
    ])
    return {"TABLE_X_PR":table_x, "TABLE_Y_VND":table_y,
        "BY_N_P":groups, "BY_INSTANCE":instances,
        "RUN_DATA":rows, "PARAMETERS":params}


def _csv_atomic(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";", extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    os.replace(temp, path)


def _generic_sheet_xml(rows, columns):
    headings = [xlsx_cell(f"{excel_col(j)}1",k,style=1)
                for j,k in enumerate(columns,start=1)]
    xml_rows = ['<row r="1" ht="27" customHeight="1">' + ''.join(headings) + '</row>']
    for index, row in enumerate(rows, start=2):
        cells=[]
        for ci, field in enumerate(columns,start=1):
            val=row.get(field)
            if isinstance(val,(dict,list,tuple)):
                val=json.dumps(val,ensure_ascii=False)
            if isinstance(val,(int,float,np.integer,np.floating)):
                style=3 if isinstance(val,float) else 5
            else:
                style=0
            cells.append(xlsx_cell(f"{excel_col(ci)}{index}",val,style=style))
        xml_rows.append(f'<row r="{index}">{"".join(cells)}</row>')
    widths=''.join(
        f'<col min="{i}" max="{i}" width="{min(48, max(14, len(c)+3))}" customWidth="1"/>'
        for i,c in enumerate(columns,start=1))
    last_row=max(1,len(rows)+1)
    last_col=excel_col(len(columns))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
      f'<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
      f'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
      f'<cols>{widths}</cols><sheetData>{"".join(xml_rows)}</sheetData>'
      f'<autoFilter ref="A1:{last_col}{last_row}"/></worksheet>')


def write_supplementary_xlsx(tables):
    """Reuse original one-sheet Excel as base; add six supplementary sheets."""
    with zipfile.ZipFile(XLSX_FILE,'r') as original:
        files={name:original.read(name) for name in original.namelist()}
    wbook=files['xl/workbook.xml'].decode('utf-8')
    rels=files['xl/_rels/workbook.xml.rels'].decode('utf-8')
    content=files['[Content_Types].xml'].decode('utf-8')
    next_id=len(CONFIGURATIONS)+1
    for sn,rows in tables.items():
        ix=next_id
        # Sheet 1 is the original; additional sheets start at 2
        rid=ix+1  # rId2 belongs to styles; first added sheet uses rId3
        wbook=wbook.replace('</sheets>', f'<sheet name="{xml_text(sn)}" sheetId="{ix}" r:id="rId{rid}"/></sheets>')
        rels=rels.replace('</Relationships>',
            '<Relationship Id="rId'+str(rid)+'" '+
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '+
            'Target="worksheets/sheet'+str(ix)+'.xml"/></Relationships>')
        content=content.replace('</Types>',
            '<Override PartName="/xl/worksheets/sheet'+str(ix)+'.xml" '+
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        cols=list(rows[0]) if rows else (["Configuration"] if sn != "RUN_DATA" else CHECKPOINT_COLUMNS)
        files[f'xl/worksheets/sheet{ix}.xml']=_generic_sheet_xml(rows,cols).encode('utf-8')
        next_id+=1
    files['xl/workbook.xml']=wbook.encode('utf-8')
    files['xl/_rels/workbook.xml.rels']=rels.encode('utf-8')
    files['[Content_Types].xml']=content.encode('utf-8')
    temp=PROFILE_XLSX.with_name(PROFILE_XLSX.stem+'_tmp.xlsx')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,blob in files.items():
            z.writestr(name,blob)
    try:
        os.replace(temp,PROFILE_XLSX)
    except PermissionError:
        backup=PROFILE_XLSX.with_name(PROFILE_XLSX.stem+'_backup.xlsx')
        os.replace(temp,backup)
        print(f'[WARNING] Profiling Excel locked, wrote {backup.name}')


def write_all_profiling_outputs(instances,results):
    tables=build_summary_tables(results)
    _csv_atomic(PROFILE_CSV,tables['RUN_DATA'],fieldnames=CHECKPOINT_COLUMNS)
    historical = []
    for row in tables['RUN_DATA']:
        history_text = row.get('history_json')
        if history_text:
            for entry in json.loads(history_text):
                historical.append({
                    'Configuration':row['Configuration'],
                    'Instance':row['Instance'],
                    'N':row['N'], 'p':row['p'], 'alpha':row['alpha'],
                    'instance_seed':row['instance_seed'],
                    'heuristic_seed':row['heuristic_seed'], **entry,
                })
    _csv_atomic(HISTORY_CSV,historical,fieldnames=[
        'Configuration','Instance','N','p','alpha','instance_seed',
        'heuristic_seed','iteration','best','current','strength',
        'stagnation','elite_size','pr_trigger_no_improve'])
    _csv_atomic(SUMMARY_5A_CSV,tables['TABLE_X_PR'])
    _csv_atomic(SUMMARY_5B_CSV,tables['TABLE_Y_VND'])
    _csv_atomic(GROUP_CSV,tables['BY_N_P'])
    _csv_atomic(INSTANCE_CSV,tables['BY_INSTANCE'])
    write_supplementary_xlsx(tables)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    validate_heuristic_seeds()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    instances = load_all_instances()
    validate_or_create_manifest(instances)
    initialize_report(instances)

    results = load_checkpoint()

    total_expected = (
        len(instances) * len(CONFIGURATIONS) * len(HEURISTIC_SEEDS)
    )

    completed_before = sum(
        1 for row in results.values() if not row.get("error")
    )

    print("=" * 118)
    print("ILS[A+S1]+PR - PROFILING COMPLETO - 30 SEEDS")
    print("=" * 118)
    print(f"Instancias encontradas : {len(instances)}")
    print(f"Seed das instancias    : {instances[0].seed}")
    print(f"Configuracoes          : {', '.join(CONFIGURATIONS)}")
    print(f"Seeds heuristica       : {HEURISTIC_SEEDS}")
    print("Estrategia             : BI")
    print("Path Relinking         : ATIVO | trigger=5 | unidirecional | PR-S2=0")
    print(f"Execucoes esperadas    : {total_expected}")
    print(f"Ja concluidas          : {completed_before}")
    print(f"Resultados             : {RESULT_DIR}")
    print()

    # Preserve the original per-seed C/Time table (one PR configuration).
    write_xlsx(instances, results)
    write_all_profiling_outputs(instances, results)

    config_names = list(CONFIGURATIONS.keys())
    completed_since_excel = 0

    # Process the original PR configuration, preserving the 30 heuristic seeds.
    for inst_pos, dt in enumerate(instances, start=1):
        print()
        print("#" * 118)
        print(
            f"INSTANCIA {inst_pos}/{len(instances)}: "
            f"{dt.name}"
        )
        print("#" * 118)

        for seed_pos, heuristic_seed in enumerate(HEURISTIC_SEEDS):
            offset = seed_pos % len(config_names)
            run_order = (
                config_names[offset:] + config_names[:offset]
            )

            print(
                f"  Heuristic seed {heuristic_seed:6d} "
                f"({seed_pos + 1:02d}/30)"
            )

            for component_name in run_order:
                key = (component_name, dt.name, heuristic_seed)

                old = results.get(key)
                if old is not None and not old.get("error"):
                    if old.get("profile_total_seconds") is None:
                        raise RuntimeError(f"Incomplete profiling checkpoint: {key}")
                    print(
                        f"    {component_name:8s}: JA CONCLUIDA | "
                        f"C={float(old['C']):.8f} | "
                        f"time={float(old['time_s']):.4f}s"
                    )
                    continue

                print(
                    f"    {component_name:8s}: rodando...",
                    end="",
                    flush=True,
                )

                try:
                    row = run_one(
                        dt,
                        component_name,
                        heuristic_seed,
                    )
                    results[key] = row

                    print(
                        f" OK | C={row['C']:.8f} | "
                        f"time={row['time_s']:.4f}s | "
                        f"audit={row['audit_abs_diff']:.2e}"
                    )

                    append_report(row)

                except Exception as exc:
                    row = {
                        "Configuration": component_name,
                        "Instance": dt.name,
                        "N": dt.total_N,
                        "ni": dt.n_i,
                        "nk": dt.n_k,
                        "nj": dt.n_j,
                        "p": dt.p,
                        "alpha": dt.alpha,
                        "instance_seed": dt.seed,
                        "heuristic_seed": heuristic_seed,
                        "C": None,
                        "time_s": None,
                        "wall_time_s": None,
                        "audit_abs_diff": None,
                        "feasible": "NAO",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results[key] = row

                    print(
                        f" ERRO | {type(exc).__name__}: {exc}"
                    )
                    append_error(
                        dt,
                        component_name,
                        heuristic_seed,
                        exc,
                    )
                    traceback.print_exc()

                # Checkpoint after EVERY execution.
                save_checkpoint(results)

                completed_since_excel += 1

                # Update Excel every 30 newly attempted executions.
                # This keeps partial results available without excessive I/O.
                if completed_since_excel >= 30:
                    write_xlsx(instances, results)
                    completed_since_excel = 0

        # Preserve the original workbook and a separate complete profile.
        write_xlsx(instances, results)
        write_all_profiling_outputs(instances, results)

        done_valid = sum(
            1 for row in results.values() if not row.get("error")
        )
        errors_now = sum(
            1 for row in results.values() if row.get("error")
        )

        print(
            f"  -> progresso: {done_valid}/{total_expected} validas | "
            f"{errors_now} erros"
        )

    save_checkpoint(results)
    write_xlsx(instances, results)
    write_all_profiling_outputs(instances, results)

    valid = sum(
        1 for row in results.values() if not row.get("error")
    )
    errors = sum(
        1 for row in results.values() if row.get("error")
    )

    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 110 + "\n")
        f.write("FINAL SUMMARY - ILS[A+S1]+PR FULL PROFILING\n")
        f.write("=" * 110 + "\n")
        f.write(f"Expected runs : {total_expected}\n")
        f.write(f"Valid runs    : {valid}\n")
        f.write(f"Errors        : {errors}\n")
        f.write(f"Excel         : {XLSX_FILE.name}\n")
        f.write(f"Checkpoint    : {CHECKPOINT_CSV.name}\n")

    print()
    print("=" * 118)
    print("FIM")
    print("=" * 118)
    print(f"Execucoes esperadas : {total_expected}")
    print(f"Execucoes validas   : {valid}")
    print(f"Erros               : {errors}")
    print(f"Excel               : {XLSX_FILE}")
    print(f"Checkpoint          : {CHECKPOINT_CSV}")
    print(f"Relatorio           : {REPORT_TXT}")
    print(f"Profiling Excel     : {PROFILE_XLSX}")
    print(f"Profiling CSV       : {PROFILE_CSV}")
    print(f"Manifesto           : {MANIFEST_JSON}")


if __name__ == "__main__":
    main()
