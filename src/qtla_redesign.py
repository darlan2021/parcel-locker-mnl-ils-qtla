#!/usr/bin/env python3
"""
QT-LA for the POP-station / parcel-locker network redesign model.

Model
-----
Let z = (r, x), where

    r_k = 1 if existing POP station k remains open,
    x_j = 1 if candidate locker j is opened.

The original fractional objective is

    F(z) = sum_i A_i(z) / B_i(z),

with

    A_i(z) = sum_l a[i,l] z_l,
    B_i(z) = sum_l b[i,l] z_l.

The redesign constraints are

    sum_{j in J^C} x_j <= p,
    sum_{k in J^E} r_k >= |J^E| - p.

The first n_k columns of a,b are existing POP stations and the following
n_j columns are candidate lockers, exactly as in the current .npz instances.

QT-LA backbone
----------------
This code preserves the Lin-style backbone used in the previous implementation:

1. Initial binary point is all ones and u^0 = 0.
2. At each major iteration, the current binary configuration is recorded and
   an OA linearization of sqrt(A_i(z)) is added.
3. The quadratic-transform parameter is updated by damping:

       u^n = (1-gamma) u^(n-1) + gamma * sqrt(A(z))/B(z).

4. A sparse M-VI MILP is solved.
5. Repeated-configuration detection is ALWAYS active and is a FIXED stopping
   rule. It is not an ablation component and is not tunable by irace.
6. The other fixed stopping rule is n >= N_max.

Enhancement components
----------------------
The following mechanisms can be switched ON/OFF for ablation:

    trust_region : Hamming trust region around the current feasible solution.
    neighbor_oa  : additional OA cuts at sampled feasible local neighbors.
    local_polish : best-improvement feasible local search using the true F(z).
    no_good      : general binary no-good cuts for processed feasible iterates.

IRACE-ready parameters
----------------------
The code exposes the numerical parameters through command-line arguments:

    --max-iterations
    --subproblem-time-limit
    --damping-gamma
    --trust-radius-factor
    --neighbor-cuts
    --polish-max-steps

The component switches are deliberately separated from numerical parameters so
that irace can calibrate the parameters while the ablation configuration is
kept fixed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB


DEFAULT_SEED = 0
DEFAULT_MAX_ITERATIONS = 40
DEFAULT_SUBPROBLEM_TIME_LIMIT = 5.0
DEFAULT_DAMPING_GAMMA = 0.60
DEFAULT_TRUST_RADIUS_FACTOR = 0.50
DEFAULT_NEIGHBOR_CUTS = 2
DEFAULT_POLISH_MAX_STEPS = 80

MIP_GAP = 1.0e-6
OA_EPSILON = 1.0e-4
IMPROVEMENT_TOLERANCE = 1.0e-12

COMPONENT_NAMES = (
    "trust_region",
    "neighbor_oa",
    "local_polish",
    "no_good",
)


@dataclass(frozen=True)
class QTLAConfig:
    trust_region: bool = False
    neighbor_oa: bool = False
    local_polish: bool = False
    no_good: bool = False

    def active_components(self) -> list[str]:
        return [name for name in COMPONENT_NAMES if getattr(self, name)]


QTLA_BASELINE = QTLAConfig(
    trust_region=False,
    neighbor_oa=False,
    local_polish=False,
    no_good=False,
)

QTLA_ENHANCED = QTLAConfig(
    trust_region=True,
    neighbor_oa=True,
    local_polish=True,
    no_good=True,
)


@dataclass(frozen=True)
class QTLAParameters:
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    subproblem_time_limit: float = DEFAULT_SUBPROBLEM_TIME_LIMIT
    damping_gamma: float = DEFAULT_DAMPING_GAMMA
    trust_radius_factor: float = DEFAULT_TRUST_RADIUS_FACTOR
    neighbor_cuts: int = DEFAULT_NEIGHBOR_CUTS
    polish_max_steps: int = DEFAULT_POLISH_MAX_STEPS
    mip_gap: float = MIP_GAP
    oa_epsilon: float = OA_EPSILON
    overall_time_limit: Optional[float] = None

    def validate(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if self.subproblem_time_limit <= 0.0:
            raise ValueError("subproblem_time_limit must be positive.")
        if not 0.0 < self.damping_gamma <= 1.0:
            raise ValueError("damping_gamma must satisfy 0 < gamma <= 1.")
        if self.trust_radius_factor <= 0.0:
            raise ValueError("trust_radius_factor must be positive.")
        if self.neighbor_cuts < 0:
            raise ValueError("neighbor_cuts must be nonnegative.")
        if self.polish_max_steps < 0:
            raise ValueError("polish_max_steps must be nonnegative.")
        if self.mip_gap < 0.0:
            raise ValueError("mip_gap must be nonnegative.")
        if self.oa_epsilon <= 0.0:
            raise ValueError("oa_epsilon must be positive.")
        if self.overall_time_limit is not None and self.overall_time_limit <= 0.0:
            raise ValueError("overall_time_limit must be positive when supplied.")


@dataclass
class QTLAResult:
    instance: str
    profile: str
    seed: int
    status: str
    objective: float
    best_seen_objective: float
    runtime: float
    iterations: int
    repeated_stop: bool
    feasible: bool
    active_components: list[str]
    existing_open_0based: list[int]
    existing_closed_0based: list[int]
    candidate_open_0based: list[int]
    combined_open_0based: list[int]
    parameters: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _scalar(npz, key: str, cast):
    value = npz[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_instance(filename: str | Path) -> dict:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("This redesign QT-LA expects the current .npz instance format.")

    with np.load(path, allow_pickle=False) as data:
        required = ("n_i", "n_j", "n_k", "p", "a", "b")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path.name}: missing fields in .npz: {missing}")

        ni = _scalar(data, "n_i", int)
        nj = _scalar(data, "n_j", int)
        nk = _scalar(data, "n_k", int)
        p = _scalar(data, "p", int)
        alpha = _scalar(data, "alpha", float) if "alpha" in data else float("nan")
        instance_seed = _scalar(data, "seed", int) if "seed" in data else -1
        a = np.asarray(data["a"], dtype=float)
        b = np.asarray(data["b"], dtype=float)

    n = nk + nj
    if a.shape != (ni, n):
        raise ValueError(f"{path.name}: a.shape={a.shape}, expected {(ni, n)}.")
    if b.shape != (ni, n):
        raise ValueError(f"{path.name}: b.shape={b.shape}, expected {(ni, n)}.")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError(f"{path.name}: a must be finite and nonnegative.")
    if not np.all(np.isfinite(b)) or np.any(b <= 0.0):
        raise ValueError(f"{path.name}: b must be finite and strictly positive.")
    if p < 0 or p > nj:
        raise ValueError(f"Invalid p={p} for nj={nj}.")

    return {
        "name": path.stem,
        "path": str(path),
        "ni": ni,
        "nj": nj,
        "nk": nk,
        "n": n,
        "p": p,
        "alpha": alpha,
        "instance_seed": instance_seed,
        "a": a,
        "b": b,
    }


def solution_key(z: np.ndarray) -> tuple[int, ...]:
    return tuple((np.asarray(z) > 0.5).astype(np.int8).tolist())


def split_counts(z: np.ndarray, instance: dict) -> tuple[int, int]:
    z_bin = np.asarray(z) > 0.5
    nk = instance["nk"]
    return int(np.sum(z_bin[:nk])), int(np.sum(z_bin[nk:]))


def is_feasible(z: np.ndarray, instance: dict) -> bool:
    existing_open, candidate_open = split_counts(z, instance)
    nk = instance["nk"]
    p = instance["p"]
    return (
        candidate_open <= p
        and existing_open >= nk - p
        and existing_open + candidate_open >= 1
    )


def objective_value(z: np.ndarray, instance: dict) -> float:
    zf = np.asarray(z, dtype=float)
    numerator = instance["a"] @ zf
    denominator = instance["b"] @ zf
    if np.any(denominator <= 0.0):
        raise ValueError("A nonpositive denominator was found in the evaluated solution.")
    return float(np.sum(numerator / denominator))


def qt_target_parameters(z: np.ndarray, instance: dict) -> np.ndarray:
    zf = np.asarray(z, dtype=float)
    numerator = instance["a"] @ zf
    denominator = instance["b"] @ zf
    if np.any(denominator <= 0.0):
        raise ValueError("A nonpositive denominator was found while updating u.")
    return np.sqrt(np.maximum(numerator, 0.0)) / denominator


def oa_linearization(
    z_reference: np.ndarray,
    instance: dict,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    z_ref = np.asarray(z_reference, dtype=float)
    a = instance["a"]
    values = np.maximum(a @ z_ref, 0.0)
    roots = np.sqrt(values)
    denominators = 2.0 * roots + float(epsilon)
    coefficients = a / denominators[:, None]
    intercept = roots - coefficients @ z_ref
    return intercept, coefficients


def add_oa_point(linearizations: dict, z_reference: np.ndarray, instance: dict, epsilon: float) -> None:
    key = solution_key(z_reference)
    if key not in linearizations:
        linearizations[key] = oa_linearization(z_reference, instance, epsilon)


def _single_flip_is_feasible(index: int, z: np.ndarray, instance: dict) -> bool:
    nk = instance["nk"]
    p = instance["p"]
    existing_open, candidate_open = split_counts(z, instance)
    currently_open = z[index] > 0.5

    if index < nk:
        new_existing = existing_open + (-1 if currently_open else 1)
        new_candidate = candidate_open
    else:
        new_existing = existing_open
        new_candidate = candidate_open + (-1 if currently_open else 1)

    return (
        new_candidate <= p
        and new_existing >= nk - p
        and new_existing + new_candidate >= 1
    )


def local_neighbor_moves(z_reference: np.ndarray, instance: dict) -> list[tuple[int, ...]]:
    z = np.asarray(z_reference, dtype=float)
    n = instance["n"]
    nk = instance["nk"]
    moves: list[tuple[int, ...]] = []

    for index in range(n):
        if _single_flip_is_feasible(index, z, instance):
            moves.append((index,))

    if is_feasible(z, instance):
        for start, end in ((0, nk), (nk, n)):
            opened = np.flatnonzero(z[start:end] > 0.5) + start
            closed = np.flatnonzero(z[start:end] < 0.5) + start
            for out_index in opened:
                for in_index in closed:
                    moves.append((int(out_index), int(in_index)))

    return moves


def apply_move(z: np.ndarray, move: tuple[int, ...]) -> np.ndarray:
    out = np.asarray(z, dtype=float).copy()
    for index in move:
        out[index] = 1.0 - out[index]
    return out


def selected_feasible_neighbors(
    z_reference: np.ndarray,
    instance: dict,
    number_of_neighbors: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if number_of_neighbors <= 0 or not is_feasible(z_reference, instance):
        return []
    moves = local_neighbor_moves(z_reference, instance)
    if not moves:
        return []
    number = min(int(number_of_neighbors), len(moves))
    chosen = rng.choice(len(moves), size=number, replace=False)
    return [apply_move(z_reference, moves[int(pos)]) for pos in np.atleast_1d(chosen)]


def hamming_expression(z_variables, center: np.ndarray):
    center_bin = np.asarray(center) > 0.5
    ones = np.flatnonzero(center_bin)
    zeros = np.flatnonzero(~center_bin)
    return (
        gp.quicksum(1.0 - z_variables[int(j)] for j in ones)
        + gp.quicksum(z_variables[int(j)] for j in zeros)
    )


def add_general_no_good(model: gp.Model, z_variables, solution: np.ndarray, name: str) -> None:
    sol = np.asarray(solution) > 0.5
    ones = np.flatnonzero(sol)
    zeros = np.flatnonzero(~sol)
    model.addConstr(
        gp.quicksum(1.0 - z_variables[int(j)] for j in ones)
        + gp.quicksum(z_variables[int(j)] for j in zeros)
        >= 1,
        name=name,
    )


def best_improvement_local_search(instance: dict, initial_z: np.ndarray, max_steps: int) -> np.ndarray:
    z = np.asarray(initial_z, dtype=float).copy()
    if not is_feasible(z, instance):
        raise ValueError("local_polish requires a feasible initial solution.")
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
        best_move: Optional[tuple[int, ...]] = None

        for index in range(n):
            if not _single_flip_is_feasible(index, z, instance):
                continue
            sign = -1.0 if z[index] > 0.5 else 1.0
            new_den = current_den + sign * b[:, index]
            if np.any(new_den <= 0.0):
                continue
            new_num = current_num + sign * a[:, index]
            value = float(np.sum(new_num / new_den))
            if value > best_value + IMPROVEMENT_TOLERANCE:
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
                values[valid] = np.sum(cand_num[:, valid] / cand_den[:, valid], axis=0)
                pos = int(np.argmax(values))
                value = float(values[pos])
                if value > best_value + IMPROVEMENT_TOLERANCE:
                    best_value = value
                    best_move = (int(out_index), int(closed[pos]))

        if best_move is None:
            break
        z = apply_move(z, best_move)

    return z


def solve_surrogate(
    instance: dict,
    u: np.ndarray,
    z_center: np.ndarray,
    trust_region_radius: int,
    linearizations: dict,
    no_good_solutions: list[np.ndarray],
    config: QTLAConfig,
    seed: int,
    iteration: int,
    time_limit: float,
    mip_gap: float,
) -> Optional[np.ndarray]:
    ni = instance["ni"]
    nk = instance["nk"]
    nj = instance["nj"]
    n = instance["n"]
    p = instance["p"]
    b = instance["b"]

    model = gp.Model(f"QTLA_redesign_iteration_{iteration}")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.MIPGap = float(mip_gap)
    model.Params.TimeLimit = float(time_limit)
    model.Params.Seed = int(seed)

    z = model.addVars(n, vtype=GRB.BINARY, name="z")
    beta = model.addVars(ni, lb=0.0, vtype=GRB.CONTINUOUS, name="beta")

    model.addConstr(
        gp.quicksum(z[nk + j] for j in range(nj)) <= p,
        name="candidate_open_limit",
    )
    model.addConstr(
        gp.quicksum(z[k] for k in range(nk)) >= nk - p,
        name="existing_keep_minimum",
    )
    model.addConstr(
        gp.quicksum(z[l] for l in range(n)) >= 1,
        name="at_least_one_facility",
    )

    if config.trust_region and is_feasible(z_center, instance):
        model.addConstr(
            hamming_expression(z, z_center) <= trust_region_radius,
            name="trust_region",
        )

    if config.no_good:
        for idx, old_solution in enumerate(no_good_solutions):
            add_general_no_good(model, z, old_solution, name=f"no_good_{idx}")

    for cut_number, (intercept, coefficients) in enumerate(linearizations.values()):
        for i in range(ni):
            model.addConstr(
                beta[i]
                <= float(intercept[i])
                + gp.quicksum(float(coefficients[i, l]) * z[l] for l in range(n)),
                name=f"oa_{cut_number}_{i}",
            )

    objective_terms = []
    for i in range(ni):
        denominator = gp.quicksum(float(b[i, l]) * z[l] for l in range(n))
        objective_terms.append(
            2.0 * float(u[i]) * beta[i]
            - float(u[i] ** 2) * denominator
        )

    model.setObjective(gp.quicksum(objective_terms), GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        return None

    return np.array([1.0 if z[l].X > 0.5 else 0.0 for l in range(n)], dtype=float)


def solve_qtla(
    instance: dict,
    config: QTLAConfig,
    parameters: QTLAParameters,
    seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> QTLAResult:
    parameters.validate()

    ni = instance["ni"]
    nk = instance["nk"]
    n = instance["n"]
    p = instance["p"]
    rng = np.random.default_rng(int(seed))

    trust_region_radius = max(
        1,
        int(math.ceil(parameters.trust_radius_factor * max(1, p))),
    )
    trust_region_radius = min(trust_region_radius, n)

    start_time = perf_counter()

    # Lin-style Step 0: all ones, u^0=0. Initial point need not be feasible.
    z_current = np.ones(n, dtype=float)
    u_previous = np.zeros(ni, dtype=float)

    linearizations: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray]] = {}

    # Fixed repeated-configuration stopping set. Always active.
    recorded_iterate_keys: set[tuple[int, ...]] = set()

    # Separate structure used only by the optional no-good enhancement.
    no_good_solutions: list[np.ndarray] = []
    no_good_keys: set[tuple[int, ...]] = set()

    status = "MAXIMUM ITERATIONS"
    iterations_completed = 0
    repeated_stop = False
    best_seen_objective = -np.inf
    best_seen_z: Optional[np.ndarray] = None

    for iteration in range(1, parameters.max_iterations + 1):
        elapsed = perf_counter() - start_time
        if parameters.overall_time_limit is not None and elapsed >= parameters.overall_time_limit:
            status = "OVERALL TIME LIMIT"
            break

        current_key = solution_key(z_current)
        recorded_iterate_keys.add(current_key)

        add_oa_point(
            linearizations,
            z_current,
            instance,
            parameters.oa_epsilon,
        )

        if is_feasible(z_current, instance):
            current_value = objective_value(z_current, instance)
            if current_value > best_seen_objective:
                best_seen_objective = current_value
                best_seen_z = z_current.copy()

        if config.neighbor_oa and is_feasible(z_current, instance):
            neighbors = selected_feasible_neighbors(
                z_current,
                instance,
                parameters.neighbor_cuts,
                rng,
            )
            for neighbor in neighbors:
                add_oa_point(
                    linearizations,
                    neighbor,
                    instance,
                    parameters.oa_epsilon,
                )

        if config.no_good and is_feasible(z_current, instance) and current_key not in no_good_keys:
            no_good_solutions.append(z_current.copy())
            no_good_keys.add(current_key)

        u_target = qt_target_parameters(z_current, instance)
        u_current = (
            (1.0 - parameters.damping_gamma) * u_previous
            + parameters.damping_gamma * u_target
        )

        surrogate_limit = parameters.subproblem_time_limit
        if parameters.overall_time_limit is not None:
            remaining = parameters.overall_time_limit - (perf_counter() - start_time)
            if remaining <= 0.0:
                status = "OVERALL TIME LIMIT"
                break
            surrogate_limit = min(surrogate_limit, remaining)

        z_surrogate = solve_surrogate(
            instance=instance,
            u=u_current,
            z_center=z_current,
            trust_region_radius=trust_region_radius,
            linearizations=linearizations,
            no_good_solutions=no_good_solutions,
            config=config,
            seed=seed,
            iteration=iteration,
            time_limit=surrogate_limit,
            mip_gap=parameters.mip_gap,
        )

        if z_surrogate is None:
            status = "SURROGATE NOT SOLVED"
            break

        if not is_feasible(z_surrogate, instance):
            raise RuntimeError("M-VI returned an infeasible redesign solution.")

        if config.local_polish:
            z_next = best_improvement_local_search(
                instance,
                z_surrogate,
                parameters.polish_max_steps,
            )
        else:
            z_next = z_surrogate

        if not is_feasible(z_next, instance):
            raise RuntimeError("Local polishing returned an infeasible redesign solution.")

        next_value = objective_value(z_next, instance)
        if next_value > best_seen_objective:
            best_seen_objective = next_value
            best_seen_z = z_next.copy()

        # FIXED stopping rule from QT-LA backbone: repetition always stops.
        next_key = solution_key(z_next)
        repeated = next_key in recorded_iterate_keys

        z_current = z_next
        u_previous = u_current
        iterations_completed = iteration

        if verbose:
            print(
                f"iter={iteration:3d} "
                f"C={next_value:.10f} "
                f"OA={len(linearizations):4d} "
                f"TR={trust_region_radius:3d} "
                f"repeat={int(repeated)}"
            )

        if repeated:
            status = "REPEATED CONFIGURATION"
            repeated_stop = True
            break

        if iteration >= parameters.max_iterations:
            status = "MAXIMUM ITERATIONS"
            break

    runtime = perf_counter() - start_time

    if not is_feasible(z_current, instance):
        raise RuntimeError("QT-LA terminated before generating a feasible redesign solution.")

    final_objective = objective_value(z_current, instance)
    if best_seen_z is None:
        best_seen_z = z_current.copy()
        best_seen_objective = final_objective

    z_bin = (z_current > 0.5).astype(np.int8)
    existing_open = np.flatnonzero(z_bin[:nk] == 1).astype(int).tolist()
    existing_closed = np.flatnonzero(z_bin[:nk] == 0).astype(int).tolist()
    candidate_open = np.flatnonzero(z_bin[nk:] == 1).astype(int).tolist()
    combined_open = np.flatnonzero(z_bin == 1).astype(int).tolist()

    if config == QTLA_BASELINE:
        profile = "QTLA-B"
    elif config == QTLA_ENHANCED:
        profile = "QTLA-E"
    else:
        profile = "QTLA-CUSTOM"

    return QTLAResult(
        instance=instance["name"],
        profile=profile,
        seed=int(seed),
        status=status,
        objective=float(final_objective),
        best_seen_objective=float(best_seen_objective),
        runtime=float(runtime),
        iterations=int(iterations_completed),
        repeated_stop=bool(repeated_stop),
        feasible=True,
        active_components=config.active_components(),
        existing_open_0based=existing_open,
        existing_closed_0based=existing_closed,
        candidate_open_0based=candidate_open,
        combined_open_0based=combined_open,
        parameters={
            **asdict(parameters),
            "trust_region_radius_effective": trust_region_radius,
        },
    )


def resolve_config(mode: str, enable: list[str], disable: list[str]) -> QTLAConfig:
    overlap = set(enable) & set(disable)
    if overlap:
        raise ValueError(
            "Components cannot be both enabled and disabled: " + ", ".join(sorted(overlap))
        )

    config = QTLA_BASELINE if mode == "baseline" else QTLA_ENHANCED
    for name in enable:
        config = replace(config, **{name: True})
    for name in disable:
        config = replace(config, **{name: False})
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "QT-LA for the POP-station / locker redesign model, with ablation switches "
            "and irace-ready numerical parameters."
        )
    )

    parser.add_argument("instance", help="Path to one current .npz instance.")
    parser.add_argument("--mode", choices=("baseline", "enhanced"), default="enhanced")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--enable", action="append", choices=COMPONENT_NAMES, default=[])
    parser.add_argument("--disable", action="append", choices=COMPONENT_NAMES, default=[])

    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument(
        "--subproblem-time-limit",
        type=float,
        default=DEFAULT_SUBPROBLEM_TIME_LIMIT,
    )
    parser.add_argument("--damping-gamma", type=float, default=DEFAULT_DAMPING_GAMMA)
    parser.add_argument(
        "--trust-radius-factor",
        type=float,
        default=DEFAULT_TRUST_RADIUS_FACTOR,
    )
    parser.add_argument("--neighbor-cuts", type=int, default=DEFAULT_NEIGHBOR_CUTS)
    parser.add_argument(
        "--polish-max-steps",
        type=int,
        default=DEFAULT_POLISH_MAX_STEPS,
    )
    parser.add_argument("--overall-time-limit", type=float, default=None)
    parser.add_argument("--machine-readable", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def print_human_result(result: QTLAResult) -> None:
    print("=" * 84)
    print("QT-LA REDESIGN RESULTS")
    print("=" * 84)
    print(f"Instance              : {result.instance}")
    print(f"Profile               : {result.profile}")
    print(f"Seed                  : {result.seed}")
    print(
        "Components ON         : "
        + (",".join(result.active_components) if result.active_components else "NONE (QTLA-B)")
    )
    print(f"Status                : {result.status}")
    print(f"Repeated-stop         : {result.repeated_stop}")
    print(f"Objective C           : {result.objective:.12f}")
    print(f"Best seen C (diag.)   : {result.best_seen_objective:.12f}")
    print(f"Iterations            : {result.iterations}")
    print(f"Runtime (s)           : {result.runtime:.6f}")
    print(f"Existing open 0-based : {result.existing_open_0based}")
    print(f"Existing closed 0-base: {result.existing_closed_0based}")
    print(f"Candidate open 0-based: {result.candidate_open_0based}")
    print(f"Parameters            : {result.parameters}")
    print("=" * 84)
    print(
        "RESULT "
        f"C={result.objective:.12f} "
        f"TIME={result.runtime:.6f} "
        f"ITER={result.iterations} "
        f"FEASIBLE={int(result.feasible)} "
        f"STATUS={result.status.replace(' ', '_')}"
    )


def main() -> int:
    args = parse_args()

    try:
        config = resolve_config(args.mode, args.enable, args.disable)
        params = QTLAParameters(
            max_iterations=args.max_iterations,
            subproblem_time_limit=args.subproblem_time_limit,
            damping_gamma=args.damping_gamma,
            trust_radius_factor=args.trust_radius_factor,
            neighbor_cuts=args.neighbor_cuts,
            polish_max_steps=args.polish_max_steps,
            overall_time_limit=args.overall_time_limit,
        )
        instance = load_instance(args.instance)
        result = solve_qtla(
            instance=instance,
            config=config,
            parameters=params,
            seed=args.seed,
            verbose=(not args.quiet and not args.machine_readable),
        )

        if args.machine_readable:
            print(json.dumps(result.to_dict(), separators=(",", ":")))
        else:
            print_human_result(result)
        return 0

    except FileNotFoundError as error:
        print(f"error: file not found: {error}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError, gp.GurobiError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
