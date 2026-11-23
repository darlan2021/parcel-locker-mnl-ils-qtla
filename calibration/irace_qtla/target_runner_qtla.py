#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

# This file lives in calibration/irace_qtla/.
# The QTLA implementation lives in src/qtla_redesign.py.
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from qtla_redesign import (  # noqa: E402
    QTLA_ENHANCED,
    QTLAParameters,
    load_instance,
    solve_qtla,
)

RUN_LOG = HERE / "run_level_results.csv"
ERROR_LOG = HERE / "target_runner_errors.log"

# Fixed methodological settings for calibration.
OVERALL_TIME_LIMIT = 3600.0
MIP_GAP = 1.0e-6
OA_EPSILON = 1.0e-4

LOG_COLUMNS = [
    "configuration_id",
    "instance_id",
    "irace_seed",
    "qtla_seed",
    "instance",
    "N",
    "ni",
    "nk",
    "nj",
    "p",
    "alpha",
    "instance_seed",
    "maxIterations",
    "subproblemTimeLimit",
    "dampingGamma",
    "trustRadiusFactor",
    "neighborCuts",
    "polishMaxSteps",
    "objective_C",
    "irace_cost",
    "runtime_s",
    "iterations",
    "status",
    "repeated_stop",
    "feasible",
]


@contextmanager
def suppress_native_output():
    """
    Silence stdout/stderr at OS file-descriptor level while QT-LA/Gurobi runs.

    This is necessary because Gurobi may print a license message such as
    'Set parameter LicenseID ...' directly to the process output even when
    model.Params.OutputFlag = 0.

    irace requires targetRunner stdout to contain only the numeric cost.
    After leaving this context, stdout/stderr are restored normally.
    """
    sys.stdout.flush()
    sys.stderr.flush()

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    saved_stdout = os.dup(stdout_fd)
    saved_stderr = os.dup(stderr_fd)

    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stdout_fd)
            os.dup2(devnull.fileno(), stderr_fd)
            try:
                yield
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.dup2(saved_stderr, stderr_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)

    # Arguments supplied by irace targetCmdline.
    parser.add_argument("configuration_id", type=int)
    parser.add_argument("instance_id", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("instance")

    # Tunable QT-LA parameters.
    parser.add_argument(
        "--max-iterations",
        dest="max_iterations",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--subproblem-time-limit",
        dest="subproblem_time_limit",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--damping-gamma",
        dest="damping_gamma",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--trust-radius-factor",
        dest="trust_radius_factor",
        type=float,
        required=True,
    )
    parser.add_argument(
        "--neighbor-cuts",
        dest="neighbor_cuts",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--polish-max-steps",
        dest="polish_max_steps",
        type=int,
        required=True,
    )

    return parser.parse_args()


def normalize_gurobi_seed(seed: int) -> int:
    # Gurobi accepts integer Seed values in [0, 2,000,000,000].
    return int(seed) % 2_000_000_001


def append_csv(row: dict) -> None:
    exists = RUN_LOG.exists() and RUN_LOG.stat().st_size > 0

    with RUN_LOG.open("a", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=LOG_COLUMNS,
            delimiter=";",
        )
        if not exists:
            writer.writeheader()

        writer.writerow(
            {key: row.get(key, "") for key in LOG_COLUMNS}
        )
        fh.flush()
        os.fsync(fh.fileno())


def write_error(message: str) -> None:
    with ERROR_LOG.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")


def main() -> int:
    args = parse_args()

    try:
        instance_path = Path(args.instance)
        if not instance_path.is_absolute():
            instance_path = (Path.cwd() / instance_path).resolve()

        # These operations do not create a Gurobi model and may remain outside
        # the suppression context.
        instance = load_instance(instance_path)
        qtla_seed = normalize_gurobi_seed(args.seed)

        params = QTLAParameters(
            max_iterations=args.max_iterations,
            subproblem_time_limit=args.subproblem_time_limit,
            damping_gamma=args.damping_gamma,
            trust_radius_factor=args.trust_radius_factor,
            neighbor_cuts=args.neighbor_cuts,
            polish_max_steps=args.polish_max_steps,
            mip_gap=MIP_GAP,
            oa_epsilon=OA_EPSILON,
            overall_time_limit=OVERALL_TIME_LIMIT,
        )

        t0 = perf_counter()

        # IMPORTANT:
        # Suppress both Python- and native-level output while Gurobi is
        # initialized and the QT-LA is solved. This prevents the Gurobi
        # LicenseID banner from contaminating irace stdout.
        with suppress_native_output():
            result = solve_qtla(
                instance=instance,
                config=QTLA_ENHANCED,
                parameters=params,
                seed=qtla_seed,
                verbose=False,
            )

        wall = perf_counter() - t0

        if not result.feasible:
            raise RuntimeError("QT-LA returned an infeasible result.")

        objective = float(result.objective)
        if not math.isfinite(objective):
            raise RuntimeError(
                f"Non-finite objective: {objective}"
            )

        # irace MINIMIZES. QT-LA MAXIMIZES C.
        cost = -objective

        # N follows the current generator convention: N = 2*n_i.
        N = 2 * int(instance["ni"])

        append_csv(
            {
                "configuration_id": args.configuration_id,
                "instance_id": args.instance_id,
                "irace_seed": args.seed,
                "qtla_seed": qtla_seed,
                "instance": instance["name"],
                "N": N,
                "ni": instance["ni"],
                "nk": instance["nk"],
                "nj": instance["nj"],
                "p": instance["p"],
                "alpha": instance["alpha"],
                "instance_seed": instance["instance_seed"],
                "maxIterations": args.max_iterations,
                "subproblemTimeLimit": args.subproblem_time_limit,
                "dampingGamma": args.damping_gamma,
                "trustRadiusFactor": args.trust_radius_factor,
                "neighborCuts": args.neighbor_cuts,
                "polishMaxSteps": args.polish_max_steps,
                "objective_C": objective,
                "irace_cost": cost,
                "runtime_s": (
                    result.runtime
                    if math.isfinite(float(result.runtime))
                    else wall
                ),
                "iterations": result.iterations,
                "status": result.status,
                "repeated_stop": int(result.repeated_stop),
                "feasible": int(result.feasible),
            }
        )

        # CRITICAL FOR IRACE:
        # This must be the ONLY line written to stdout.
        print(f"{cost:.15g}", flush=True)
        return 0

    except Exception as exc:
        message = (
            f"configuration_id="
            f"{getattr(args, 'configuration_id', '?')} | "
            f"instance_id="
            f"{getattr(args, 'instance_id', '?')} | "
            f"seed={getattr(args, 'seed', '?')} | "
            f"instance={getattr(args, 'instance', '?')} | "
            f"{type(exc).__name__}: {exc}"
        )
        write_error(message)
        print(message, file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
