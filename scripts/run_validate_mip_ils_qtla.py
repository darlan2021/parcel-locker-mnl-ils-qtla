#!/usr/bin/env python3
from __future__ import annotations

"""
Cross-validation table:
MILP solution value vs. ILS objective evaluator vs. QT-LA objective evaluator.

IMPORTANT:
- This script DOES NOT run ILS or QT-LA as optimization algorithms.
- For each MILP instance with certified optimality, it takes the EXACT SAME
  binary solution returned by the MILP and evaluates that solution with:
    1) the C value stored by the MILP run,
    2) the objective evaluator used inside mnl_ils.py,
    3) the objective evaluator used inside qtla_redesign.py.
- If the MILP did NOT certify optimality, only the MILP incumbent value,
  time, status, and open lockers are reported. ILS/QTLA/difference stay blank,
  exactly as required for the validation table.

Expected project structure:

parcel-locker-mnl-ils-qtla/
├── src/
│   ├── mnl_ils.py
│   └── qtla_redesign.py
├── scripts/
│   └── run_validate_mip_ils_qtla.py  <-- this file
├── instances/
│   └── instances_seed_1_N_60_100_200_400/
│       └── N60_..._s1.npz, N100_..._s1.npz, ...
└── results/
    ├── m_iii/
    │   └── resultados_mip_N60_N100.csv
    └── validation_objective/
        ├── validation_mip_ils_qtla.csv
        └── validation_mip_ils_qtla.txt
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import math
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mnl_ils import ILSConfig, MNLParcelLockerILS
from qtla_redesign import (
    load_instance as load_qtla_instance,
    objective_value as qtla_objective_value,
)


# =============================================================================
# PATHS
# =============================================================================

MIP_DIR = ROOT / "results" / "m_iii"
MIP_CSV = MIP_DIR / "resultados_mip_N60_N100.csv"

INSTANCE_DIR = ROOT / "instances" / "instances_seed_1_N_60_100_200_400"

OUTPUT_DIR = ROOT / "results" / "validation_objective"
OUTPUT_CSV = OUTPUT_DIR / "validation_mip_ils_qtla.csv"
OUTPUT_TXT = OUTPUT_DIR / "validation_mip_ils_qtla.txt"


# =============================================================================
# LIGHTWEIGHT DATA OBJECT FOR THE ILS EVALUATOR
# =============================================================================

@dataclass
class ILSInstanceData:
    name: str
    n_i: int
    n_j: int
    n_k: int
    p: int
    alpha: float
    seed: int
    a: np.ndarray
    b: np.ndarray


def _scalar(npz: Any, key: str, cast):
    value = npz[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_ils_instance(path: Path) -> ILSInstanceData:
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
        raise ValueError(
            f"{path.name}: a.shape={a.shape}, expected {(n_i, n)}."
        )
    if b.shape != (n_i, n):
        raise ValueError(
            f"{path.name}: b.shape={b.shape}, expected {(n_i, n)}."
        )

    return ILSInstanceData(
        name=path.stem,
        n_i=n_i,
        n_j=n_j,
        n_k=n_k,
        p=p,
        alpha=alpha,
        seed=seed,
        a=a,
        b=b,
    )


# =============================================================================
# HELPERS
# =============================================================================

def normalize_headers(row: dict[str, str]) -> dict[str, str]:
    """Remove UTF-8 BOM and surrounding spaces from CSV header names."""
    return {
        str(key).replace("\ufeff", "").strip():
        value
        for key, value in row.items()
    }


def parse_indices(value: str | None) -> list[int]:
    if value is None:
        return []

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return []

    return [int(x) for x in re.findall(r"-?\d+", text)]


def compact_instance_name(
    N: int,
    p: int,
    alpha: float,
) -> str:
    """
    Examples:
        N60, p3, alpha0.5 -> n60p3a05
        N60, p3, alpha1.0 -> n60p3a10
        N60, p3, alpha2.0 -> n60p3a20
    """
    alpha_code = int(round(10.0 * float(alpha)))
    return f"n{int(N)}p{int(p)}a{alpha_code:02d}"


def open_set_text(indices: list[int]) -> str:
    return "{" + ",".join(str(i) for i in indices) + "}"


def find_instance_file(instance_name: str) -> Path:
    exact = INSTANCE_DIR / f"{instance_name}.npz"

    if exact.exists():
        return exact

    matches = list(INSTANCE_DIR.glob(f"{instance_name}*.npz"))

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            f"Instance file not found for {instance_name} in "
            f"{INSTANCE_DIR.resolve()}"
        )

    raise RuntimeError(
        f"More than one instance file matches {instance_name}: "
        + ", ".join(p.name for p in matches)
    )


def is_certified_optimal(row: dict[str, str]) -> bool:
    status = str(row.get("Status", "")).strip().upper()
    certified = str(
        row.get("Optimality_certified", "")
    ).strip().upper()

    return status == "OPTIMAL" and certified == "SIM"


def safe_float(value: str | None) -> float:
    if value is None:
        return math.nan

    text = str(value).strip().replace(",", ".")

    if not text:
        return math.nan

    return float(text)


# =============================================================================
# OBJECTIVE CROSS-VALIDATION
# =============================================================================

def evaluate_with_ils(
    instance_path: Path,
    z: np.ndarray,
) -> float:
    """
    Evaluate z through the ACTUAL objective evaluation path used by mnl_ils.py.

    The private _make_state() method computes:
        num = a @ z
        den = b @ z
        raw = sum(num / den)
    """
    data = load_ils_instance(instance_path)

    cfg = ILSConfig(
        seed=0,
        verbose=False,
    )

    solver = MNLParcelLockerILS(
        data,
        cfg,
    )

    state = solver._make_state(
        np.asarray(z, dtype=np.int8)
    )

    return float(state.raw)


def evaluate_with_qtla(
    instance_path: Path,
    z: np.ndarray,
) -> float:
    """
    Evaluate the SAME z through qtla_redesign.objective_value().
    """
    instance = load_qtla_instance(instance_path)

    return float(
        qtla_objective_value(
            np.asarray(z, dtype=float),
            instance,
        )
    )


def independent_direct_value(
    instance_path: Path,
    z: np.ndarray,
) -> float:
    """
    Independent audit used internally by this validation run.
    It is NOT an extra column in the article table.
    """
    with np.load(instance_path, allow_pickle=False) as data:
        a = np.asarray(data["a"], dtype=float)
        b = np.asarray(data["b"], dtype=float)

    zf = np.asarray(z, dtype=float)

    num = a @ zf
    den = b @ zf

    if np.any(den <= 0.0):
        raise ValueError(
            f"{instance_path.name}: nonpositive denominator "
            "in direct objective audit."
        )

    return float(np.sum(num / den))


# =============================================================================
# MAIN TABLE
# =============================================================================

OUTPUT_FIELDS = [
    "Instance",
    "MILP_C",
    "MILP_Time_s",
    "MILP_Status",
    "Open_lockers_x_0based",
    "ILS_C",
    "QTLA_C",
    "Abs_diff",
]


def main() -> None:
    if not MIP_CSV.exists():
        raise FileNotFoundError(
            f"MILP result file not found:\n{MIP_CSV.resolve()}"
        )

    if not INSTANCE_DIR.exists():
        raise FileNotFoundError(
            f"Instance folder not found:\n{INSTANCE_DIR.resolve()}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_rows: list[dict[str, str]] = []

    with MIP_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as fh:
        # The MILP CSV is semicolon-separated.  Detect the delimiter
        # defensively so the script also works if the file is later exported
        # with commas.
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t,")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ";"

        reader = csv.DictReader(fh, delimiter=delimiter)

        for raw in reader:
            source_rows.append(
                normalize_headers(raw)
            )

    if not source_rows:
        raise RuntimeError(
            f"No rows found in {MIP_CSV.name}."
        )

    table_rows: list[dict[str, object]] = []
    validation_details: list[str] = []

    for row in source_rows:
        full_name = str(row["Instance"]).strip()

        N = int(float(row["N"]))
        p = int(float(row["p"]))
        alpha = safe_float(row["alpha"])

        short_name = compact_instance_name(
            N=N,
            p=p,
            alpha=alpha,
        )

        mip_c = safe_float(row["C"])
        mip_time = safe_float(row["Time_s"])
        mip_status = str(
            row.get("Status", "")
        ).strip()

        combined_open = parse_indices(
            row.get("Combined_open_0based")
        )

        if not combined_open:
            raise RuntimeError(
                f"{full_name}: Combined_open_0based is empty."
            )

        output_row: dict[str, object] = {
            "Instance": short_name,
            "MILP_C": mip_c,
            "MILP_Time_s": mip_time,
            "MILP_Status": mip_status,
            "Open_lockers_x_0based": open_set_text(
                combined_open
            ),
            "ILS_C": "",
            "QTLA_C": "",
            "Abs_diff": "",
        }

        # -------------------------------------------------------------
        # Only certified-optimal MILP rows are cross-evaluated.
        # -------------------------------------------------------------
        if is_certified_optimal(row):
            instance_path = find_instance_file(
                full_name
            )

            ils_data = load_ils_instance(
                instance_path
            )

            n = ils_data.n_k + ils_data.n_j

            z = np.zeros(
                n,
                dtype=np.int8,
            )

            for idx in combined_open:
                if idx < 0 or idx >= n:
                    raise IndexError(
                        f"{full_name}: open index {idx} outside "
                        f"[0, {n - 1}]."
                    )
                z[idx] = 1

            ils_c = evaluate_with_ils(
                instance_path,
                z,
            )

            qtla_c = evaluate_with_qtla(
                instance_path,
                z,
            )

            direct_c = independent_direct_value(
                instance_path,
                z,
            )

            values = [
                float(mip_c),
                float(ils_c),
                float(qtla_c),
            ]

            pairwise_diffs = [
                abs(values[0] - values[1]),
                abs(values[0] - values[2]),
                abs(values[1] - values[2]),
            ]

            abs_diff = max(pairwise_diffs)

            # Strong internal audit: the independent direct formula must
            # agree with both heuristic objective evaluators.
            audit_diff = max(
                abs(direct_c - ils_c),
                abs(direct_c - qtla_c),
            )

            if audit_diff > 1.0e-8:
                raise RuntimeError(
                    f"{full_name}: evaluator audit failed. "
                    f"direct={direct_c:.15g}, "
                    f"ILS={ils_c:.15g}, "
                    f"QTLA={qtla_c:.15g}, "
                    f"max diff={audit_diff:.3e}"
                )

            output_row["ILS_C"] = ils_c
            output_row["QTLA_C"] = qtla_c
            output_row["Abs_diff"] = abs_diff

            validation_details.append(
                f"{short_name}: "
                f"MILP_C={mip_c:.12f} | "
                f"ILS_C={ils_c:.12f} | "
                f"QTLA_C={qtla_c:.12f} | "
                f"direct={direct_c:.12f} | "
                f"Abs_diff={abs_diff:.3e}"
            )

        else:
            validation_details.append(
                f"{short_name}: "
                f"MILP not certified optimal "
                f"(status={mip_status}); "
                f"reported incumbent C={mip_c:.12f}; "
                "ILS/QTLA evaluation intentionally skipped."
            )

        table_rows.append(
            output_row
        )

    # Sort exactly as article presentation:
    # N, p, alpha.
    table_rows.sort(
        key=lambda r: (
            int(
                re.search(
                    r"n(\d+)",
                    str(r["Instance"]),
                ).group(1)
            ),
            int(
                re.search(
                    r"p(\d+)",
                    str(r["Instance"]),
                ).group(1)
            ),
            int(
                re.search(
                    r"a(\d+)",
                    str(r["Instance"]),
                ).group(1)
            ),
        )
    )

    # -----------------------------------------------------------------
    # CSV: Excel-readable article table.
    # -----------------------------------------------------------------
    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=OUTPUT_FIELDS,
            delimiter=";",
        )

        writer.writeheader()
        writer.writerows(table_rows)

    # -----------------------------------------------------------------
    # TXT: audit/report for reproducibility.
    # -----------------------------------------------------------------
    with OUTPUT_TXT.open(
        "w",
        encoding="utf-8",
    ) as fh:
        fh.write("=" * 100 + "\n")
        fh.write(
            "MILP x ILS OBJECTIVE EVALUATOR x QT-LA OBJECTIVE EVALUATOR\n"
        )
        fh.write("=" * 100 + "\n\n")

        fh.write(
            "Rule: ILS/QTLA evaluators are applied only when the MILP "
            "solution has certified optimality.\n"
        )
        fh.write(
            "For non-certified MILP rows, only the best incumbent C "
            "and MILP information are retained.\n\n"
        )

        fh.write(
            "Abs_diff = max{|MILP_C-ILS_C|, |MILP_C-QTLA_C|, "
            "|ILS_C-QTLA_C|}.\n\n"
        )

        for line in validation_details:
            fh.write(line + "\n")

        fh.write("\n" + "=" * 100 + "\n")
        fh.write("ARTICLE TABLE\n")
        fh.write("=" * 100 + "\n")

        header = (
            f"{'Instance':12s} "
            f"{'MILP C':>16s} "
            f"{'Time (s)':>12s} "
            f"{'Status':>12s} "
            f"{'Open lockers x':>34s} "
            f"{'ILS C':>16s} "
            f"{'QTLA C':>16s} "
            f"{'Abs. diff.':>14s}\n"
        )

        fh.write(header)
        fh.write("-" * 140 + "\n")

        for r in table_rows:
            ils_text = (
                ""
                if r["ILS_C"] == ""
                else f"{float(r['ILS_C']):.10f}"
            )

            qtla_text = (
                ""
                if r["QTLA_C"] == ""
                else f"{float(r['QTLA_C']):.10f}"
            )

            diff_text = (
                ""
                if r["Abs_diff"] == ""
                else f"{float(r['Abs_diff']):.3e}"
            )

            fh.write(
                f"{str(r['Instance']):12s} "
                f"{float(r['MILP_C']):16.10f} "
                f"{float(r['MILP_Time_s']):12.3f} "
                f"{str(r['MILP_Status']):>12s} "
                f"{str(r['Open_lockers_x_0based']):>34s} "
                f"{ils_text:>16s} "
                f"{qtla_text:>16s} "
                f"{diff_text:>14s}\n"
            )

    n_opt = sum(
        1
        for r in source_rows
        if is_certified_optimal(r)
    )
    n_not_opt = len(source_rows) - n_opt

    print("=" * 100)
    print("VALIDATION TABLE CREATED")
    print("=" * 100)
    print(f"MILP rows                  : {len(source_rows)}")
    print(f"Certified optimal          : {n_opt}")
    print(f"Not certified              : {n_not_opt}")
    print(f"CSV table                  : {OUTPUT_CSV}")
    print(f"Validation/audit report    : {OUTPUT_TXT}")
    print()
    print(
        "Important: ILS and QT-LA were NOT optimized here. "
        "They only evaluated the exact same MILP solution vector."
    )


if __name__ == "__main__":
    main()
