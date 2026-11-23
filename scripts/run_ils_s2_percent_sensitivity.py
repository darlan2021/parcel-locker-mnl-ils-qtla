from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from xml.sax.saxutils import escape as xml_escape
import csv
import math
import sys
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from mnl_ils import ILSConfig, MNLParcelLockerILS
from mnl_ils_s2_percent import MNLParcelLockerILS_S2Percent


# =============================================================================
# PATHS
# =============================================================================

INSTANCE_DIR = ROOT / "instances" / "instances_seed_1_N_60_100_200_400"

RESULT_DIR = ROOT / "results" / "ils" / "ils_results_s2_percent_sensitivity"
CHECKPOINT_CSV = RESULT_DIR / "checkpoint_s2_sensitivity.csv"
REPORT_TXT = RESULT_DIR / "relatorio_s2_sensitivity.txt"
XLSX_FILE = RESULT_DIR / "resultados_s2_sensitivity.xlsx"


# =============================================================================
# EXPERIMENT
# =============================================================================
#
# Only the twelve explicitly selected .npz instances are executed.
#
# Four families, each with baseline / 5% / 10% / 20% / 30%, PR OFF:
#   S2, S1S2, AS2, AS1S2
#
# Five heuristic seeds: 1, 2, 3, 4, 5
#
# Parameters reproduce the configuration currently used in heur.py.
# Parameters not explicitly listed remain at ILSConfig defaults.
#
# Fixed selection: 12 instances x 20 configurations x 5 seeds = 1200 runs
# =============================================================================

HEURISTIC_SEEDS = list(range(1, 6))

# Fixed 12-instance sample: one instance seed, four sizes, three lambdas.
INSTANCE_NAMES = [
    f"N{N}_I{ni}_K{nk}_J{nj}_P{pp}_a{a}_s1"
    for N, ni, nk, nj, pp in [
        (60, 30, 15, 15, 6),
        (100, 50, 25, 25, 10),
        (200, 100, 50, 50, 20),
        (400, 200, 100, 100, 30),
    ]
    for a in ("0p5", "1p0", "2p0")
]

# All flags, sample size and other parameters are identical within each group.
# ORIGINAL calls the existing unchanged class. Percent variants override ONLY
# the search-local S2 scorer; S2 perturbation remains unchanged in both classes.
CONFIGURATIONS = {}
for family, ena, ens1 in [
    ("S2", False, False),
    ("S1S2", False, True),
    ("AS2", True, False),
    ("AS1S2", True, True),
]:
    for suffix in ("", "_05", "_10", "_20", "_30"):
        CONFIGURATIONS[family + suffix] = {
            "enable_A": ena,
            "enable_S1": ens1,
            "enable_S2": True,
            "enable_path_relink": False,
        }

CUSTOMER_FRACTIONS = {"": None, "_05": 0.05, "_10": 0.10,
                      "_20": 0.20, "_30": 0.30}

# Explicit fixed value: original ILSConfig default is 350.
S2_SAMPLE_SIZE = 350

# Same explicit settings used in heur.py supplied by the advisor.
GRASP_STARTS = 8
MAX_ITERATIONS = 10
TAU_TABU = 7
MAX_CARDINALITY_VIOLATION = 2
INITIAL_PERTURBATION_STRENGTH = 2
MAX_PERTURBATION_STRENGTH = 10
ELITE_SIZE = 8
PATH_RELINK_INTERVAL = 10  # kept for fidelity; PR is disabled here.

# None = no time limit, which preserves the current heur.py behavior.
ILS_TIME_LIMIT = None

# Suppress internal ILS progress printing in the 1200-run batch.
# This affects only terminal output, not the algorithm.
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


def load_all_instances() -> list[InstanceData]:
    if not INSTANCE_DIR.exists():
        raise FileNotFoundError(
            f"Folder not found: {INSTANCE_DIR}\n"
            "Expected the development instances under the project instances/ directory."
        )

    missing = [name for name in INSTANCE_NAMES if not (INSTANCE_DIR / (name + ".npz")).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing selected .npz files (no execution started):\n  " +
            "\n  ".join(missing) +
            f"\nExpected folder: {INSTANCE_DIR}"
        )
    instances = [load_instance(INSTANCE_DIR / (name + ".npz")) for name in INSTANCE_NAMES]
    expected = {(60,6): (30,15,15), (100,10):(50,25,25),
                (200,20):(100,50,50), (400,30):(200,100,100)}
    for dt in instances:
        assert (dt.n_i,dt.n_k,dt.n_j) == expected[(dt.total_N,dt.p)], dt.name
        assert dt.seed == 1 and dt.alpha in (0.5,1.0,2.0), dt.name
    return instances


# =============================================================================
# ILS
# =============================================================================

def make_config(component_name: str, heuristic_seed: int) -> ILSConfig:
    flags = CONFIGURATIONS[component_name]

    return ILSConfig(
        grasp_starts=GRASP_STARTS,
        s2_sample_size=S2_SAMPLE_SIZE,
        max_iterations=MAX_ITERATIONS,
        tau_tabu=TAU_TABU,
        max_cardinality_violation=MAX_CARDINALITY_VIOLATION,
        initial_perturbation_strength=INITIAL_PERTURBATION_STRENGTH,
        max_perturbation_strength=MAX_PERTURBATION_STRENGTH,
        elite_size=ELITE_SIZE,
        path_relink_interval=PATH_RELINK_INTERVAL,
        enable_A=flags["enable_A"],
        enable_S1=flags["enable_S1"],
        enable_S2=flags["enable_S2"],
        enable_path_relink=False,
        time_limit=ILS_TIME_LIMIT,
        seed=heuristic_seed,
        verbose=ILS_VERBOSE,
    )


def run_one(
    dt: InstanceData,
    component_name: str,
    heuristic_seed: int,
) -> dict[str, Any]:
    cfg = make_config(component_name, heuristic_seed)
    suffix = next((s for s in ("_05", "_10", "_20", "_30") if component_name.endswith(s)), "")
    fraction = CUSTOMER_FRACTIONS[suffix]
    solver = (MNLParcelLockerILS(dt, cfg) if fraction is None
              else MNLParcelLockerILS_S2Percent(dt, cfg, customer_fraction=fraction))

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
    if audit_diff > 1e-7 * max(1.0, abs(direct_C)):
        raise AssertionError(f"Objective audit failed: {audit_diff}")
    profile = result.profile.to_dict()
    vc = max(0, int(np.sum(y[dt.n_k:])) - dt.p)
    ve = max(0, (dt.n_k - dt.p) - int(np.sum(y[:dt.n_k])))
    if vc or ve:
        raise AssertionError(f"Final cardinalities invalid: {vc=}, {ve=}")

    return {
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
        "C": direct_C,
        "time_s": float(result.elapsed_seconds),
        "wall_time_s": float(wall_time),
        "audit_abs_diff": audit_diff,
        "feasible": "SIM",
        "S2_proxy_evals": getattr(solver, "s2_proxy_evaluations", 0),
        "S2_full_evals": getattr(solver, "s2_full_validations", 0),
        "S2_proxy_rejected": getattr(solver, "s2_proxy_best_rejected", 0),
        "S2_accepted": int(result.profile.S2_accepted),
        "S2_selected_customers": getattr(solver, "s2_customers_last", dt.n_i),
        **profile,
        "error": "",
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
    "S2_proxy_evals",
    "S2_full_evals",
    "S2_proxy_rejected",
    "S2_accepted",
    "S2_selected_customers",
    "total_seconds", "grasp_seconds", "local_search_seconds",
    "perturbation_seconds", "path_relink_seconds",
    "A_seconds", "S1_seconds", "S2_seconds", "repair_seconds",
    "grasp_calls", "local_search_calls", "perturbation_calls",
    "path_relink_calls", "repair_calls", "A_calls", "S1_calls", "S2_calls",
    "grasp_move_evaluations", "grasp_steps", "A_move_evaluations",
    "S1_move_evaluations", "S2_move_evaluations",
    "A_accepted", "S1_accepted", "repair_moves",
    "perturb_A_moves", "perturb_S1_moves", "perturb_S2_moves",
    "PR_move_evaluations", "PR_steps", "PR_A_steps", "PR_S1_steps", "PR_S2_steps",
    "other_seconds", "local_search_other_seconds",
    "error",
]


PROFILE_INT_COLUMNS = set(CHECKPOINT_COLUMNS[CHECKPOINT_COLUMNS.index("S2_proxy_evals"):]) - {
    "error", "total_seconds", "grasp_seconds", "local_search_seconds",
    "perturbation_seconds", "path_relink_seconds", "A_seconds",
    "S1_seconds", "S2_seconds", "repair_seconds",
    "other_seconds", "local_search_other_seconds",
}


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
            for k in CHECKPOINT_COLUMNS:
                if k in ("C", "time_s", "wall_time_s", "audit_abs_diff") or k.endswith("_seconds"):
                    row[k] = float(row[k]) if row.get(k) not in ("", None) else None
                elif k in PROFILE_INT_COLUMNS:
                    row[k] = int(row.get(k) or 0)

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

    with REPORT_TXT.open("w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("S2 PERCENT SENSITIVITY - 12 INSTANCES X 20 CONFIGS X 5 SEEDS\n")
        f.write("=" * 100 + "\n")
        f.write(f"Instances              : {len(instances)}\n")
        f.write(f"Configurations         : {', '.join(CONFIGURATIONS)}\n")
        f.write(f"Heuristic seeds        : {HEURISTIC_SEEDS}\n")
        f.write("Path Relinking         : OFF in all configurations\n")
        f.write("Fractions             : Original, 5%, 10%, 20%, 30%\n")
        f.write("Sample S2 moves        : 350 draws/call for all configurations\n")
        f.write("Acceptance             : full objective check for best proxy move\n")
        f.write(f"GRASP starts           : {GRASP_STARTS}\n")
        f.write(f"ILS max iterations     : {MAX_ITERATIONS}\n")
        f.write(f"Tabu tenure            : {TAU_TABU}\n")
        f.write(
            f"Max cardinality viol.  : {MAX_CARDINALITY_VIOLATION}\n"
        )
        f.write(
            f"Perturbation strength  : {INITIAL_PERTURBATION_STRENGTH}"
            f"..{MAX_PERTURBATION_STRENGTH}\n"
        )
        f.write(f"Elite size             : {ELITE_SIZE}\n")
        f.write(f"ILS time limit         : {ILS_TIME_LIMIT}\n")
        f.write("VND seconds includes A/S1/S2/repair: do NOT double count.\n")
        f.write("\nINSTANCES\n")
        for dt in instances:
            f.write(
                f"  {dt.name} | N={dt.total_N} | p={dt.p} | "
                f"alpha={dt.alpha} | instance_seed={dt.seed}\n"
            )
        f.write("\n" + "=" * 100 + "\n\n")


def append_report(row: dict[str, Any]) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write(
            f"{row['Configuration']:6s} | "
            f"{row['Instance']} | "
            f"seed={int(row['heuristic_seed']):2d} | "
            f"C={float(row['C']):.10f} | "
            f"time={float(row['time_s']):.6f}s | "
            f"audit={float(row['audit_abs_diff']):.3e} | "
            f"proxy={row['S2_proxy_evals']} | "
            f"full={row['S2_full_evals']} | "
            f"S2_accepted={row['S2_accepted']}\n"
        )


def append_error(
    dt: InstanceData,
    component_name: str,
    heuristic_seed: int,
    exc: Exception,
) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write(
            f"ERROR | {component_name:6s} | {dt.name} | "
            f"seed={heuristic_seed:2d} | "
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
      <pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/>
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
 <dc:title>ILS Component Analysis Results</dc:title>
 <dc:creator>ILS batch runner</dc:creator>
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
# MAIN
# =============================================================================

def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    instances = load_all_instances()
    initialize_report(instances)

    results = load_checkpoint()

    total_expected = (
        len(instances) * len(CONFIGURATIONS) * len(HEURISTIC_SEEDS)
    )
    assert len(instances) == 12 and len(CONFIGURATIONS) == 20
    assert total_expected == 1200
    completed_before = sum(
        1 for row in results.values() if not row.get("error")
    )

    print("=" * 110)
    print("ILS - S2 SENSIBILIDADE DE PERCENTUAIS - 1200 EXECUCOES")
    print("=" * 110)
    print(f"Instancias encontradas : {len(instances)}")
    print(f"Configuracoes          : {', '.join(CONFIGURATIONS)}")
    print(f"Seeds heuristica       : {HEURISTIC_SEEDS}")
    print("Path Relinking         : DESLIGADO em todas")
    print(f"Execucoes esperadas    : {total_expected}")
    print(f"Ja concluidas          : {completed_before}")
    print(f"Resultados             : {RESULT_DIR}")
    print()

    for component_name in CONFIGURATIONS:
        print()
        print("#" * 110)
        print(f"CONFIGURACAO: {component_name}")
        print("#" * 110)

        for inst_pos, dt in enumerate(instances, start=1):
            print()
            print(
                f"[{component_name}] Instancia {inst_pos}/{len(instances)}: "
                f"{dt.name}"
            )

            for heuristic_seed in HEURISTIC_SEEDS:
                key = (component_name, dt.name, heuristic_seed)

                old = results.get(key)
                if old is not None and not old.get("error"):
                    print(
                        f"  seed {heuristic_seed:2d}: JA CONCLUIDA | "
                        f"C={float(old['C']):.8f} | "
                        f"time={float(old['time_s']):.4f}s"
                    )
                    continue

                print(
                    f"  seed {heuristic_seed:2d}: rodando...",
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
                        "S2_proxy_evals": 0,
                        "S2_full_evals": 0,
                        "S2_proxy_rejected": 0,
                        "S2_accepted": 0,
                        "S2_selected_customers": 0,
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

                # Checkpoint after EVERY seed.
                save_checkpoint(results)

            # Excel after EVERY complete 5-seed instance/config block.
            write_xlsx(instances, results)

            done_valid = sum(
                1 for row in results.values() if not row.get("error")
            )
            print(
                f"  -> progresso valido: {done_valid}/{total_expected}"
            )

    save_checkpoint(results)
    write_xlsx(instances, results)

    valid = sum(
        1 for row in results.values() if not row.get("error")
    )
    errors = sum(
        1 for row in results.values() if row.get("error")
    )

    with REPORT_TXT.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 100 + "\n")
        f.write("FINAL SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Expected runs : {total_expected}\n")
        f.write(f"Valid runs    : {valid}\n")
        f.write(f"Errors        : {errors}\n")
        f.write(f"Excel         : {XLSX_FILE.name}\n")
        f.write(f"Checkpoint    : {CHECKPOINT_CSV.name}\n")

    print()
    print("=" * 110)
    print("FIM")
    print("=" * 110)
    print(f"Execucoes esperadas : {total_expected}")
    print(f"Execucoes validas   : {valid}")
    print(f"Erros               : {errors}")
    print(f"Excel               : {XLSX_FILE}")
    print(f"Checkpoint          : {CHECKPOINT_CSV}")
    print(f"Relatorio           : {REPORT_TXT}")


if __name__ == "__main__":
    main()
