#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QTLA[TR+LP+NG] - Official LARGE stress experiment, 30 calibrated seeds.

Project layout:
    src/qtla_redesign.py
    instances/instances_seed_21_22_23_N_900_1200_1500/ (54 .npz)
    scripts/ (this runner)
    results/qtla_stress_large_900_1200_1500/ (outputs)

No overall time limit. The calibrated 2-second limit for each Gurobi MILP
subproblem is retained (this is NOT a global run limit).
Saves CSV after EVERY execution and XLSX periodically; safe to resume.
Only QTLA is imported, not the ILS implementation.
"""
from __future__ import annotations

from pathlib import Path
import sys
from time import perf_counter
from typing import Any
from dataclasses import dataclass
from xml.sax.saxutils import escape as xml_escape
import csv
import json
import math
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from qtla_redesign import (
    QTLAConfig, QTLAParameters, load_instance,
    objective_value, solve_qtla,
)

INSTANCE_DIR = ROOT / "instances" / "instances_seed_21_22_23_N_900_1200_1500"
RESULT_DIR = ROOT / "results" / "qtla_stress_large_900_1200_1500"
CHECKPOINT_CSV = RESULT_DIR / "checkpoint_qtla_stress_large_official_30seeds.csv"
REPORT_TXT = RESULT_DIR / "relatorio_qtla_stress_large_official_30seeds.txt"
DESIGN_JSON = RESULT_DIR / "desenho_qtla_stress_large_official_30seeds.json"
XLSX_FILE = RESULT_DIR / "resultados_qtla_stress_large_official_30seeds.xlsx"
METHODS = ["QTLA_TR_LP_NG"]
METHOD = METHODS[0]

# EXACT SAME 30 ALGORITHMIC SEEDS AS THE PREVIOUS HELD-OUT RUN
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

# Official generator configuration (distinct from algorithm seeds).
INSTANCE_SEEDS = [21, 22, 23]
EXPECTED_DESIGN = {900: [60, 80], 1200: [100, 150], 1500: [200, 300]}
LAMBDA_VALUES = [0.5, 1.0, 2.0]
EXPECTED_COUNT = 54

# Frozen irace configuration ID 68. NOA OFF, TR+LP+NG ON.
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
    "overall_time_limit": None,
}


@dataclass(frozen=True)
class InstanceData:
    name: str
    path: Path
    N: int
    ni: int
    nk: int
    nj: int
    p: int
    lambda_: float
    instance_seed: int

    @property
    def n_i(self): return self.ni
    @property
    def n_k(self): return self.nk
    @property
    def n_j(self): return self.nj
    @property
    def alpha(self): return self.lambda_


def scalar(z, key, cast):
    v = z[key]
    return cast(v.item() if isinstance(v, np.ndarray) and v.shape == () else v)


def read_header(path: Path) -> InstanceData:
    with np.load(path, allow_pickle=False) as z:
        ni = scalar(z, "n_i", int)
        nk = scalar(z, "n_k", int)
        nj = scalar(z, "n_j", int)
        p = scalar(z, "p", int)
        lam = scalar(z, "alpha", float)
        instance_seed = scalar(z, "seed", int)
        # Validate dimensions without copying a/b matrices into memory.
        a_shape = z["a"].shape
        b_shape = z["b"].shape
    N = ni + nk + nj
    if N != 2 * ni or nk != nj or N % 4 != 0 or nk != N // 4:
        raise ValueError(f"Bad N/I/K/J proportions in {path.name}: {N}/{ni}/{nk}/{nj}")
    if a_shape != (ni, nk + nj) or b_shape != a_shape:
        raise ValueError(f"Bad coefficient dimensions in {path.name}")
    if not (0 < p <= nj and p <= nk):
        raise ValueError(f"Invalid p={p} in {path.name}")
    return InstanceData(path.stem, path, N, ni, nk, nj, p, lam, instance_seed)


def load_official_instances() -> list[InstanceData]:
    if not INSTANCE_DIR.is_dir():
        raise FileNotFoundError(f"Missing official instance directory: {INSTANCE_DIR}")
    files = sorted(INSTANCE_DIR.glob("*.npz"))
    if len(files) != EXPECTED_COUNT:
        raise RuntimeError(f"Expected {EXPECTED_COUNT} .npz files, found {len(files)} in {INSTANCE_DIR}")
    instances = [read_header(path) for path in files]
    expected = {
        (s, N, p, round(lam, 10))
        for s in INSTANCE_SEEDS
        for N, ps in EXPECTED_DESIGN.items()
        for p in ps
        for lam in LAMBDA_VALUES
    }
    found = [(d.instance_seed, d.N, d.p, round(d.lambda_, 10)) for d in instances]
    duplicates = len(found) - len(set(found))
    if duplicates or set(found) != expected:
        raise RuntimeError(
            f"Official instance design mismatch: duplicates={duplicates}; "
            f"missing={sorted(expected-set(found))}; extra={sorted(set(found)-expected)}"
        )
    instances.sort(key=lambda d: (d.instance_seed, d.N, d.p, d.lambda_, d.name))
    return instances


def experiment_design(instances: list[InstanceData]) -> dict:
    return {
        "experiment": "Official large stress: calibrated QTLA[TR+LP+NG], 30 seeds",
        "instance_dir": str(INSTANCE_DIR.resolve()),
        "instances": [d.name for d in instances],
        "instance_seeds": INSTANCE_SEEDS,
        "expected_design": EXPECTED_DESIGN,
        "lambda_values": LAMBDA_VALUES,
        "algorithm_seeds": ALGORITHM_SEEDS,
        "qtla_config": QTLA_FINAL_CONFIG,
        "qtla_parameters": FROZEN_QTLA_PARAMETERS,
        "expected_runs": len(instances) * len(ALGORITHM_SEEDS),
    }


def ensure_design(instances: list[InstanceData]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(experiment_design(instances), indent=2, ensure_ascii=False, sort_keys=True)
    if DESIGN_JSON.exists():
        old = json.loads(DESIGN_JSON.read_text(encoding="utf-8"))
        current = experiment_design(instances)
        # Historical design files store the previous absolute input path.
        # Ignore ONLY that relocated path; every experimental setting must match.
        old_without_path = {k: v for k, v in old.items() if k != "instance_dir"}
        current_without_path = {k: v for k, v in current.items() if k != "instance_dir"}
        if old_without_path != current_without_path:
            raise RuntimeError("Design file differs from this run. Do not mix experimental results.")
    else:
        DESIGN_JSON.write_text(payload + "\n", encoding="utf-8")


COLUMNS = [
    "Method", "Instance", "N", "ni", "nk", "nj", "p", "alpha",
    "instance_seed", "algorithm_seed", "C", "time_s", "wall_time_s",
    "iterations", "status", "feasible", "audit_abs_diff", "best_seen_C", "error",
]


def row_key(row):
    return (str(row["Instance"]), int(row["algorithm_seed"]))


def load_checkpoint() -> dict:
    rows = {}
    if not CHECKPOINT_CSV.exists():
        return rows
    with CHECKPOINT_CSV.open("r", newline="", encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp, delimiter=";"):
            if not r.get("Instance") or not r.get("algorithm_seed"):
                continue
            for key in ("N", "ni", "nk", "nj", "p", "instance_seed", "algorithm_seed"):
                r[key] = int(r[key])
            r["alpha"] = float(r["alpha"])
            for key in ("C", "time_s", "wall_time_s", "audit_abs_diff", "best_seen_C"):
                r[key] = float(r[key]) if r.get(key) not in ("", None) else None
            for key in ("iterations", "feasible"):
                r[key] = int(r[key]) if r.get(key) not in ("", None) else None
            rows[row_key(r)] = r
    return rows


def save_checkpoint(results: dict) -> None:
    order = {s: i for i, s in enumerate(ALGORITHM_SEEDS)}
    sorted_rows = sorted(results.values(), key=lambda r: (
        int(r["instance_seed"]), int(r["N"]), int(r["p"]), float(r["alpha"]),
        str(r["Instance"]), order.get(int(r["algorithm_seed"]), 100000),
    ))
    temp = CHECKPOINT_CSV.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=COLUMNS, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted_rows)
    try:
        temp.replace(CHECKPOINT_CSV)
    except PermissionError:
        backup = CHECKPOINT_CSV.with_name(CHECKPOINT_CSV.stem + "_backup.csv")
        temp.replace(backup)
        print(f"[AVISO] CSV aberto/bloqueado: backup salvo em {backup}")


def run_one(dt: InstanceData, algorithm_seed: int) -> dict:
    instance = load_instance(dt.path)
    cfg = QTLAConfig(**QTLA_FINAL_CONFIG)
    params = QTLAParameters(**FROZEN_QTLA_PARAMETERS)
    start = perf_counter()
    result = solve_qtla(instance, cfg, params, seed=int(algorithm_seed), verbose=False)
    wall = perf_counter() - start
    if not result.feasible:
        raise RuntimeError("QTLA returned an infeasible solution")
    z = np.zeros(instance["n"], dtype=float)
    for index in result.combined_open_0based:
        z[int(index)] = 1.0
    # Check original maximization objective without changing the result.
    C = float(objective_value(z, instance))
    delta = abs(float(result.objective) - C)
    if not math.isclose(float(result.objective), C, rel_tol=1e-9, abs_tol=1e-8):
        raise RuntimeError(f"Objective audit mismatch: reported={result.objective}, audit={C}")
    if not math.isfinite(C) or not math.isfinite(result.runtime):
        raise RuntimeError("Non-finite C or time returned")
    if result.active_components != ["trust_region", "local_polish", "no_good"]:
        raise RuntimeError(f"Incorrect active components: {result.active_components}")
    return {
        "Method": METHOD, "Instance": dt.name, "N": dt.N, "ni": dt.ni,
        "nk": dt.nk, "nj": dt.nj, "p": dt.p, "alpha": dt.lambda_,
        "instance_seed": dt.instance_seed, "algorithm_seed": int(algorithm_seed),
        "C": C, "time_s": float(result.runtime), "wall_time_s": wall,
        "iterations": int(result.iterations), "status": result.status,
        "feasible": 1, "audit_abs_diff": delta,
        "best_seen_C": float(result.best_seen_objective), "error": "",
    }


def log_event(row: dict) -> None:
    with REPORT_TXT.open("a", encoding="utf-8") as fp:
        if row.get("error"):
            fp.write(f"ERROR | {row['Instance']} | generation_seed={row['instance_seed']} "
                     f"| algorithm_seed={row['algorithm_seed']} | {row['error']}\n")
        else:
            fp.write(f"{row['Instance']} | gen={row['instance_seed']} "
                     f"| seed={row['algorithm_seed']} | C={row['C']:.10f} "
                     f"| time={row['time_s']:.6f}s | iterations={row['iterations']} "
                     f"| status={row['status']} | audit={row['audit_abs_diff']:.3e}\n")


def initialize_report(instances):
    if REPORT_TXT.exists():
        return
    with REPORT_TXT.open("w", encoding="utf-8") as fp:
        fp.write("QTLA[TR+LP+NG] - OFFICIAL LARGE STRESS - 30 ALGORITHM SEEDS\n")
        fp.write(f"Instances={len(instances)}, runs={len(instances)*30}\n")
        fp.write("No overall limit. Each Gurobi surrogate has a calibrated 2.0s limit.\n")
        fp.write(f"Algorithm seeds: {ALGORITHM_SEEDS}\n")
        fp.write(f"Config: {QTLA_FINAL_CONFIG}\n")
        fp.write(f"Parameters: {FROZEN_QTLA_PARAMETERS}\n\n")


def summary_for_instance(dt, results):
    valid = [results.get((dt.name, seed), results.get((METHOD, dt.name, seed))) for seed in ALGORITHM_SEEDS]
    valid = [r for r in valid if r is not None and not r.get("error") and int(r.get("feasible") or 0) == 1]
    cs = np.asarray([float(r["C"]) for r in valid], dtype=float)
    ts = np.asarray([float(r["time_s"]) for r in valid], dtype=float)
    def stats(arr):
        if not arr.size:
            return (None, None, None, None, None)
        mean = float(np.mean(arr))
        sd = float(np.std(arr, ddof=0))
        cv = 100.0*sd/mean if mean != 0.0 else None
        return (mean, sd, cv, float(np.min(arr)), float(np.max(arr)))
    return len(valid), stats(cs), stats(ts)


def append_final_report(instances, results):
    text = REPORT_TXT.read_text(encoding="utf-8") if REPORT_TXT.exists() else ""
    if "FINAL SUMMARY (54 INSTANCES)" in text:
        return
    with REPORT_TXT.open("a", encoding="utf-8") as fp:
        fp.write("\nFINAL SUMMARY (54 INSTANCES)\n")
        for dt in instances:
            n, c, t = summary_for_instance(dt, results)
            fp.write(f"{dt.name} | valid={n}/30 | C_mean={c[0]:.10f} "
                     f"| C_CV(%)={c[2]:.6f} | T_mean={t[0]:.6f} "
                     f"| T_CV(%)={t[2]:.6f}\n")


# -----------------------------------------------------------------------------
# Excel writer: uses the same self-contained OOXML exporter as the user's
# original held-out runner. The exact per-seed sheet is kept; "Resumo" is added.
# -----------------------------------------------------------------------------


def build_summary_sheet_xml(instances, results) -> str:
    headers = ["Instance", "N", "p", "lambda", "instance_seed", "Valid_runs",
               "Mean_C", "SD_C", "CV_C_%", "Min_C", "Max_C",
               "Mean_Time_s", "SD_Time_s", "CV_Time_%", "Min_Time_s", "Max_Time_s"]
    lines = [f'<row r="1">' + "".join(
        xlsx_cell(f"{excel_col(i)}1", h, style=1)
        for i, h in enumerate(headers, 1)) + '</row>']
    for ri, dt in enumerate(instances, 2):
        n, c, t = summary_for_instance(dt, results)
        values = [dt.name, dt.N, dt.p, dt.lambda_, dt.instance_seed, n,
                  *c, *t]
        # Values are drawn from completed runs; no fabricated averages.
        lines.append(f'<row r="{ri}">' + "".join(
            xlsx_cell(f"{excel_col(i)}{ri}", v, style=(3 if isinstance(v, float) else 0))
            for i, v in enumerate(values, 1)) + '</row>')
    last = excel_col(len(headers))
    cols = ''.join(f'<col min="{i}" max="{i}" width="{38 if i == 1 else 17}" customWidth="1"/>'
                   for i in range(1, len(headers)+1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<sheetViews><sheetView workbookViewId="0"><pane xSplit="1" ySplit="1" topLeftCell="B2" activePane="bottomRight" state="frozen"/></sheetView></sheetViews>
<cols>{cols}</cols><sheetData>{"".join(lines)}</sheetData>
<autoFilter ref="A1:{last}{len(instances)+1}"/></worksheet>"""


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
                formula=f'IFERROR(AVERAGE({",".join(c_refs)}),"")',
            )
        )
        cells.append(
            xlsx_cell(
                mean_t_ref,
                style=7,
                formula=f'IFERROR(AVERAGE({",".join(t_refs)}),"")',
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
    sheet_names = METHODS + ["Resumo"]

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
 <dc:title>QTLA TR+LP+NG - Official large stress test - 30 algorithm seeds</dc:title>
 <dc:creator>Held-out runner</dc:creator>
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
                build_summary_sheet_xml(instances, results) if method == "Resumo" else build_sheet_xml(method, instances, results),
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



def main():
    if len(ALGORITHM_SEEDS) != 30 or len(set(ALGORITHM_SEEDS)) != 30:
        raise RuntimeError("Invalid 30-seed algorithm list")
    instances = load_official_instances()
    ensure_design(instances)
    initialize_report(instances)
    results = load_checkpoint()
    expected_keys = {(dt.name, s) for dt in instances for s in ALGORITHM_SEEDS}
    unknown = set(results) - expected_keys
    if unknown:
        raise RuntimeError(f"Checkpoint contains unrecognized experiment runs: {list(unknown)[:5]}")
    jobs = [(dt, seed) for dt in instances for seed in ALGORITHM_SEEDS]
    print("="*90)
    print("OFFICIAL LARGE STRESS - QTLA[TR+LP+NG] - 30 ALGORITHMIC SEEDS")
    print("="*90)
    print(f"Instance directory : {INSTANCE_DIR}")
    print(f"Instances          : {len(instances)} (seeds 21,22,23)")
    print(f"Algorithmic seeds  : {len(ALGORITHM_SEEDS)}")
    print(f"Expected runs      : {len(jobs)}")
    print(f"Previously valid   : {sum(not r.get('error') for r in results.values())}")
    print("Overall time limit : NONE (per-Gurobi-subproblem limit = 2s)")
    print(f"Output directory   : {RESULT_DIR}")
    write_xlsx(instances, {(METHOD, *key): r for key, r in results.items()})
    excel_counter = 0
    for position, (dt, seed) in enumerate(jobs, 1):
        key = (dt.name, seed)
        if key in results and not results[key].get("error"):
            continue
        print(f"[{position:4d}/{len(jobs)}] {dt.name} | generation={dt.instance_seed} "
              f"| algorithm={seed} | ", end="", flush=True)
        try:
            row = run_one(dt, seed)
            print(f"OK C={row['C']:.8f} Time={row['time_s']:.4f}s "
                  f"status={row['status']}")
        except Exception as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")
            traceback.print_exc()
            row = {"Method":METHOD, "Instance":dt.name, "N":dt.N, "ni":dt.ni,
                   "nk":dt.nk, "nj":dt.nj, "p":dt.p, "alpha":dt.lambda_,
                   "instance_seed":dt.instance_seed, "algorithm_seed":seed,
                   "C":None, "time_s":None, "wall_time_s":None,
                   "iterations":None, "status":"ERROR", "feasible":0,
                   "audit_abs_diff":None, "best_seen_C":None,
                   "error":f"{type(exc).__name__}: {exc}"}
        results[key] = row
        log_event(row)
        save_checkpoint(results)
        excel_counter += 1
        if excel_counter >= 10:
            write_xlsx(instances, {(METHOD, *key): r for key, r in results.items()})
            excel_counter = 0
    save_checkpoint(results)
    write_xlsx(instances, {(METHOD, *key): r for key, r in results.items()})
    valid = sum(not r.get("error") for r in results.values())
    errors = sum(bool(r.get("error")) for r in results.values())
    if valid == len(jobs) and errors == 0:
        append_final_report(instances, results)
    print(f"\nFINISHED: {valid}/{len(jobs)} valid; errors={errors}")
    print(f"Checkpoint: {CHECKPOINT_CSV}")
    print(f"Excel: {XLSX_FILE}")
    print(f"Report: {REPORT_TXT}")
    if valid != len(jobs) or errors:
        raise RuntimeError("Some runs failed/missing. Reexecute the script to retry them.")


if __name__ == "__main__":
    main()
