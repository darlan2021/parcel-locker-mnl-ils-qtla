from __future__ import annotations

from pathlib import Path
from time import perf_counter
import csv
import math
import traceback
import zipfile
from xml.sax.saxutils import escape

import gurobipy as gp
from gurobipy import GRB
import numpy as np


# ============================================================
# CONFIGURACAO DO LOTE
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = ROOT / "instances" / "instances_seed_1_N_60_100_200_400"
RESULT_DIR = ROOT / "results" / "m_iii"
LOG_DIR = RESULT_DIR / "gurobi_logs"

# Casos que realmente serao executados:
# N=60  -> p=3 e p=6
# N=100 -> p=5
TARGET_CASES = {
    60: {3, 6},
    100: {5},
}

TIME_LIMIT_SECONDS = 3600.0   # 1 hora POR INSTANCIA
THREADS = 1
MIP_GAP = 1e-6
INSTANCE_SEED = 1
OUTPUT_FLAG = 1

REPORT_FILE = RESULT_DIR / "relatorio_mip_N60_N100.txt"
CSV_FILE = RESULT_DIR / "resultados_mip_N60_N100.csv"
XLSX_FILE = RESULT_DIR / "resultados_mip_N60_N100.xlsx"

STATUS_NAME = {
    GRB.LOADED: "LOADED",
    GRB.OPTIMAL: "OPTIMAL",
    GRB.INFEASIBLE: "INFEASIBLE",
    GRB.INF_OR_UNBD: "INF_OR_UNBD",
    GRB.UNBOUNDED: "UNBOUNDED",
    GRB.CUTOFF: "CUTOFF",
    GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
    GRB.NODE_LIMIT: "NODE_LIMIT",
    GRB.TIME_LIMIT: "TIME_LIMIT",
    GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    GRB.INTERRUPTED: "INTERRUPTED",
    GRB.NUMERIC: "NUMERIC",
    GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.INPROGRESS: "INPROGRESS",
    GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
    GRB.WORK_LIMIT: "WORK_LIMIT",
    GRB.MEM_LIMIT: "MEM_LIMIT",
}

COLUMNS = [
    "Instance",
    "N",
    "ni",
    "nk",
    "nj",
    "p",
    "alpha",
    "seed",
    "Status",
    "Optimality_certified",
    "C",
    "MIP_ObjVal",
    "BestBound",
    "Gap_percent",
    "Time_s",
    "Nodes",
    "Audit_abs_diff",
    "Existing_open_0based",
    "Existing_closed_0based",
    "Candidate_open_0based",
    "Combined_open_0based",
    "Error",
]


# ============================================================
# LEITURA DAS INSTANCIAS .NPZ
# ============================================================

def scalar(z, key, cast):
    value = z[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return cast(value)


def load_instance(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        ni = scalar(z, "n_i", int)
        nj = scalar(z, "n_j", int)
        nk = scalar(z, "n_k", int)
        p = scalar(z, "p", int)
        alpha = scalar(z, "alpha", float)
        seed = scalar(z, "seed", int)
        a = np.asarray(z["a"], dtype=float)
        b = np.asarray(z["b"], dtype=float)

    n = nk + nj
    N_total = ni + nk + nj

    if a.shape != (ni, n):
        raise ValueError(f"{path.name}: a.shape={a.shape}; esperado {(ni, n)}")
    if b.shape != (ni, n):
        raise ValueError(f"{path.name}: b.shape={b.shape}; esperado {(ni, n)}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{path.name}: matriz a contem valor nao-finito")
    if np.any(b <= 0.0) or not np.all(np.isfinite(b)):
        raise ValueError(f"{path.name}: matriz b deve ser positiva e finita")

    return {
        "name": path.stem,
        "path": path,
        "ni": ni,
        "nj": nj,
        "nk": nk,
        "n": n,
        "N_total": N_total,
        "p": p,
        "alpha": alpha,
        "seed": seed,
        "a": a,
        "b": b,
    }


def get_instances() -> list[dict]:
    if not INSTANCE_DIR.exists():
        raise FileNotFoundError(f"Pasta nao encontrada: {INSTANCE_DIR}")

    selected = []
    for path in INSTANCE_DIR.glob("*.npz"):
        dt = load_instance(path)
        N = dt["N_total"]
        p = dt["p"]
        if N in TARGET_CASES and p in TARGET_CASES[N] and dt["seed"] == INSTANCE_SEED:
            selected.append(dt)

    selected.sort(key=lambda d: (d["N_total"], d["p"], d["alpha"], d["seed"], d["name"]))
    return selected


# ============================================================
# MIP - MESMA FORMULACAO DE mthds/mip.py
# ============================================================

def solve_mip(dt: dict) -> dict:
    ni = dt["ni"]
    nj = dt["nj"]
    nk = dt["nk"]
    n = dt["n"]
    p = dt["p"]
    a = dt["a"]
    b = dt["b"]

    I = range(ni)
    J = range(nj)
    K = range(nk)
    NN = range(n)

    U_x1 = np.zeros((ni, n), dtype=float)
    L_x0 = np.zeros((ni, n), dtype=float)
    L_x1 = np.zeros((ni, n), dtype=float)
    U = np.zeros(ni, dtype=float)

    sum_bk = np.sum(b[:, :nk], axis=1)

    for i in I:
        bk_i = b[i, :nk]
        sorted_bk_i = np.sort(bk_i)

        bj_i = b[i, nk:]
        sorted_bj_i_desc = np.sort(bj_i)[::-1]

        sum_bk_smallest_nk_p = np.sum(sorted_bk_i[: nk - p])
        U[i] = 1.0 / sum_bk_smallest_nk_p

        sum_bj_largest_p = np.sum(sorted_bj_i_desc[:p])

        for j in NN:
            if j < nk:
                bk_i_without_k_sorted = np.sort(np.delete(bk_i, j))

                # Eq. (25)
                U_x1[i, j] = 1.0 / (
                    np.sum(bk_i_without_k_sorted[: nk - p - 1]) + bk_i[j]
                )

                # Eq. (26)
                sum_bk_without_k = sum_bk[i] - bk_i[j]
                L_x0[i, j] = 1.0 / (
                    sum_bk_without_k + sum_bj_largest_p
                )

                # Eq. (27)
                L_x1[i, j] = 1.0 / (
                    sum_bk[i] + sum_bj_largest_p
                )

            else:
                jj = j - nk
                bj_i_without_j_desc = np.sort(np.delete(bj_i, jj))[::-1]

                # Eq. (22)
                U_x1[i, j] = 1.0 / (
                    sum_bk_smallest_nk_p + bj_i[jj]
                )

                # Eq. (23)
                L_x0[i, j] = 1.0 / (
                    sum_bk[i] + np.sum(bj_i_without_j_desc[:p])
                )

                # Eq. (24)
                if p > 1:
                    L_x1[i, j] = 1.0 / (
                        sum_bk[i]
                        + bj_i[jj]
                        + np.sum(bj_i_without_j_desc[: p - 1])
                    )
                else:
                    L_x1[i, j] = 1.0 / (
                        sum_bk[i] + bj_i[jj]
                    )

    model = gp.Model(f"MIP_{dt['name']}")
    model.setParam("Threads", THREADS)
    model.setParam("OutputFlag", OUTPUT_FLAG)
    model.setParam("TimeLimit", TIME_LIMIT_SECONDS)
    model.setParam("MIPGap", MIP_GAP)
    # LogFile desativado: evita falha de escrita em caminhos longos/OneDrive.

    x = model.addVars(NN, vtype=GRB.BINARY, name="x")
    z = model.addVars(I, vtype=GRB.CONTINUOUS, lb=0.0, name="w")
    y = model.addVars(I, NN, vtype=GRB.CONTINUOUS, lb=0.0, name="y")

    model.setObjective(
        gp.quicksum(a[i][j] * y[i, j] for i in I for j in NN),
        GRB.MAXIMIZE,
    )

    # no maximo p lockers candidatos abertos
    model.addConstr(
        gp.quicksum(x[nk + j] for j in J) <= p,
        name="candidate_cardinality",
    )

    # pelo menos nk-p POPs existentes permanecem abertos
    model.addConstr(
        gp.quicksum(x[k] for k in K) >= nk - p,
        name="existing_cardinality",
    )

    for i in I:
        model.addConstr(
            gp.quicksum(b[i][k] * y[i, k] for k in NN) == 1,
            name=f"den_{i}",
        )

        for k in NN:
            model.addConstr(y[i, k] >= z[i] - U[i] * (1 - x[k]))
            model.addConstr(y[i, k] <= z[i] - L_x0[i, k] * (1 - x[k]))
            model.addConstr(y[i, k] >= L_x1[i, k] * x[k])
            model.addConstr(y[i, k] <= U_x1[i, k] * x[k])

    start = perf_counter()
    model.optimize()
    elapsed = perf_counter() - start

    status_code = int(model.Status)
    status = STATUS_NAME.get(status_code, str(status_code))
    certified = status_code == GRB.OPTIMAL

    result = {
        "Instance": dt["name"],
        "N": dt["N_total"],
        "ni": ni,
        "nk": nk,
        "nj": nj,
        "p": p,
        "alpha": dt["alpha"],
        "seed": dt["seed"],
        "Status": status,
        "Optimality_certified": "SIM" if certified else "NAO",
        "C": None,
        "MIP_ObjVal": None,
        "BestBound": None,
        "Gap_percent": None,
        "Time_s": elapsed,
        "Nodes": int(round(model.NodeCount)),
        "Audit_abs_diff": None,
        "Existing_open_0based": "",
        "Existing_closed_0based": "",
        "Candidate_open_0based": "",
        "Combined_open_0based": "",
        "Error": "",
    }

    try:
        result["BestBound"] = float(model.ObjBound)
    except Exception:
        pass

    if model.SolCount > 0:
        x_val = np.array([1 if x[j].X > 0.5 else 0 for j in NN], dtype=np.int8)

        num_val = a @ x_val
        den_val = b @ x_val
        if np.any(den_val <= 0.0):
            raise FloatingPointError("Denominador nao-positivo na auditoria final")

        C = float(np.sum(num_val / den_val))
        obj = float(model.ObjVal)

        result["C"] = C
        result["MIP_ObjVal"] = obj
        result["Audit_abs_diff"] = abs(obj - C)

        try:
            result["Gap_percent"] = float(model.MIPGap) * 100.0
        except Exception:
            pass

        existing_open = np.flatnonzero(x_val[:nk] == 1).astype(int).tolist()
        existing_closed = np.flatnonzero(x_val[:nk] == 0).astype(int).tolist()
        candidate_open = np.flatnonzero(x_val[nk:] == 1).astype(int).tolist()
        combined_open = np.flatnonzero(x_val == 1).astype(int).tolist()

        result["Existing_open_0based"] = " ".join(map(str, existing_open))
        result["Existing_closed_0based"] = " ".join(map(str, existing_closed))
        result["Candidate_open_0based"] = " ".join(map(str, candidate_open))
        result["Combined_open_0based"] = " ".join(map(str, combined_open))

    return result


# ============================================================
# SAIDAS TXT / CSV / XLSX
# ============================================================

def fmt(value, digits=8):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def init_report(instances: list[dict]):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("MIP - LOTE PILOTO\n")
        f.write("=" * 100 + "\n")
        f.write(f"Limite por instancia : {TIME_LIMIT_SECONDS:.0f} s (1 hora)\n")
        f.write(f"Threads              : {THREADS}\n")
        f.write(f"MIPGap               : {MIP_GAP:.1e}\n")
        f.write(f"Seed                 : {INSTANCE_SEED}\n")
        f.write("Casos                 : N60/p3,p6 e N100/p5\n")
        f.write(f"Numero de instancias  : {len(instances)}\n\n")

        for dt in instances:
            f.write(
                f"{dt['name']} | N={dt['N_total']} | p={dt['p']} | "
                f"alpha={dt['alpha']} | seed={dt['seed']}\n"
            )
        f.write("\n")


def append_report(r: dict):
    with REPORT_FILE.open("a", encoding="utf-8") as f:
        f.write("-" * 100 + "\n")
        f.write(f"Instance             : {r['Instance']}\n")
        f.write(f"Status               : {r['Status']}\n")
        f.write(f"Otimalidade certific.: {r['Optimality_certified']}\n")
        f.write(f"C                    : {fmt(r['C'], 10)}\n")
        f.write(f"MIP ObjVal           : {fmt(r['MIP_ObjVal'], 10)}\n")
        f.write(f"Best bound           : {fmt(r['BestBound'], 10)}\n")
        f.write(f"Gap (%)              : {fmt(r['Gap_percent'], 8)}\n")
        f.write(f"Time (s)             : {fmt(r['Time_s'], 3)}\n")
        f.write(f"Nodes                : {r['Nodes']}\n")
        f.write(f"|ObjVal-C|           : {fmt(r['Audit_abs_diff'], 12)}\n")
        f.write(f"Existing open        : {r['Existing_open_0based']}\n")
        f.write(f"Existing closed      : {r['Existing_closed_0based']}\n")
        f.write(f"Candidate open       : {r['Candidate_open_0based']}\n")
        if r["Error"]:
            f.write(f"ERROR                : {r['Error']}\n")
        f.write("\n")


def write_csv(results: list[dict]):
    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
        writer.writeheader()
        for r in results:
            writer.writerow({c: r.get(c, "") for c in COLUMNS})


# XLSX simples, sem dependencias extras.
def excel_col(n: int) -> str:
    s = ""
    while n:
        n, rem = divmod(n - 1, 26)
        s = chr(65 + rem) + s
    return s


def cell_xml(ref: str, value, header=False) -> str:
    style = ' s="1"' if header else ""
    if value is None or value == "":
        return f'<c r="{ref}"{style}/>'
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f'<c r="{ref}"{style}><v>{int(value)}</v></c>'
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        return f'<c r="{ref}"{style}><v>{float(value):.15g}</v></c>'
    return f'<c r="{ref}" t="inlineStr"{style}><is><t>{escape(str(value))}</t></is></c>'


def write_xlsx(results: list[dict]):
    rows = []
    header_cells = [cell_xml(f"{excel_col(j)}1", c, header=True) for j, c in enumerate(COLUMNS, 1)]
    rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for i, r in enumerate(results, start=2):
        cells = [cell_xml(f"{excel_col(j)}{i}", r.get(c)) for j, c in enumerate(COLUMNS, 1)]
        rows.append(f'<row r="{i}">{"".join(cells)}</row>')

    last_row = len(results) + 1
    last_col = excel_col(len(COLUMNS))

    sheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" state="frozen"/></sheetView></sheetViews>
  <sheetData>{"".join(rows)}</sheetData>
  <autoFilter ref="A1:{last_col}{last_row}"/>
</worksheet>'''

    workbook_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="Resultados" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''

    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''

    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font><sz val="11"/><name val="Calibri"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font>
  </fonts>
  <fills count="3">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF1F4E78"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>
  </cellXfs>
</styleSheet>'''

    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''

    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

    with zipfile.ZipFile(XLSX_FILE, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("xl/workbook.xml", workbook_xml)
        z.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        z.writestr("xl/styles.xml", styles_xml)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def save_partial(results: list[dict]):
    write_csv(results)
    write_xlsx(results)


# ============================================================
# MAIN
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    instances = get_instances()
    if not instances:
        raise RuntimeError("Nenhuma instancia correspondente foi encontrada.")

    print("=" * 100)
    print("MIP - LOTE PILOTO")
    print("=" * 100)
    print("Casos: N60/p3,p6 e N100/p5")
    print(f"Time limit: {TIME_LIMIT_SECONDS:.0f} s POR INSTANCIA")
    print(f"MIPGap: {MIP_GAP:.1e} | Threads: {THREADS} | Seed: {INSTANCE_SEED}")
    print(f"Saida: {RESULT_DIR}")
    print(f"Instancias encontradas: {len(instances)}")
    print()

    for dt in instances:
        print(f"  {dt['name']} | N={dt['N_total']} | p={dt['p']} | alpha={dt['alpha']}")

    print()
    init_report(instances)
    results = []

    for idx, dt in enumerate(instances, start=1):
        print("\n" + "=" * 100)
        print(f"[{idx}/{len(instances)}] {dt['name']}")
        print("=" * 100)

        try:
            r = solve_mip(dt)
        except Exception as exc:
            traceback.print_exc()
            r = {
                "Instance": dt["name"],
                "N": dt["N_total"],
                "ni": dt["ni"],
                "nk": dt["nk"],
                "nj": dt["nj"],
                "p": dt["p"],
                "alpha": dt["alpha"],
                "seed": dt["seed"],
                "Status": "ERROR",
                "Optimality_certified": "NAO",
                "C": None,
                "MIP_ObjVal": None,
                "BestBound": None,
                "Gap_percent": None,
                "Time_s": None,
                "Nodes": 0,
                "Audit_abs_diff": None,
                "Existing_open_0based": "",
                "Existing_closed_0based": "",
                "Candidate_open_0based": "",
                "Combined_open_0based": "",
                "Error": f"{type(exc).__name__}: {exc}",
            }

        results.append(r)
        append_report(r)
        save_partial(results)  # salva apos CADA instancia

        print(f"Status               : {r['Status']}")
        print(f"Otimalidade certific.: {r['Optimality_certified']}")
        print(f"C                    : {fmt(r['C'], 10)}")
        print(f"Gap (%)              : {fmt(r['Gap_percent'], 8)}")
        print(f"Time (s)             : {fmt(r['Time_s'], 3)}")

    print("\n" + "=" * 100)
    print("FIM")
    print("=" * 100)
    print(f"Relatorio : {REPORT_FILE}")
    print(f"Excel     : {XLSX_FILE}")
    print(f"CSV       : {CSV_FILE}")


if __name__ == "__main__":
    main()
