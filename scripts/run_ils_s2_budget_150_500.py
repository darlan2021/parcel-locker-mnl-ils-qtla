from __future__ import annotations

"""S2 sampling-budget sensitivity: budgets 150 and 500 (350 previously tested).

This runner supports both original sampling budgets (150 and 500) in one
execution. They are candidate-generation attempts, NOT instance sizes N.
Each budget uses its own original checkpoint CSV; completed runs are skipped.

Protocol: 12 development instances (instance seed 1), four S2-containing
families, heuristic seeds 1..5, PR off, all customers evaluated per sampled
move, with the original ILSConfig defaults except the explicit settings below.

Repository layout: this file is placed under scripts/; implementations under
src/; input instances under instances/; outputs under results/ils/.
The original numerical settings and output schema are unchanged.
"""

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / 'src'
sys.path.insert(0, str(SRC_DIR))

from mnl_ils import ILSConfig, MNLParcelLockerILS

INSTANCE_DIR = ROOT / 'instances' / 'instances_seed_1_N_60_100_200_400'
OUTPUT_DIR = ROOT / 'results' / 'ils' / 'ils_results_s2_budget_150_500'

# Run both historic budgets; each remains in a separate CSV.
BUDGETS = (150, 500)

def checkpoint_path(budget):
    return OUTPUT_DIR / f'checkpoint_s2_budget_{budget}.csv' 
HEURISTIC_SEEDS = range(1, 6)
FAMILIES = {
    'S2': (False, False),
    'S1S2': (False, True),
    'AS2': (True, False),
    'AS1S2': (True, True),
}
STRUCTURES = ((60,30,15,15,6), (100,50,25,25,10),
              (200,100,50,50,20), (400,200,100,100,30))
LAMBDAS = {'0p5':0.5, '1p0':1.0, '2p0':2.0}

FIELDS = [
    'instance','N','n_i','n_k','n_j','p','lambda','instance_seed',
    'configuration','s2_sample_size','heuristic_seed',
    'C','time_s','audit_abs_diff','feasible',
    's2_evaluations','s2_accepted','s2_calls','s2_seconds',
    's1_evaluations','s1_accepted','a_evaluations','a_accepted',
    'grasp_seconds','local_search_seconds','perturbation_seconds',
    'total_seconds','error',
]

@dataclass
class Instance:
    name: str
    n_i: int
    n_k: int
    n_j: int
    p: int
    alpha: float
    seed: int
    a: np.ndarray
    b: np.ndarray


def load_instances() -> list[Instance]:
    if not INSTANCE_DIR.is_dir():
        raise FileNotFoundError(f'Pasta de instancias ausente: {INSTANCE_DIR}')
    objects = []
    for N, ni, nk, nj, p in STRUCTURES:
        for ltoken, lam in LAMBDAS.items():
            name = f'N{N}_I{ni}_K{nk}_J{nj}_P{p}_a{ltoken}_s1'
            path = INSTANCE_DIR / (name + '.npz')
            if not path.is_file():
                raise FileNotFoundError(f'Instancia nao encontrada: {path}')
            with np.load(path, allow_pickle=False) as z:
                a = np.array(z['a'], dtype=float)
                b = np.array(z['b'], dtype=float)
                zi = int(z['n_i']); zk = int(z['n_k']); zj = int(z['n_j'])
                zp = int(z['p']); zs = int(z['seed']); za = float(z['alpha'])
            if (zi,zk,zj,zp,zs) != (ni,nk,nj,p,1) or not math.isclose(za,lam):
                raise ValueError(f'Metadados divergentes no arquivo {name}')
            if a.shape != (ni,nk+nj) or b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(b <= 0):
                raise ValueError(f'Matrizes invalidas em {name}')
            objects.append(Instance(name,ni,nk,nj,p,lam,zs,a,b))
    return objects


def key(row):
    return (str(row['instance']),str(row['configuration']),int(row['s2_sample_size']),int(row['heuristic_seed']))


def load_checkpoint(budget):
    done = {}
    checkpoint = checkpoint_path(budget)
    if checkpoint.exists():
        with checkpoint.open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f,delimiter=';'):
                if row.get('feasible') == 'SIM' and row.get('C') and int(row['s2_sample_size']) == budget:
                    done[key(row)] = row
    return done


def write_checkpoint(done, budget):
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    checkpoint = checkpoint_path(budget)
    temp = checkpoint.with_suffix('.tmp')
    with temp.open('w',newline='',encoding='utf-8-sig') as f:
        w = csv.DictWriter(f,fieldnames=FIELDS,delimiter=';')
        w.writeheader()
        for k in sorted(done):
            w.writerow({col:done[k].get(col,'') for col in FIELDS})
    temp.replace(checkpoint)


def run_one(instance, family, flags, budget, seed):
    ena,ens1 = flags
    cfg = ILSConfig(
        grasp_starts=8,
        s2_sample_size=budget,
        max_iterations=10,
        tau_tabu=7,
        max_cardinality_violation=2,
        initial_perturbation_strength=2,
        max_perturbation_strength=10,
        elite_size=8,
        path_relink_interval=10,
        enable_A=ena,
        enable_S1=ens1,
        enable_S2=True,
        enable_path_relink=False,
        time_limit=None,
        seed=seed,
        verbose=False,
    )
    solver = MNLParcelLockerILS(instance,cfg)
    result = solver.solve()
    y = np.asarray(result.y,dtype=float)
    num=instance.a@y; den=instance.b@y
    if np.any(den <= 0):
        raise AssertionError('Denominador nao positivo')
    full_C = float(np.sum(num/den))
    audit = abs(full_C-result.objective)
    if audit > 1e-7 * max(1.0,abs(full_C)) or not result.feasible:
        raise AssertionError(f'Falha de auditoria C={full_C} delta={audit} feasible={result.feasible}')
    pr = result.profile
    return {
        'instance':instance.name, 'N':instance.n_i+instance.n_k+instance.n_j,
        'n_i':instance.n_i,'n_k':instance.n_k,'n_j':instance.n_j,
        'p':instance.p,'lambda':instance.alpha,'instance_seed':instance.seed,
        'configuration':family,'s2_sample_size':budget,'heuristic_seed':seed,
        'C':f'{full_C:.12f}','time_s':f'{result.elapsed_seconds:.9f}',
        'audit_abs_diff':f'{audit:.3e}','feasible':'SIM',
        's2_evaluations':pr.S2_move_evaluations,
        's2_accepted':pr.S2_accepted,'s2_calls':pr.S2_calls,
        's2_seconds':f'{pr.S2_seconds:.9f}',
        's1_evaluations':pr.S1_move_evaluations,'s1_accepted':pr.S1_accepted,
        'a_evaluations':pr.A_move_evaluations,'a_accepted':pr.A_accepted,
        'grasp_seconds':f'{pr.grasp_seconds:.9f}',
        'local_search_seconds':f'{pr.local_search_seconds:.9f}',
        'perturbation_seconds':f'{pr.perturbation_seconds:.9f}',
        'total_seconds':f'{pr.total_seconds:.9f}','error':''
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true',help='Verificar arquivos e enumerar execucoes sem rodar a ILS')
    parser.add_argument('--limit',type=int,default=None,help='Executar no maximo N novos casos (teste rapido)')
    args=parser.parse_args()
    instances=load_instances()
    done={}
    for budget in BUDGETS:
        done.update(load_checkpoint(budget))
    cases=[(ins,fam,flags,b,seed) for ins in instances for fam,flags in FAMILIES.items() for b in BUDGETS for seed in HEURISTIC_SEEDS]
    missing=[v for v in cases if (v[0].name,v[1],v[3],v[4]) not in done]
    print(f'Instancias: {len(instances)} | familias S2: {len(FAMILIES)} | orcamentos: {BUDGETS} | sementes: 1..5')
    print(f'Execucoes totais: {len(cases)} | concluidas: {len(done)} | pendentes: {len(missing)}',flush=True)
    for budget in BUDGETS:
        print(f'Checkpoint S2={budget}: {checkpoint_path(budget)}',flush=True)
    if args.dry_run:
        return
    if args.limit is not None:
        missing=missing[:max(0,args.limit)]
    start=time.perf_counter()
    for idx,(ins,fam,flags,b,seed) in enumerate(missing,1):
        print(f'[{idx}/{len(missing)}] {ins.name} {fam} S2={b} seed={seed}',end=' ',flush=True)
        try:
            row=run_one(ins,fam,flags,b,seed)
        except Exception as exc:
            print(f'ERRO: {exc}',flush=True)
            raise  # incomplete job remains in the checkpoint for later resume
        done[key(row)]=row
        write_checkpoint({k:v for k,v in done.items() if k[2] == b}, b)
        print(f"C={float(row['C']):.6f} T={float(row['time_s']):.3f}s Evals={row['s2_evaluations']} Aceitos={row['s2_accepted']}",flush=True)
    print(f'Concluido. Tempo da sessao: {time.perf_counter()-start:.1f}s',flush=True)
    for budget in BUDGETS:
        print(f'Resultados S2={budget}: {checkpoint_path(budget)}',flush=True)

if __name__=='__main__':
    main()
