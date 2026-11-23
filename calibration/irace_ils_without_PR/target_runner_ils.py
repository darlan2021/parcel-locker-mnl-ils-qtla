from __future__ import annotations
import argparse,csv,math,sys,time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
SRC_DIR = PROJECT / "src"
INSTANCE_DIR = PROJECT / "instances" / "instances_seed_11_12_13_N60_100_200_400"
sys.path.insert(0, str(SRC_DIR))
from mnl_ils import ILSConfig,MNLParcelLockerILS
@dataclass
class InstanceData:
    n_i:int; n_j:int; n_k:int; p:int; alpha:float; seed:int; a:np.ndarray; b:np.ndarray

def scalar(z,key):
    a=np.asarray(z[key]); return a.item() if a.size==1 else a

def load_instance(path):
    with np.load(path,allow_pickle=False) as z:
        if 'alpha' in z.files: alpha=float(scalar(z,'alpha'))
        elif 'lambda_' in z.files: alpha=float(scalar(z,'lambda_'))
        elif 'lambda' in z.files: alpha=float(scalar(z,'lambda'))
        else: raise RuntimeError(f'{path.name}: alpha/lambda missing')
        dt=InstanceData(int(scalar(z,'n_i')),int(scalar(z,'n_j')),int(scalar(z,'n_k')),int(scalar(z,'p')),alpha,int(scalar(z,'seed')),np.asarray(z['a'],dtype=float),np.asarray(z['b'],dtype=float))
    if dt.a.shape!=(dt.n_i,dt.n_k+dt.n_j): raise RuntimeError(f'{path.name}: bad a.shape')
    if dt.b.shape!=(dt.n_i,dt.n_k+dt.n_j): raise RuntimeError(f'{path.name}: bad b.shape')
    return dt

def parser():
    p=argparse.ArgumentParser(); p.add_argument('configuration_id',type=int); p.add_argument('instance_id',type=int); p.add_argument('algorithm_seed',type=int); p.add_argument('instance')
    p.add_argument('--grasp-alpha',type=float,required=True); p.add_argument('--tau-tabu',type=int,required=True); p.add_argument('--max-cardinality-violation',type=int,required=True); p.add_argument('--penalty-multiplier',type=float,required=True); p.add_argument('--penalty-quadratic',type=float,required=True); p.add_argument('--initial-perturbation',type=int,required=True); p.add_argument('--max-perturbation',type=int,required=True); p.add_argument('--stagnation-window',type=int,required=True); p.add_argument('--acceptance-tolerance',type=float,required=True); p.add_argument('--elite-size',type=int,required=True); p.add_argument('--elite-min-hamming',type=int,required=True); p.add_argument('--max-iterations',type=int,required=True); p.add_argument('--first-improvement',type=int,choices=(0,1),required=True); return p

def append_log(row):
    path=HERE/'run_level_results.csv'; new=(not path.exists()) or path.stat().st_size==0
    with path.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if new: w.writeheader()
        w.writerow(row); f.flush()

def main():
    a=parser().parse_args()
    raw = Path(a.instance)
    # Resolve the instance filename within the official calibration set.
    # This works whether irace supplies an absolute or a relative path.
    ip = raw.resolve() if raw.is_file() else (INSTANCE_DIR / raw.name).resolve()
    dt=load_instance(ip)
    cfg=ILSConfig(grasp_starts=8,grasp_alpha=a.grasp_alpha,max_vnd_passes=250,tau_tabu=a.tau_tabu,max_cardinality_violation=a.max_cardinality_violation,s2_sample_size=350,first_improvement=bool(a.first_improvement),enable_A=True,enable_S1=True,enable_S2=True,enable_path_relink=False,penalty_weight=None,penalty_multiplier=a.penalty_multiplier,penalty_quadratic=a.penalty_quadratic,max_iterations=a.max_iterations,time_limit=None,initial_perturbation_strength=a.initial_perturbation,min_perturbation_strength=1,max_perturbation_strength=a.max_perturbation,stagnation_window=a.stagnation_window,acceptance_tolerance=a.acceptance_tolerance,perturb_prob_A=0.15,perturb_prob_S1=0.55,perturb_prob_S2=0.30,elite_size=a.elite_size,elite_min_hamming=a.elite_min_hamming,path_relink_interval=10,path_relink_bidirectional=True,path_relink_s2_samples=80,seed=a.algorithm_seed,verbose=False)
    t0=time.perf_counter(); res=MNLParcelLockerILS(dt,cfg).solve(); elapsed=time.perf_counter()-t0
    if not res.feasible: raise RuntimeError('ILS returned infeasible final solution')
    y=np.asarray(res.y,dtype=float); den=dt.b@y
    if np.any(den<=0): raise RuntimeError('Objective audit failed: nonpositive denominator')
    obj=float(np.sum((dt.a@y)/den))
    if not math.isfinite(obj): raise RuntimeError('Objective audit failed: nonfinite objective')
    if not np.isclose(obj,float(res.objective),rtol=1e-9,atol=1e-8): raise RuntimeError(f'Objective mismatch: {obj} vs {res.objective}')
    row={'configuration_id':a.configuration_id,'instance_id':a.instance_id,'algorithm_seed':a.algorithm_seed,'instance_file':ip.name,'instance_seed':dt.seed,'N':2*dt.n_i,'p':dt.p,'alpha':dt.alpha,'strategy':'FI' if a.first_improvement else 'BI','first_improvement':a.first_improvement,'objective_C':obj,'irace_cost':-obj,'elapsed_seconds':elapsed,'feasible':int(res.feasible),'iterations_done':int(res.iterations),'grasp_alpha':a.grasp_alpha,'tau_tabu':a.tau_tabu,'max_cardinality_violation':a.max_cardinality_violation,'penalty_multiplier':a.penalty_multiplier,'penalty_quadratic':a.penalty_quadratic,'initial_perturbation':a.initial_perturbation,'max_perturbation':a.max_perturbation,'stagnation_window':a.stagnation_window,'acceptance_tolerance':a.acceptance_tolerance,'elite_size':a.elite_size,'elite_min_hamming':a.elite_min_hamming,'max_iterations':a.max_iterations}
    append_log(row); print(f'{-obj:.15g}')
if __name__=='__main__': main()
