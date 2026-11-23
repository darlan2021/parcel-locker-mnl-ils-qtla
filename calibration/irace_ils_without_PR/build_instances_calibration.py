from pathlib import Path
import csv
import numpy as np
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
INSTANCE_DIR=PROJECT/'instances_calibration'
EXPECTED_N=(60,100,200,400)
EXPECTED_ALPHA=(0.5,1.0,2.0)
EXPECTED_SEEDS=(11,12,13)
P_TIERS={
    'LOW': {60:3,100:5,200:10,400:20},
    'HIGH':{60:6,100:10,200:20,400:30},
}
EXPECTED_COUNT=72

def scalar(z,key):
    a=np.asarray(z[key]); return a.item() if a.size==1 else a

def load_meta(path):
    with np.load(path,allow_pickle=False) as z:
        n_i=int(scalar(z,'n_i')); n_j=int(scalar(z,'n_j')); n_k=int(scalar(z,'n_k'))
        p=int(scalar(z,'p')); seed=int(scalar(z,'seed'))
        if 'alpha' in z.files: alpha=float(scalar(z,'alpha'))
        elif 'lambda_' in z.files: alpha=float(scalar(z,'lambda_'))
        elif 'lambda' in z.files: alpha=float(scalar(z,'lambda'))
        else: raise RuntimeError(f'{path.name}: alpha/lambda missing')
        a=np.asarray(z['a']); b=np.asarray(z['b'])
    N=2*n_i
    if a.shape!=(n_i,n_k+n_j): raise RuntimeError(f'{path.name}: bad a.shape={a.shape}')
    if b.shape!=(n_i,n_k+n_j): raise RuntimeError(f'{path.name}: bad b.shape={b.shape}')
    if np.any(b<=0) or not np.all(np.isfinite(b)): raise RuntimeError(f'{path.name}: invalid b')
    if not np.all(np.isfinite(a)): raise RuntimeError(f'{path.name}: invalid a')
    return dict(file=path.name,N=N,n_i=n_i,n_k=n_k,n_j=n_j,p=p,alpha=alpha,seed=seed)

def tier_of(N,p):
    hits=[t for t,m in P_TIERS.items() if m.get(N)==p]
    if len(hits)!=1: raise RuntimeError(f'Unexpected design point N={N}, p={p}')
    return hits[0]

def main():
    if not INSTANCE_DIR.exists(): raise SystemExit(f'ERROR: folder not found: {INSTANCE_DIR}')
    paths=sorted(INSTANCE_DIR.glob('*.npz'))
    print(f'NPZ files found: {len(paths)}')
    if len(paths)!=EXPECTED_COUNT: raise SystemExit(f'ERROR: expected {EXPECTED_COUNT} NPZ files, found {len(paths)}')
    rows=[load_meta(p) for p in paths]
    Ns=tuple(sorted({r['N'] for r in rows})); alphas=tuple(sorted({round(r['alpha'],10) for r in rows})); seeds=tuple(sorted({r['seed'] for r in rows}))
    if Ns!=EXPECTED_N: raise SystemExit(f'ERROR: N levels={Ns}')
    if alphas!=EXPECTED_ALPHA: raise SystemExit(f'ERROR: alpha levels={alphas}')
    if seeds!=EXPECTED_SEEDS: raise SystemExit(f'ERROR: seeds={seeds}')
    for r in rows: r['p_tier']=tier_of(r['N'],r['p'])
    grouped={}
    for r in rows: grouped.setdefault((r['seed'],round(r['alpha'],10),r['p_tier']),[]).append(r)
    if len(grouped)!=18: raise SystemExit(f'ERROR: expected 18 blocks, found {len(grouped)}')
    order_t={'LOW':0,'HIGH':1}; ordered=[]; bid=0
    for key in sorted(grouped,key=lambda k:(k[0],k[1],order_t[k[2]])):
        block=sorted(grouped[key],key=lambda r:r['N'])
        if [r['N'] for r in block]!=list(EXPECTED_N): raise SystemExit(f'ERROR: incomplete block {key}')
        tier=key[2]
        for r in block:
            if r['p']!=P_TIERS[tier][r['N']]: raise SystemExit(f"ERROR: wrong p in {r['file']}")
        bid+=1
        for pos,r in enumerate(block,1):
            q=dict(r); q['block_id']=bid; q['position_in_block']=pos; ordered.append(q)
    with (HERE/'instances_calibration.txt').open('w',encoding='utf-8',newline='\n') as f:
        for r in ordered: f.write(r['file']+'\n')
    fields=['block_id','position_in_block','p_tier','file','N','n_i','n_k','n_j','p','alpha','seed']
    with (HERE/'manifest_calibration.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); [w.writerow({k:r[k] for k in fields}) for r in ordered]
    print('OK'); print(f'Instances      : {len(ordered)}'); print(f'N              : {list(EXPECTED_N)}'); print(f'alpha          : {list(EXPECTED_ALPHA)}'); print(f'instance seeds : {list(EXPECTED_SEEDS)}'); print(f"LOW p          : {P_TIERS['LOW']}"); print(f"HIGH p         : {P_TIERS['HIGH']}"); print('blocks         : 18 x 4')
if __name__=='__main__': main()
