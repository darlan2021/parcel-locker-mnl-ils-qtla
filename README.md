# Parcel Locker Location under Multinomial Logit Choice

This repository contains the Python implementations, benchmark instances, parameter-calibration files, and computational results associated with a study of parcel-locker network reconfiguration under multinomial logit (MNL) customer choice.

The problem considers a set of customers, existing facilities, and candidate parcel lockers. The decision concerns which existing facilities to close and which candidate lockers to open. The computational study includes the exact **M-III** formulation, **Iterated Local Search (ILS)** variants, and the **QT-LA** heuristic.

## Repository organization

```text
parcel-locker-mnl-ils-qtla/
├── calibration/
│   ├── irace_ils_without_PR/
│   ├── irace_ils_with_PR/
│   └── irace_qtla/
├── instances/
│   ├── instances_seed_1_N_60_100_200_400/
│   ├── instances_seed_11_12_13_N60_100_200_400/
│   ├── instances_seed_21_22_23_N_60_100_200_400/
│   ├── instances_seed_21_22_23_N_500/
│   └── instances_seed_21_22_23_N_900_1200_1500/
├── results/
│   ├── held_ils_qt_la/
│   ├── ils/
│   │   ├── ils_component_analysis_30seeds_PROFILING_COMPLETO/
│   │   ├── ils_ilvas1p_30seeds_PROFILING_COMPLETO/
│   │   ├── ils_results_s2_budget_150_500/
│   │   └── ils_results_s2_percent_sensitivity/
│   ├── m_iii/
│   ├── qtla_stress_large_900_1200_1500/
│   ├── stress_n500_ils_qt_la/
│   ├── ttt_n500_p30_lambda2/
│   └── validation_objective/
├── scripts/
└── src/
    └── mthds/
```

- **`src/`** contains the implementations of the optimization methods and supporting data structures, including `data.py`, `mnl_ils.py`, `qtla_redesign.py`, and the modules under `mthds/`.
- **`scripts/`** contains the instance generators and experimental runners.
- **`instances/`** contains the benchmark datasets.
- **`calibration/`** contains the irace settings, target runners, and retained calibration records.
- **`results/`** contains the computational records produced by the experiments.

## Benchmark instances

Let \(I\), \(K\), and \(J\) denote the customers, existing facilities, and candidate lockers, respectively. The total instance size is \(N=|I|+|K|+|J|\), with \(|I|=N/2\) and \(|K|=|J|=N/4\). Instance parameters include \(p\), the distance-sensitivity coefficient \(\lambda\), and the instance-generation seed. Legacy data fields named `alpha` correspond to \(\lambda\).

| Dataset | \(N\) | Instance seeds | Number of instances | Use |
| --- | --- | --- | ---: | --- |
| Development | 60, 100, 200, 400 | 1 | 24 | ILS component experiments and M-III benchmarks |
| Calibration | 60, 100, 200, 400 | 11, 12, 13 | 72 | irace calibration |
| Held-out | 60, 100, 200, 400 | 21, 22, 23 | 72 | ILS and QT-LA comparison |
| Stress test | 500 | 21, 22, 23 | 27 | ILS and QT-LA comparison |
| Large-scale stress test | 900, 1200, 1500 | 21, 22, 23 | 54 | QT-LA evaluation |

For \(N=60,100,200,400\), the respective values of \(p\) are `(3,6)`, `(5,10)`, `(10,20)`, and `(20,30)`. For \(N=500\), \(p\in\{20,25,30\}\). For \(N=900,1200,1500\), the values are `(60,80)`, `(100,150)`, and `(200,300)`, respectively. All instance groups use \(\lambda\in\{0.5,1.0,2.0\}\).

Instances are distributed as `.npz` files, with corresponding `.json` metadata. Instance-generation seeds differ from the algorithmic seeds used in repeated heuristic runs.

## Optimization methods

**M-III (MILP).** Exact mixed-integer programming formulation used as a reference on selected development instances.

**ILS.** Component configurations are denoted `ILS[A]`, `ILS[S1]`, `ILS[S2]`, `ILS[A+S1]`, `ILS[A+S2]`, `ILS[S1+S2]`, and `ILS[A+S1+S2]`. The experiments also include the selected `ILS[A+S1]` configuration with Path Relinking (PR), plus studies of the S2 neighborhood.

**QT-LA.** The repository includes the QT-LA implementation and experimental runs for the selected variant with trust region (TR), local polishing (LP), and no-good cuts (NG).

## Parameter calibration

| Directory | Experiment |
| --- | --- |
| `calibration/irace_ils_without_PR/` | ILS parameter calibration, including First Improvement (FI) and Best Improvement (BI), with PR disabled |
| `calibration/irace_ils_with_PR/` | Calibration of the PR activation trigger and direction, with the ILS base configuration fixed |
| `calibration/irace_qtla/` | QT-LA parameter calibration |

Each calibration folder retains its R scripts, `scenario.txt`, `parameters.txt`, `instances_calibration.txt`, Python target runner, and original calibration outputs. The calibration dataset uses the 72 instances generated with seeds 11, 12, and 13. To run a calibration, set the R/RStudio working directory to the corresponding folder and follow its local README.

## Experimental scripts and results

| Script in `scripts/` | Experiment | Results directory |
| --- | --- | --- |
| `run_m_iii.py` | M-III exact benchmark | `results/m_iii/` |
| `run_validate_mip_ils_qtla.py` | Objective-value consistency between implementations | `results/validation_objective/` |
| `run_ils_components_30seeds_profiling_completo.py` | ILS component configurations | `results/ils/ils_component_analysis_30seeds_PROFILING_COMPLETO/` |
| `run_ilvas1p_30seeds_profiling_completo.py` | ILS[A+S1] with PR | `results/ils/ils_ilvas1p_30seeds_PROFILING_COMPLETO/` |
| `run_ils_s2_percent_sensitivity.py` | S2 customer-percentage experiments | `results/ils/ils_results_s2_percent_sensitivity/` |
| `run_ils_s2_budget_150_500.py` | S2 sampling budgets of 150 and 500 | `results/ils/ils_results_s2_budget_150_500/` |
| `run_held_out_ils_qt_la.py` | Held-out comparison between ILS and QT-LA | `results/held_ils_qt_la/` |
| `run_stress_n500_ils_qt_la.py` | Stress test with \(N=500\) | `results/stress_n500_ils_qt_la/` |
| `run_qtla_stress_large_900_1200_1500.py` | Large-scale QT-LA stress test | `results/qtla_stress_large_900_1200_1500/` |
| `run_ttt_n500_p30_lambda2.py` | Time-to-target experiment | `results/ttt_n500_p30_lambda2/` |

The instance-generation scripts are also located in `scripts/`. Original result filenames and checkpoints have been retained for traceability.

## Running the code

The Python runners are designed to be invoked from the repository root, for example:

```bash
python scripts/run_m_iii.py
```

Python dependencies include NumPy and method-specific packages. Gurobi-based methods require `gurobipy` and an appropriate Gurobi license. The calibration workflows additionally require R and the `irace` package.

Some runners write to or resume from their existing output directories. To preserve the supplied computational records, use a separate working copy when repeating experiments.
