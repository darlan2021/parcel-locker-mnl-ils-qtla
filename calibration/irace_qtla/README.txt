QT-LA / irace calibration
=========================

PROJECT STRUCTURE
-----------------
Keep this folder at:

    parcel-locker-mnl-ils-qtla/calibration/irace_qtla/

The existing QTLA code must be at:

    parcel-locker-mnl-ils-qtla/src/qtla_redesign.py

The 72 calibration instances must already be at:

    parcel-locker-mnl-ils-qtla/instances/instances_seed_11_12_13_N60_100_200_400/

Do not regenerate the instances or rename existing output files.

FIXED COMPONENTS
----------------
- trust_region
- neighbor_oa
- local_polish
- no_good

CALIBRATED PARAMETERS
---------------------
- maxIterations: {10,20,30,40,60}
- subproblemTimeLimit: {1,2,5,10}
- dampingGamma: [0.20,1.00]
- trustRadiusFactor: [0.10,1.00]
- neighborCuts: {1,2,4,6}
- polishMaxSteps: {5,10,20,40,80}

FIXED SETTINGS
--------------
- mip_gap = 1e-6
- oa_epsilon = 1e-4
- overall_time_limit = 3600 seconds per QTLA execution
- Gurobi Threads = 1 (in the QTLA implementation)
- repeated-configuration stopping is enabled

CALIBRATION DESIGN
------------------
- 72 instances: N = 60, 100, 200, 400
- Two p values per N; lambda = 0.5, 1.0, 2.0
- Instance-generation seeds = 11, 12, 13
- Ordered as 18 blocks of 4 size groups (see instances_calibration.txt)
- irace: blockSize = 4; maxExperiments = 5000; Friedman; seed = 12345; parallel = 1
- irace minimizes cost, so target_runner_qtla.py returns -C

REPRODUCTION (IN A CLEAN COPY ONLY)
-----------------------------------
Set the RStudio working directory to:
    parcel-locker-mnl-ils-qtla/calibration/irace_qtla/

Ensure the Python executable used by RStudio includes numpy, gurobipy,
and the other project dependencies. scenario.txt uses:
    targetRunnerLauncher = "python"
If Python is not on PATH, update only targetRunnerLauncher to the
appropriate local Python executable before reproducing.

Then run:
    source("01_preflight.R")
    source("02_run_irace.R")

IMPORTANT: The current calibration directory already contains completed
results. The scripts intentionally protect historical output files against
overwriting. Do not delete, replace, or rerun against those results.
Reproduce in an isolated copy without the historical run-level outputs,
Rdata/RDS files, or preflight marker.

EXISTING OUTPUTS
----------------
Preserve the original names and content of:
- irace_qtla_calibration.Rdata
- irace_qtla_results.rds
- run_level_results.csv
- final_elites.csv
- irace_summary.txt
- preflight_ok.txt

LEIA-ME.txt is documentation only: it is not imported by the algorithms.
