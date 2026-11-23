IRACE — PATH RELINKING CALIBRATION FOR THE FROZEN ILvAS1
========================================================

WHERE TO PLACE THIS FOLDER
--------------------------
The calibration directory is located at:

    parcel-locker-mnl-ils-qtla/calibration/irace_ils_with_PR/

Expected project structure:

parcel-locker-mnl-ils-qtla/
├── src/
│   └── mnl_ils.py
├── instances/
│   └── instances_seed_11_12_13_N60_100_200_400/
│       └── 72 calibration .npz instances
└── calibration/
    └── irace_ils_with_PR/
        ├── 01_preflight.R
        ├── 02_run_irace.R
        ├── 03_summarize_runs.R
        ├── scenario.txt
        ├── parameters.txt
        ├── initial_configuration.txt
        ├── instances_calibration.txt
        ├── manifest_calibration.csv
        ├── frozen_parameters.txt
        └── target_runner_pr.py

The calibration results already generated (including the .Rdata file,
CSV tables and logs) belong to this same calibration directory.

WHAT IS FROZEN
--------------
The target runner passes the selected ILvAS1 values explicitly. It does NOT
depend on the default parameter values currently written in mnl_ils.py.

Frozen:
- A enabled
- S1 enabled
- S2 disabled
- BI local search
- grasp_starts = 8
- grasp_alpha = 0.0823
- tau_tabu = 13
- max_cardinality_violation = 1
- penalty_multiplier = 2.2255
- penalty_quadratic = 0.6576
- initial_perturbation_strength = 2
- max_perturbation_strength = 10
- stagnation_window = 22
- acceptance_tolerance = 0.0074
- elite_size = 6
- elite_min_hamming = 6
- max_iterations = 10

PR-S2 is fixed OFF:
- path_relink_s2_samples = 0

WHAT IRACE CALIBRATES
---------------------
Only two PR parameters:

1) prTriggerNoImprove = {1, 2, 3, 5}
2) prBidirectional    = {0, 1}

There are exactly 8 admissible combinations.

IMPLEMENTATION OF THE NEW PR TRIGGER
------------------------------------
The original ../../src/mnl_ils.py is imported unchanged via the configured Python import path.

target_runner_pr.py defines a small subclass whose only algorithmic control-flow
change is the PR activation rule:

    old: periodic activation by path_relink_interval
    new: activation after q consecutive non-improving ILS iterations

For trigger q, PR is attempted at q, 2q, 3q, ... stagnating iterations.
Any global improvement resets the stagnation counter.

The actual path-relinking routine, elite pool, guide selection, local search,
objective evaluation, and all other ILS mechanisms are inherited from the
original mnl_ils.py.

CALIBRATION INSTANCES
---------------------
72 NPZ instances:
- N = 60, 100, 200, 400
- two p levels per size group
- alpha = 0.5, 1.0, 2.0
- instance seeds = 11, 12, 13

The list is ordered as 18 blocks x 4 size groups, matching the previous
four-group calibration design.

HOW TO RUN IN RSTUDIO
---------------------
1. Keep this folder under calibration/ in the project root.
2. Set the RStudio working directory to calibration/irace_ils_with_PR.
3. Run:

       source("01_preflight.R")

4. Only if the pre-flight ends with SUCCESSFUL: TRUE, run:

       source("02_run_irace.R")

5. After completion, optionally run:

       source("03_summarize_runs.R")

IRACE COST
----------
The model maximizes C, whereas irace minimizes cost. Therefore:

    cost = -C

CPU time and detailed PR effort are logged in run_level_results.csv for
subsequent tie-breaking/interpretation.

BUDGET
------
There are 8 PR configurations and 72 instances:

    8 x 72 = 576

Therefore maxExperiments = 576 is sufficient even if no configuration is
discarded before the final block.

PYTHON ENVIRONMENT
------------------
scenario.txt uses targetRunnerLauncher = "python". Ensure that this points
to a Python interpreter with the required dependencies installed, and that
it is accessible from the RStudio process.

The scenario uses the relative instance path:

    ../../instances/instances_seed_11_12_13_N60_100_200_400

The original instances_calibration.txt must retain its order of 72 entries.
Do not rerun the official calibration in the directory holding historic
results: 01_preflight.R and 02_run_irace.R intentionally guard against
accidental reuse/overwriting. Run reproductions from a separate copy.
