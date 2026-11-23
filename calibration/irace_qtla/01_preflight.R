rm(list = ls())
suppressPackageStartupMessages(library(irace))
cat("============================================================\n")
cat("QT-LA IRACE PRE-FLIGHT\n")
cat("============================================================\n\n")
required <- c("parameters.txt","initial_configuration.txt","scenario.txt","instances_calibration.txt","target_runner_qtla.py")
missing <- required[!file.exists(required)]
if (length(missing) > 0L) stop("Missing files: ", paste(missing, collapse=", "))
if (file.exists("run_level_results.csv")) stop("run_level_results.csv already exists. Move/delete old output first.")
instances <- trimws(readLines("instances_calibration.txt", warn=FALSE)); instances <- instances[nzchar(instances)]
if (length(instances) != 72L) stop("Expected 72 instances, found ", length(instances))
# Reuse the official calibration instances in the repository (no regeneration).
project_root <- normalizePath(file.path(getwd(), "..", ".."), winslash="/", mustWork=TRUE)
instance_dir <- file.path(project_root, "instances", "instances_seed_11_12_13_N60_100_200_400")
if (!file.exists(file.path(project_root, "src", "qtla_redesign.py"))) {
  stop("Missing QTLA implementation in src/qtla_redesign.py")
}
if (!dir.exists(instance_dir)) stop("Missing calibration directory: ", instance_dir)
paths <- file.path(instance_dir, instances)
miss <- instances[!file.exists(paths)]
if (length(miss)>0L) stop("Missing calibration instances:\n", paste(miss, collapse="\n"))
scenario <- readScenario(filename="scenario.txt")
checkIraceScenario(scenario)
cat("Instances:", length(instances), "\n")
cat("Blocks: 18 x 4\n")
cat("maxExperiments:", scenario$maxExperiments, "\n")
cat("seed:", scenario$seed, "\n\n")
python <- scenario$targetRunnerLauncher
runner <- normalizePath("target_runner_qtla.py", winslash="/", mustWork=TRUE)
inst <- normalizePath(paths[1], winslash="/", mustWork=TRUE)
args <- c(runner,"0","0","123456",inst,"--max-iterations","40","--subproblem-time-limit","5","--damping-gamma","0.60","--trust-radius-factor","0.50","--neighbor-cuts","2","--polish-max-steps","80")
cat("Running target-runner smoke test...\n")
out <- system2(python,args=args,stdout=TRUE,stderr=TRUE)
status <- attr(out,"status"); if (is.null(status)) status <- 0L
if (status != 0L) stop("Target runner failed:\n", paste(out,collapse="\n"))
nums <- suppressWarnings(as.numeric(out)); nums <- nums[is.finite(nums)]
if (length(nums)<1L) stop("No numeric cost returned:\n",paste(out,collapse="\n"))
cat("Smoke-test cost:", tail(nums,1), "\n")
if (file.exists("run_level_results.csv")) file.remove("run_level_results.csv")
if (file.exists("target_runner_errors.log")) file.remove("target_runner_errors.log")
writeLines(c("QT-LA PRE-FLIGHT SUCCESSFUL","instances = 72","blocks = 18 x 4","maxExperiments = 5000","irace seed = 12345"),"preflight_ok.txt")
cat("\nPRE-FLIGHT SUCCESSFUL: TRUE\n")
cat("Temporary target-runner rows were removed.\n")
cat('Next command: source("02_run_irace.R")\n')
