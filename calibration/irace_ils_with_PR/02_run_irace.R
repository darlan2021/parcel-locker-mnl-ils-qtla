library(irace)

EXPECTED_DIR <- "irace_ils_with_PR"

if (basename(normalizePath(getwd(), winslash="/", mustWork=TRUE)) != EXPECTED_DIR) {
  stop("Set the RStudio working directory to ", EXPECTED_DIR, ".")
}

required <- c(
  "scenario.txt",
  "parameters.txt",
  "initial_configuration.txt",
  "instances_calibration.txt",
  "manifest_calibration.csv",
  "frozen_parameters.txt",
  "target_runner_pr.py",
  "preflight_ok.txt"
)

missing <- required[!file.exists(required)]
if (length(missing) > 0) {
  stop("Missing file(s): ", paste(missing, collapse=", "))
}

old_outputs <- c(
  "irace_ils_pr_calibration.Rdata",
  "run_level_results.csv",
  "final_elites.csv",
  "all_configurations.csv",
  "experiments_matrix.csv",
  "instance_algorithm_seed_pairs.csv",
  "calibration_metadata.txt",
  "irace_summary.txt"
)

existing <- old_outputs[file.exists(old_outputs)]
if (length(existing) > 0) {
  stop(
    "Previous calibration output(s) already exist: ",
    paste(existing, collapse=", "),
    "\nArchive/remove them before starting a NEW official run."
  )
}

scenario <- readScenario("scenario.txt")

cat("\n========================================\n")
cat("OFFICIAL PR CALIBRATION — FROZEN ILvAS1\n")
cat("========================================\n")
cat("Frozen base: A + S1, BI\n")
cat("S2: disabled\n")
cat("PR-S2: disabled\n")
cat("Parameters calibrated: 2\n")
cat("Configurations in the discrete domain: 8\n")
cat("Instances: 72 (18 blocks x 4 groups)\n")
cat("maxExperiments =", scenario$maxExperiments, "\n")
cat("irace seed     =", scenario$seed, "\n")
cat("blockSize      =", scenario$blockSize, "\n")
cat("testType       =", scenario$testType, "\n")
cat("\nStarting official calibration...\n\n")

start_time <- Sys.time()

timing <- system.time({
  final_elites <- irace(scenario=scenario)
})

end_time <- Sys.time()

cat("\n========================================\n")
cat("FINAL ELITE CONFIGURATION(S)\n")
cat("========================================\n")
print(final_elites)

write.csv(final_elites, "final_elites.csv", row.names=FALSE)

if (!file.exists("irace_ils_pr_calibration.Rdata")) {
  stop("Expected irace_ils_pr_calibration.Rdata was not created.")
}

load("irace_ils_pr_calibration.Rdata")

if (!exists("iraceResults")) {
  stop("Object 'iraceResults' was not found in irace_ils_pr_calibration.Rdata.")
}

write.csv(iraceResults$allConfigurations, "all_configurations.csv", row.names=FALSE)
write.csv(as.data.frame(iraceResults$experiments), "experiments_matrix.csv", row.names=TRUE)

pairs <- tryCatch(
  get_instanceID_seed_pairs(iraceResults),
  error=function(e) NULL
)

if (!is.null(pairs)) {
  write.csv(as.data.frame(pairs), "instance_algorithm_seed_pairs.csv", row.names=FALSE)
}

summary_txt <- tryCatch(
  capture.output(irace_summarise(iraceResults)),
  error=function(e) paste("irace_summarise unavailable:", conditionMessage(e))
)
writeLines(summary_txt, "irace_summary.txt")

meta <- c(
  paste("Start:", start_time),
  paste("End:", end_time),
  paste("Elapsed system.time:", paste(names(timing), timing, collapse="; ")),
  "Baseline: frozen ILvAS1.",
  "Only PR parameters were calibrated.",
  "prTriggerNoImprove domain: {1,2,3,5}.",
  "prBidirectional domain: {0,1}.",
  "path_relink_s2_samples = 0.",
  "Quality cost returned to irace = -C.",
  "CPU and PR effort are recorded in run_level_results.csv."
)
writeLines(meta, "calibration_metadata.txt")

cat("\nOFFICIAL PR CALIBRATION FINISHED\n")
cat("Main outputs:\n")
cat("  final_elites.csv\n")
cat("  run_level_results.csv\n")
cat("  all_configurations.csv\n")
cat("  experiments_matrix.csv\n")
cat("  irace_summary.txt\n")
