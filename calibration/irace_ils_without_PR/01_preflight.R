library(irace)

EXPECTED_DIR <- "irace_ils_without_PR"

current_dir <- basename(
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
)

if (current_dir != EXPECTED_DIR) {
  stop(
    paste0(
      "Set the RStudio working directory to ",
      EXPECTED_DIR,
      "."
    )
  )
}

required_files <- c(
  "scenario.txt",
  "parameters.txt",
  "initial_configuration.txt",
  "instances_calibration.txt",
  "manifest_calibration.csv",
  "target_runner_ils.py"
)

missing_files <- required_files[!file.exists(required_files)]

if (length(missing_files) > 0) {
  stop(
    paste0(
      "Missing file(s): ",
      paste(missing_files, collapse = ", ")
    )
  )
}

PROJECT_ROOT <- normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = TRUE)
ILS_SOURCE <- file.path(PROJECT_ROOT, "src", "mnl_ils.py")
INSTANCE_DIR <- file.path(PROJECT_ROOT, "instances", "instances_seed_11_12_13_N60_100_200_400")
if (!file.exists(ILS_SOURCE)) stop("ILS source not found: ", ILS_SOURCE)
if (!dir.exists(INSTANCE_DIR)) stop("Calibration instance folder not found: ", INSTANCE_DIR)

instance_lines <- readLines(
  "instances_calibration.txt",
  warn = FALSE
)
instance_lines <- instance_lines[
  nzchar(trimws(instance_lines))
]

if (length(instance_lines) != 72) {
  stop(
    paste0(
      "instances_calibration.txt must contain exactly 72 instances; found ",
      length(instance_lines),
      "."
    )
  )
}

instance_names <- basename(gsub("\\\\", "/", trimws(instance_lines)))
if (anyDuplicated(instance_names)) stop("Duplicated instance names in instance list.")
if (any(!grepl("\\.npz$", instance_names, ignore.case=TRUE))) stop("Expected only .npz filenames.")
missing_instances <- instance_names[!file.exists(file.path(INSTANCE_DIR, instance_names))]
if (length(missing_instances)) stop("Missing calibration instances: ", paste(missing_instances, collapse=", "))

# Do not preflight in the historical folder containing published calibration outputs.
historical_outputs <- c("irace_ils_calibration_fi_bi_4groups.Rdata", "final_elites.csv", "run_level_results.csv")
if (any(file.exists(historical_outputs))) {
  stop("Historical calibration outputs present. Use a clean copy for reproduction; do not delete originals.")
}

if (file.exists("run_level_results.csv")) {
  stop(
    paste0(
      "run_level_results.csv already exists before the pre-flight. ",
      "Delete it and run this script again."
    )
  )
}

scenario <- readScenario("scenario.txt")

cat("\n")
cat("========================================\n")
cat("PRE-FLIGHT - ILS BI/FI, FOUR N GROUPS\n")
cat("========================================\n")
cat("Instances: 72\n")
cat("Blocks: 18 x 4\n")
cat("N groups: 60, 100, 200, 400\n")
cat("The technical check executes the target runner three times.\n")
cat("Temporary run-level rows will be removed after the check.\n")
cat("\n")

ok <- checkIraceScenario(scenario)

if (!isTRUE(ok)) {
  stop("Pre-flight did not return TRUE.")
}

# checkIraceScenario() executes the target runner.
# Because the folder was required to be clean before the check,
# any run_level_results.csv created here contains only temporary
# pre-flight executions and can be safely removed.
if (file.exists("run_level_results.csv")) {
  removed <- file.remove("run_level_results.csv")

  if (!isTRUE(removed)) {
    stop(
      paste0(
        "Pre-flight passed, but the temporary ",
        "run_level_results.csv could not be removed."
      )
    )
  }
}

if (file.exists("run_level_results.csv")) {
  stop(
    paste0(
      "Safety check failed: temporary ",
      "run_level_results.csv still exists."
    )
  )
}

preflight_info <- c(
  paste("PRE-FLIGHT OK:", Sys.time()),
  paste(
    "irace_version:",
    as.character(packageVersion("irace"))
  ),
  paste(
    "R_version:",
    R.version.string
  ),
  "Design: 72 instances, 18 blocks x 4.",
  "N groups: 60, 100, 200, 400.",
  "FI/BI parameter: firstImprovement = 0/1.",
  "Temporary target-runner rows removed after checkIraceScenario()."
)

writeLines(
  preflight_info,
  "preflight_ok.txt"
)

cat("\n")
cat("PRE-FLIGHT SUCCESSFUL: TRUE\n")
cat("Temporary target-runner rows were removed.\n")
cat("No run_level_results.csv is present.\n")
cat("Next command: source(\"02_run_irace.R\")\n")
