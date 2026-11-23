library(irace)

EXPECTED_DIR <- "irace_ils_with_PR"

current_dir <- basename(
  normalizePath(getwd(), winslash = "/", mustWork = TRUE)
)

if (current_dir != EXPECTED_DIR) {
  stop(paste0("Set the RStudio working directory to ", EXPECTED_DIR, "."))
}

required_files <- c(
  "scenario.txt",
  "parameters.txt",
  "initial_configuration.txt",
  "instances_calibration.txt",
  "manifest_calibration.csv",
  "frozen_parameters.txt",
  "target_runner_pr.py"
)

missing_files <- required_files[!file.exists(required_files)]
if (length(missing_files) > 0) {
  stop(paste0("Missing file(s): ", paste(missing_files, collapse = ", ")))
}

PROJECT_ROOT <- normalizePath(file.path(getwd(), "..", ".."), winslash = "/", mustWork = TRUE)
ILS_SOURCE <- file.path(PROJECT_ROOT, "src", "mnl_ils.py")
INSTANCE_DIR <- file.path(PROJECT_ROOT, "instances", "instances_seed_11_12_13_N60_100_200_400")

if (!file.exists(ILS_SOURCE)) stop("ILS source not found: ", ILS_SOURCE)
if (!dir.exists(INSTANCE_DIR)) stop("Calibration instance directory not found: ", INSTANCE_DIR)

instance_lines <- readLines("instances_calibration.txt", warn = FALSE)
instance_lines <- instance_lines[nzchar(trimws(instance_lines))]

if (length(instance_lines) != 72) {
  stop(
    paste0(
      "instances_calibration.txt must contain exactly 72 instances; found ",
      length(instance_lines), "."
    )
  )
}

# Preserve the original 72-instance ordering, including four-group blocking.
# Accept historical relative path strings in the list but resolve their filenames
# against the official calibration instance folder.
instance_names <- basename(gsub("\\\\", "/", instance_lines))
if (anyDuplicated(instance_names)) stop("Duplicate calibration instance names.")
if (any(!grepl("\\.npz$", instance_names, ignore.case = TRUE))) {
  stop("Each calibration instance entry must identify an .npz file.")
}
missing_instances <- instance_names[
  !file.exists(file.path(INSTANCE_DIR, instance_names))
]

if (length(missing_instances) > 0) {
  stop(
    paste0(
      "Missing calibration instance(s): ",
      paste(missing_instances, collapse = ", ")
    )
  )
}

init <- read.table(
  "initial_configuration.txt",
  header = TRUE,
  stringsAsFactors = FALSE
)

if (nrow(init) != 8) {
  stop("initial_configuration.txt must contain exactly 8 PR configurations.")
}

if (!setequal(unique(init$prTriggerNoImprove), c(1, 2, 3, 5))) {
  stop("Unexpected prTriggerNoImprove values.")
}

if (!setequal(unique(init$prBidirectional), c(0, 1))) {
  stop("Unexpected prBidirectional values.")
}

if (file.exists("run_level_results.csv")) {
  stop(
    paste0(
      "run_level_results.csv already exists before the pre-flight. ",
      "Archive/delete it and run this script again."
    )
  )
}

scenario <- readScenario("scenario.txt")

cat("\n")
cat("========================================\n")
cat("PRE-FLIGHT — PR CALIBRATION FOR ILvAS1\n")
cat("========================================\n")
cat("Frozen baseline: ILvAS1 (A + S1, BI)\n")
cat("PR-S2: disabled (path_relink_s2_samples = 0)\n")
cat("Free PR parameters: 2\n")
cat("  prTriggerNoImprove = {1,2,3,5}\n")
cat("  prBidirectional    = {0,1}\n")
cat("Configurations: 8\n")
cat("Instances: 72\n")
cat("Blocks: 18 x 4 size groups\n")
cat("The technical check executes the target runner three times.\n")
cat("Temporary run-level rows will be removed afterwards.\n\n")

ok <- checkIraceScenario(scenario)

if (!isTRUE(ok)) {
  stop("Pre-flight did not return TRUE.")
}

if (file.exists("run_level_results.csv")) {
  removed <- file.remove("run_level_results.csv")
  if (!isTRUE(removed)) {
    stop("Pre-flight passed, but run_level_results.csv could not be removed.")
  }
}

if (file.exists("run_level_results.csv")) {
  stop("Safety check failed: temporary run_level_results.csv still exists.")
}

writeLines(
  c(
    paste("PRE-FLIGHT OK:", Sys.time()),
    paste("irace_version:", as.character(packageVersion("irace"))),
    paste("R_version:", R.version.string),
    "Baseline: frozen ILvAS1.",
    "PR-S2 disabled.",
    "Free parameters: prTriggerNoImprove and prBidirectional.",
    "Design: 72 instances, 18 blocks x 4 size groups.",
    "Temporary target-runner rows removed after checkIraceScenario()."
  ),
  "preflight_ok.txt"
)

cat("\nPRE-FLIGHT SUCCESSFUL: TRUE\n")
cat("Temporary target-runner rows were removed.\n")
cat("No run_level_results.csv is present.\n")
cat("Next command: source(\"02_run_irace.R\")\n")
