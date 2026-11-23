# Descriptive summary of the PR calibration runs.
# Run only after 02_run_irace.R has finished.

if (!file.exists("run_level_results.csv")) {
  stop("run_level_results.csv not found. Run 02_run_irace.R first.")
}

x <- read.csv("run_level_results.csv", stringsAsFactors=FALSE)
ok <- x$feasible %in% c(TRUE, "True", "TRUE", 1, "1")
x <- x[ok, , drop=FALSE]

if (nrow(x) == 0) {
  stop("No feasible run rows found.")
}

key <- interaction(
  x$pr_trigger_no_improve,
  x$pr_bidirectional,
  drop=TRUE,
  sep="_"
)

groups <- split(x, key)

tab <- do.call(
  rbind,
  lapply(groups, function(g) {
    data.frame(
      prTriggerNoImprove = g$pr_trigger_no_improve[1],
      prBidirectional = g$pr_bidirectional[1],
      Runs = nrow(g),
      Mean_C = mean(g$objective_C),
      Median_C = median(g$objective_C),
      Mean_CPU_s = mean(g$cpu_seconds),
      Median_CPU_s = median(g$cpu_seconds),
      Mean_PR_calls = mean(g$path_relink_calls),
      Mean_PR_seconds = mean(g$path_relink_seconds),
      Mean_PR_steps = mean(g$PR_steps)
    )
  })
)

tab <- tab[order(-tab$Mean_C, tab$Mean_CPU_s), ]
row.names(tab) <- NULL

print(tab)
write.csv(tab, "pr_calibration_descriptive_summary.csv", row.names=FALSE)

cat("\nSaved: pr_calibration_descriptive_summary.csv\n")
cat("Because irace racing may evaluate configurations on different subsets,\n")
cat("this table is descriptive only. Use the irace race to identify finalists,\n")
cat("then validate finalists on the same instances before freezing the PR setup.\n")
