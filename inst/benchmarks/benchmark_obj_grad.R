#!/usr/bin/env Rscript
# ============================================================================
# Micro-benchmark: R vs C++ objective and gradient evaluation
#
# This measures ONLY the objective/gradient functions (no optimiser overhead)
# to isolate the per-evaluation speedup from C++/Armadillo.
# ============================================================================

library(PowerFrechetMeanR)

if (!requireNamespace("microbenchmark", quietly = TRUE))
  stop("Please install the 'microbenchmark' package.")
library(microbenchmark)

cat("Micro-benchmark: R vs C++ objective/gradient evaluation\n")
cat("========================================================\n\n")

# Use internal R functions via :::
pfm_obj_r  <- PowerFrechetMeanR:::pfm_objective
pfm_gr_r   <- PowerFrechetMeanR:::pfm_gradient
pfm_obj_c  <- PowerFrechetMeanR:::pfm_objective_cpp
pfm_gr_c   <- PowerFrechetMeanR:::pfm_gradient_cpp

ns <- c(100L, 1000L, 10000L, 100000L)
ds <- c(2L, 10L, 50L, 200L)
a_vals <- c(1, 1.5, 3)

results <- list()

for (nn in ns) {
  for (dd in ds) {
    set.seed(42L)
    X  <- matrix(rnorm(nn * dd), nrow = nn, ncol = dd)
    x0 <- colMeans(X) + rnorm(dd) * 0.01

    for (aa in a_vals) {
      cat(sprintf("n = %6d, d = %3d, a = %g ... ", nn, dd, aa))

      # Number of microbenchmark iterations (fewer for large problems)
      times <- if (nn >= 10000L) 20L else 100L

      mb <- microbenchmark(
        R_objective   = pfm_obj_r(x0, X, aa),
        Cpp_objective = pfm_obj_c(x0, X, aa),
        R_gradient    = pfm_gr_r(x0, X, aa),
        Cpp_gradient  = pfm_gr_c(x0, X, aa),
        times = times,
        unit = "ms"
      )

      s <- summary(mb)
      cat(sprintf("obj: R=%.2fms  C++=%.2fms (%.1fx)  |  ",
                  s$median[1], s$median[2], s$median[1] / s$median[2]))
      cat(sprintf("grad: R=%.2fms  C++=%.2fms (%.1fx)\n",
                  s$median[3], s$median[4], s$median[3] / s$median[4]))

      # Correctness check
      r_obj <- pfm_obj_r(x0, X, aa)
      c_obj <- pfm_obj_c(x0, X, aa)
      r_gr  <- pfm_gr_r(x0, X, aa)
      c_gr  <- pfm_gr_c(x0, X, aa)

      if (abs(r_obj - c_obj) / max(abs(r_obj), 1) > 1e-10)
        warning(sprintf("Objective mismatch! R=%.12g C++=%.12g", r_obj, c_obj))
      if (max(abs(r_gr - c_gr)) / max(max(abs(r_gr)), 1) > 1e-10)
        warning(sprintf("Gradient mismatch! max|diff|=%.2e",
                        max(abs(r_gr - c_gr))))

      results[[length(results) + 1L]] <- data.frame(
        n = nn, d = dd, a = aa,
        R_obj_ms   = s$median[1],
        Cpp_obj_ms = s$median[2],
        obj_speedup = s$median[1] / s$median[2],
        R_gr_ms    = s$median[3],
        Cpp_gr_ms  = s$median[4],
        gr_speedup = s$median[3] / s$median[4]
      )
    }
  }
}

res_df <- do.call(rbind, results)

cat("\n========================================================\n")
cat("Summary table (median ms, speedup = R/C++)\n")
cat("========================================================\n\n")
print(res_df, digits = 3, row.names = FALSE)

tryCatch({
  write.csv(res_df, "benchmark_obj_grad_results.csv", row.names = FALSE)
  cat("\nResults saved to benchmark_obj_grad_results.csv\n")
}, error = function(e) message(e$message))

cat("\nDone.\n")
