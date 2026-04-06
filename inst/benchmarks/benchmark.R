#!/usr/bin/env Rscript
# ============================================================================
# Benchmark script for PowerFrechetMeanR
#
# Compares the following implementation variants across a grid of
# (n, d, a) settings:
#
#   v1_R_optim          -- pure R objective+gradient, stats::optim backend
#   v2_cpp_optim        -- C++/Armadillo obj+grad, stats::optim backend
#   v3_cpp_nloptr       -- C++/Armadillo obj+grad, nloptr backend
#   v4_cpp_weiszfeld    -- C++ + Weiszfeld warm-start (20 iters), stats::optim
#
# Uses microbenchmark for sub-millisecond resolution timing.
#
# Usage:
#   Rscript inst/benchmarks/benchmark.R
#   Rscript inst/benchmarks/benchmark.R --reps 20   # more reps (stable)
#   Rscript inst/benchmarks/benchmark.R --reps 3    # fewer reps (fast)
#
# Results are saved to inst/benchmarks/benchmark_results.csv
# ============================================================================

library(PowerFrechetMeanR)
library(microbenchmark)

# ---------- configuration ----------------------------------------------------

ns    <- c(1e2, 1e3, 1e4)
ds    <- c(2L, 10L, 50L)
as    <- c(1, 1.5, 2, 3)
reps  <- 1e2       # microbenchmark reps per variant

args <- commandArgs(trailingOnly = TRUE)
if ("--reps" %in% args)
  reps <- as.integer(args[which(args == "--reps") + 1L])

has_nloptr <- requireNamespace("nloptr", quietly = TRUE)

cat(sprintf("PowerFrechetMeanR benchmark  (reps = %d)\n", reps=reps))
cat(sprintf("  nloptr available: %s\n", has_nloptr))
cat("------------------------------------------------------------\n")

fmt_ms  <- function(ms)  sprintf("%8.3f ms", ms)
fmt_x   <- function(r)   if (is.finite(r)) sprintf("%5.1fx", r) else "   --  "

# ---------- helpers ----------------------------------------------------------

# Run expr reps times via microbenchmark; return median in milliseconds.
mbm <- function(expr, reps) {
  expr_sub <- substitute(expr)
  mb <- microbenchmark(eval(expr_sub), times = reps, unit = "ms")
  median(mb$time) / 1e6
}

# ---------- main benchmark loop -----------------------------------------------

results <- data.frame(
  n = integer(0), d = integer(0), a = numeric(0),
  variant = character(0), median_ms = numeric(0), obj_val = numeric(0),
  stringsAsFactors = FALSE
)

total <- length(ns) * length(ds) * length(as)
count <- 0L

for (nn in ns) {
  for (dd in ds) {
    set.seed(42L)
    X <- matrix(rt(nn * dd, df=2), nrow = nn, ncol = dd)

    for (aa in as) {
      count <- count + 1L
      cat(sprintf("[%d/%d]  n = %6d,  d = %3d,  a = %g\n",
                  count, total, nn, dd, aa))

      # v1: pure R + optim
      t1 <- mbm(power_frechet_mean(X, a = aa, use_cpp = FALSE,
                                   backend = "optim"), reps=reps)
      v1 <- power_frechet_mean(X, a = aa, use_cpp = FALSE, backend = "optim")
      results <- rbind(results, data.frame(n=nn, d=dd, a=aa,
        variant="v1_R_optim", median_ms=t1, obj_val=v1$value))

      # v2: C++ + optim
      t2 <- mbm(power_frechet_mean(X, a = aa, use_cpp = TRUE,
                                   backend = "optim"), reps=reps)
      v2 <- power_frechet_mean(X, a = aa, use_cpp = TRUE, backend = "optim")
      results <- rbind(results, data.frame(n=nn, d=dd, a=aa,
        variant="v2_cpp_optim", median_ms=t2, obj_val=v2$value))

      # v3: C++ + nloptr
      if (has_nloptr) {
        t3 <- mbm(power_frechet_mean(X, a = aa, use_cpp = TRUE,
                                     backend = "nloptr"), reps=reps)
        v3 <- power_frechet_mean(X, a = aa, use_cpp = TRUE, backend = "nloptr")
        results <- rbind(results, data.frame(n=nn, d=dd, a=aa,
          variant="v3_cpp_nloptr", median_ms=t3, obj_val=v3$value))
      }

      # v4: C++ + Weiszfeld (only for a <= 2)
      t4 <- NA_real_
      if (aa <= 2) {
        t4 <- mbm(power_frechet_mean(X, a = aa, use_cpp = TRUE,
                                     backend = "optim",
                                     weiszfeld_init = 20L), reps=reps)
        v4 <- power_frechet_mean(X, a = aa, use_cpp = TRUE,
                                  backend = "optim", weiszfeld_init = 20L)
        results <- rbind(results, data.frame(n=nn, d=dd, a=aa,
          variant="v4_cpp_weiszfeld", median_ms=t4, obj_val=v4$value))
      }

      # Progress line
      cat(sprintf("  v1_R %s  v2_cpp %s  speedup %s",
                  fmt_ms(t1), fmt_ms(t2), fmt_x(t1 / t2)))
      if (has_nloptr) cat(sprintf("  v3_nloptr %s", fmt_ms(t3)))
      if (aa <= 2)    cat(sprintf("  v4_weisz %s", fmt_ms(t4)))
      cat("\n")
    }
  }
}

# ---------- summary table ----------------------------------------------------

cat("\n============================================================\n")
cat("RESULTS SUMMARY  (median milliseconds)\n")
cat("============================================================\n\n")

variants <- unique(results$variant)
header <- sprintf("%-20s", "")
for (v in variants) header <- paste0(header, sprintf(" %16s", v))
header <- paste0(header, sprintf(" %10s", "speedup"))
cat(header, "\n")
cat(strrep("-", nchar(header)), "\n")

for (nn in ns) {
  for (dd in ds) {
    for (aa in as) {
      sub <- results[results$n == nn & results$d == dd & results$a == aa, ]
      if (nrow(sub) == 0L) next
      row_label <- sprintf("n=%5d d=%3d a=%g", nn, dd, aa)
      line <- sprintf("%-20s", row_label)
      t_r   <- sub$median_ms[sub$variant == "v1_R_optim"]
      t_cpp <- sub$median_ms[sub$variant == "v2_cpp_optim"]
      for (v in variants) {
        t <- sub$median_ms[sub$variant == v]
        line <- paste0(line, if (length(t) == 1L)
          sprintf(" %16s", fmt_ms(t)) else sprintf(" %16s", "      --      "))
      }
      line <- paste0(line, sprintf(" %10s", fmt_x(t_r / t_cpp)))
      cat(line, "\n")
    }
  }
}

# ---------- save to CSV ------------------------------------------------------

out_file <- file.path(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE)[1])),
  "benchmark_results.csv")
if (is.na(out_file) || !nzchar(out_file))
  out_file <- "benchmark_results.csv"

tryCatch({
  write.csv(results, out_file, row.names = FALSE)
  cat(sprintf("\nResults saved to: %s\n", out_file))
}, error = function(e) {
  write.csv(results, "benchmark_results.csv", row.names = FALSE)
  cat("Results saved to: ./benchmark_results.csv\n")
})

cat("Done.\n")
