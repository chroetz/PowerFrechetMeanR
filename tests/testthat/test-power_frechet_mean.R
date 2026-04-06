## Tests for power_frechet_mean() and pfm_objective_value()

library(PowerFrechetMeanR)


# ==============================================================================
# 1. a = 2 returns the arithmetic mean exactly
# ==============================================================================

test_that("a=2 returns arithmetic mean (1D)", {
  set.seed(1L)
  x <- rnorm(20L)
  X <- matrix(x, ncol = 1L)
  res <- power_frechet_mean(X, a = 2)
  expect_equal(res$mean, mean(x), tolerance = 1e-6)
  expect_equal(res$convergence, 0L)
})

test_that("a=2 returns arithmetic mean (multivariate)", {
  set.seed(2L)
  X   <- matrix(rnorm(60L), nrow = 20L, ncol = 3L)
  res <- power_frechet_mean(X, a = 2)
  expect_equal(res$mean, colMeans(X), tolerance = 1e-6)
})

test_that("a=2 works for a single data point", {
  X   <- matrix(c(3, -1, 7), nrow = 1L)
  res <- power_frechet_mean(X, a = 2)
  expect_equal(res$mean, c(3, -1, 7), tolerance = 1e-6)
})


# ==============================================================================
# 2. a = 1 (geometric median) -- basic sanity checks
# ==============================================================================

test_that("a=1 converges and returns a single data point for n=1", {
  X   <- matrix(c(5, -2), nrow = 1L)
  res <- power_frechet_mean(X, a = 1)
  expect_equal(res$mean, c(5, -2), tolerance = 1e-5)
  expect_equal(res$convergence, 0L)
})

test_that("a=1 objective <= a=1 objective at arithmetic mean (2D)", {
  set.seed(3L)
  X    <- matrix(rnorm(40L), nrow = 20L, ncol = 2L)
  res  <- power_frechet_mean(X, a = 1)
  f_at_mean   <- pfm_objective_value(colMeans(X), X, a = 1)
  f_at_median <- pfm_objective_value(res$mean,    X, a = 1)
  expect_lte(f_at_median, f_at_mean + 1e-6)
})

test_that("a=1 1D optimised objective equals objective at exact median", {
  set.seed(4L)
  x   <- rnorm(30L)
  X   <- matrix(x, ncol = 1L)
  res <- power_frechet_mean(X, a = 1)
  f_opt <- pfm_objective_value(res$mean,   X, a = 1)
  f_med <- pfm_objective_value(median(x),  X, a = 1)
  expect_equal(f_opt, f_med, tolerance = 1e-5)
})


# ==============================================================================
# 3. Convergence holds across different a values
# ==============================================================================

test_that("convergence code is 0 for a range of a values", {
  set.seed(5L)
  X <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  for (aa in c(1, 1.5, 2, 3, 4)) {
    res <- power_frechet_mean(X, a = aa)
    expect_equal(res$convergence, 0L,
                 label = sprintf("convergence for a=%g", aa))
  }
})


# ==============================================================================
# 4. Symmetry: mean of symmetric data is the centre of symmetry
# ==============================================================================

test_that("Power Frechet mean of symmetric data is the centre (2D)", {
  X <- matrix(c(1, 0, -1, 0,
                0, 1,  0, -1), nrow = 4L)
  for (aa in c(1, 1.5, 2, 3)) {
    res <- power_frechet_mean(X, a = aa, tol = 1e-10)
    expect_equal(res$mean, c(0, 0), tolerance = 1e-5,
                 label = sprintf("symmetry centre for a=%g", aa))
  }
})


# ==============================================================================
# 5. pfm_objective_value() is consistent with the minimised value
# ==============================================================================

test_that("pfm_objective_value matches internal objective at solution", {
  set.seed(6L)
  X   <- matrix(rnorm(20L), nrow = 5L, ncol = 4L)
  res <- power_frechet_mean(X, a = 1.5)
  ext <- pfm_objective_value(res$mean, X, a = 1.5)
  expect_equal(ext, res$value, tolerance = 1e-10)
})


# ==============================================================================
# 6. Input coercion and error handling
# ==============================================================================

test_that("numeric vector input is coerced to matrix", {
  x   <- rnorm(10L)
  res <- power_frechet_mean(x, a = 2)
  expect_equal(res$mean, mean(x), tolerance = 1e-6)
  expect_equal(res$d, 1L)
})

test_that("error when a < 1", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(power_frechet_mean(X, a = 0.5), regexp = ">= 1")
})

test_that("error for non-numeric X", {
  X <- matrix(letters[1:6], nrow = 3L)
  expect_error(power_frechet_mean(X), regexp = "numeric")
})

test_that("error when init has wrong length", {
  X <- matrix(rnorm(20L), nrow = 10L, ncol = 2L)
  expect_error(power_frechet_mean(X, init = c(1, 2, 3)), regexp = "length")
})

test_that("error on missing values", {
  X    <- matrix(rnorm(10L), nrow = 5L)
  X[2, 1] <- NA
  expect_error(power_frechet_mean(X), regexp = "missing")
})


# ==============================================================================
# 7. Return object structure
# ==============================================================================

test_that("return value has correct structure and class", {
  X   <- matrix(rnorm(12L), nrow = 4L, ncol = 3L)
  res <- power_frechet_mean(X, a = 2)
  expect_s3_class(res, "power_frechet_mean")
  expect_named(res, c("mean", "value", "convergence", "a", "n", "d", "backend"))
  expect_equal(res$n, 4L)
  expect_equal(res$d, 3L)
  expect_equal(res$a, 2)
})


# ==============================================================================
# 8. C++ vs R implementations agree
# ==============================================================================

test_that("C++ and R objective agree", {
  set.seed(8L)
  X  <- matrix(rnorm(50L), nrow = 10L, ncol = 5L)
  x0 <- colMeans(X) + rnorm(5L) * 0.01
  for (aa in c(1, 1.5, 2, 3)) {
    r_val  <- pfm_objective_value(x0, X, aa, use_cpp = FALSE)
    c_val  <- pfm_objective_value(x0, X, aa, use_cpp = TRUE)
    expect_equal(r_val, c_val, tolerance = 1e-10,
                 label = sprintf("obj R vs C++ for a=%g", aa))
  }
})

test_that("C++ and R gradient agree", {
  set.seed(9L)
  X  <- matrix(rnorm(50L), nrow = 10L, ncol = 5L)
  x0 <- colMeans(X) + rnorm(5L) * 0.01
  pfm_gr_r <- PowerFrechetMeanR:::pfm_gradient
  pfm_gr_c <- PowerFrechetMeanR:::pfm_gradient_cpp
  for (aa in c(1, 1.5, 2, 3)) {
    r_gr <- pfm_gr_r(x0, X, aa)
    c_gr <- as.numeric(pfm_gr_c(x0, X, aa))
    expect_equal(r_gr, c_gr, tolerance = 1e-10,
                 label = sprintf("grad R vs C++ for a=%g", aa))
  }
})

test_that("use_cpp=FALSE and use_cpp=TRUE give same optimised result", {
  set.seed(10L)
  X <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  for (aa in c(1, 1.5, 3)) {
    res_r <- power_frechet_mean(X, a = aa, use_cpp = FALSE)
    res_c <- power_frechet_mean(X, a = aa, use_cpp = TRUE)
    expect_equal(res_r$value, res_c$value, tolerance = 1e-6,
                 label = sprintf("R vs C++ optimised value for a=%g", aa))
    expect_equal(res_r$mean, res_c$mean, tolerance = 1e-5,
                 label = sprintf("R vs C++ optimised mean for a=%g", aa))
  }
})


# ==============================================================================
# 9. Weiszfeld warm-start
# ==============================================================================

test_that("Weiszfeld warm-start produces valid initial point for a=1", {
  set.seed(11L)
  X <- matrix(rnorm(60L), nrow = 20L, ncol = 3L)
  # With warm-start, the objective should be at least as good as without
  res_plain <- power_frechet_mean(X, a = 1, weiszfeld_init = 0L)
  res_warm  <- power_frechet_mean(X, a = 1, weiszfeld_init = 50L)
  expect_lte(res_warm$value, res_plain$value + 1e-6)
  expect_equal(res_warm$convergence, 0L)
})

test_that("Weiszfeld R fallback works when use_cpp=FALSE", {
  set.seed(12L)
  X <- matrix(rnorm(40L), nrow = 10L, ncol = 4L)
  res <- power_frechet_mean(X, a = 1, use_cpp = FALSE, weiszfeld_init = 20L)
  expect_equal(res$convergence, 0L)
  # Should be a valid solution (objective <= obj at arithmetic mean)
  f_mean <- pfm_objective_value(colMeans(X), X, a = 1, use_cpp = FALSE)
  expect_lte(res$value, f_mean + 1e-6)
})


# ==============================================================================
# 10. nloptr backend (skipped if not installed)
# ==============================================================================

test_that("nloptr backend gives same result as optim backend", {
  skip_if_not_installed("nloptr")
  set.seed(7L)
  X      <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  res_o  <- power_frechet_mean(X, a = 1.5, backend = "optim",   tol = 1e-9)
  res_n  <- power_frechet_mean(X, a = 1.5, backend = "nloptr",  tol = 1e-9)
  expect_equal(res_o$mean, res_n$mean, tolerance = 1e-5)
  expect_equal(res_n$backend, "nloptr")
})

test_that("nloptr backend falls back to optim when nloptr not installed", {
  local_mocked_bindings(
    requireNamespace = function(pkg, ...) FALSE,
    .package = "base"
  )
  X   <- matrix(rnorm(10L), nrow = 5L)
  res <- power_frechet_mean(X, a = 2, backend = "nloptr")
  expect_equal(res$backend, "exact")   # a=2 shortcut
})
