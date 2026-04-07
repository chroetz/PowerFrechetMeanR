## Tests for power_frechet_mean() and pfm_objective_value()

library(PowerFrechetMeanR)


# ==============================================================================
# 1. Return type and structure
# ==============================================================================

test_that("return value is a tibble with correct columns", {
  X <- matrix(rnorm(20), nrow = 10, ncol = 2)
  res <- power_frechet_mean(X, a = 2)
  expect_s3_class(res, "tbl_df")
  expect_named(res, c("a", "mean", "value", "convergence", "n", "d"))
})

test_that("n and d fields are correct", {
  X <- matrix(1:20, nrow = 5, ncol = 4)
  storage.mode(X) <- "double"
  res <- power_frechet_mean(X, a = c(1.5, 2))
  expect_true(all(res$n == 5L))
  expect_true(all(res$d == 4L))
})

test_that("mean column is a k x d matrix", {
  set.seed(10)
  X <- matrix(rnorm(30), nrow = 10, ncol = 3)
  res <- power_frechet_mean(X, a = c(1, 2, 3))
  expect_true(is.matrix(res$mean))
  expect_equal(nrow(res$mean), 3L)  # k = 3 a-values
  expect_equal(ncol(res$mean), 3L)  # d = 3 dimensions
})

test_that("mean column is k x 1 for univariate data", {
  x <- rnorm(10L)
  res <- power_frechet_mean(x, a = c(1, 2))
  expect_true(is.matrix(res$mean))
  expect_equal(nrow(res$mean), 2L)
  expect_equal(ncol(res$mean), 1L)
})


# ==============================================================================
# 2. a = 2 returns the arithmetic mean exactly
# ==============================================================================

test_that("a=2 returns arithmetic mean (1D)", {
  set.seed(1L)
  x <- rnorm(20L)
  res <- power_frechet_mean(matrix(x, ncol = 1L), a = 2)
  expect_equal(res$mean[1, 1], mean(x), tolerance = 1e-6)
  expect_equal(res$convergence, 0L)
})

test_that("a=2 returns arithmetic mean (multivariate)", {
  set.seed(2L)
  X <- matrix(rnorm(60L), nrow = 20L, ncol = 3L)
  res <- power_frechet_mean(X, a = 2)
  expect_equal(res$mean[1, ], colMeans(X), tolerance = 1e-6)
})

test_that("a=2 works for a single data point", {
  X <- matrix(c(3, -1, 7), nrow = 1L)
  res <- power_frechet_mean(X, a = 2)
  expect_equal(res$mean[1, ], c(3, -1, 7), tolerance = 1e-6)
})


# ==============================================================================
# 3. a = 1 (geometric median)
# ==============================================================================

test_that("a=1 converges for n=1", {
  X <- matrix(c(5, -2), nrow = 1L)
  res <- power_frechet_mean(X, a = 1)
  expect_equal(res$mean[1, ], c(5, -2), tolerance = 1e-5)
  expect_equal(res$convergence[1], 0L)
})

test_that("a=1 objective <= objective at arithmetic mean (2D)", {
  set.seed(3L)
  X <- matrix(rnorm(40L), nrow = 20L, ncol = 2L)
  res <- power_frechet_mean(X, a = 1)
  expect_lte(pfm_objective_value(res$mean[1, ], X, a = 1),
             pfm_objective_value(colMeans(X),   X, a = 1) + 1e-6)
})

test_that("a=1 1D optimised objective equals objective at exact median", {
  # For d=1 the L1 objective is minimised by any value in the median interval;
  # both the optimiser and median(x) achieve the same minimum value.
  set.seed(4L)
  x <- rnorm(30L)
  X <- matrix(x, ncol = 1L)
  res <- power_frechet_mean(X, a = 1)
  expect_equal(pfm_objective_value(res$mean[1, ], X, a = 1),
               pfm_objective_value(median(x),     X, a = 1),
               tolerance = 1e-5)
})


# ==============================================================================
# 4. Vector of a values
# ==============================================================================

test_that("vector of a values returns one row each, sorted", {
  set.seed(20)
  X <- matrix(rnorm(40), nrow = 20, ncol = 2)
  a_vec <- c(3, 1, 2, 1.5)
  res <- power_frechet_mean(X, a = a_vec)
  expect_equal(nrow(res), 4L)
  expect_equal(res$a, sort(unique(a_vec)))
})

test_that("duplicate a values are silently deduplicated", {
  set.seed(21)
  X <- matrix(rnorm(20), nrow = 10, ncol = 2)
  res <- power_frechet_mean(X, a = c(2, 2, 1, 1))
  expect_equal(nrow(res), 2L)
  expect_equal(res$a, c(1, 2))
})

test_that("fractional a in (0, 1) is accepted", {
  set.seed(30)
  X <- matrix(rnorm(20), nrow = 10, ncol = 2)
  # a = 0.5 is non-convex but should not error
  expect_no_error(power_frechet_mean(X, a = 0.5))
})


# ==============================================================================
# 5. Convergence for a range of power values
# ==============================================================================

test_that("convergence code is 0 for a range of a values (single calls)", {
  set.seed(5L)
  X <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  for (aa in c(1, 1.5, 2, 3, 4)) {
    res <- power_frechet_mean(X, a = aa)
    expect_equal(res$convergence[1], 0L,
                 label = sprintf("convergence for a=%g", aa))
  }
})

test_that("all convergence codes are 0 for vector a call", {
  set.seed(22)
  X <- matrix(rnorm(60), nrow = 20, ncol = 3)
  res <- power_frechet_mean(X, a = c(1, 1.5, 2, 2.5, 3, 5))
  expect_true(all(res$convergence == 0L))
})


# ==============================================================================
# 6. Symmetry: mean of symmetric data is the centre
# ==============================================================================

test_that("power Frechet mean of symmetric data is the centre", {
  X <- matrix(c(1, 0, -1, 0,
                0, 1,  0, -1), nrow = 4L)
  for (aa in c(1, 1.5, 2, 3)) {
    res <- power_frechet_mean(X, a = aa, tol = 1e-10)
    expect_equal(res$mean[1, ], c(0, 0), tolerance = 1e-5,
                 label = sprintf("symmetry for a=%g", aa))
  }
})


# ==============================================================================
# 7. pfm_objective_value() consistent with minimised value
# ==============================================================================

test_that("pfm_objective_value matches value at solution", {
  set.seed(6L)
  X   <- matrix(rnorm(20L), nrow = 5L, ncol = 4L)
  res <- power_frechet_mean(X, a = 1.5)
  expect_equal(pfm_objective_value(res$mean[1, ], X, a = 1.5),
               res$value[1], tolerance = 1e-10)
})

test_that("objective values are consistent for all rows in multi-a result", {
  set.seed(23)
  X <- matrix(rnorm(30), nrow = 10, ncol = 3)
  res <- power_frechet_mean(X, a = c(1, 2, 3))
  for (i in seq_len(nrow(res))) {
    computed <- pfm_objective_value(res$mean[i, ], X, a = res$a[i])
    expect_lt(abs(computed - res$value[i]), 1e-8)
  }
})

test_that("pfm_objective_value basic calculation", {
  X <- matrix(1:6, nrow = 3)
  storage.mode(X) <- "double"
  val <- pfm_objective_value(c(2, 3), X, a = 2)
  # sum of squared Euclidean distances from (2,3) to rows (1,4),(2,5),(3,6)
  expected <- sum((1-2)^2 + (4-3)^2) +
              sum((2-2)^2 + (5-3)^2) +
              sum((3-2)^2 + (6-3)^2)
  expect_lt(abs(val - expected), 1e-10)
})


# ==============================================================================
# 8. Input coercion and error handling
# ==============================================================================

test_that("numeric vector input is coerced to a column matrix", {
  x   <- rnorm(10L)
  res <- power_frechet_mean(x, a = 2)
  expect_equal(res$mean[1, 1], mean(x), tolerance = 1e-6)
  expect_equal(res$d[1], 1L)
})

test_that("error when a = 0", {
  expect_error(power_frechet_mean(matrix(rnorm(10L), nrow = 5L), a = 0),
               regexp = "> 0")
})

test_that("error when a is negative", {
  expect_error(power_frechet_mean(matrix(rnorm(10L), nrow = 5L), a = -1),
               regexp = "> 0")
})

test_that("error when a vector contains a non-positive value", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(power_frechet_mean(X, a = c(1, 2, 0)), regexp = "> 0")
})

test_that("error for non-numeric X", {
  expect_error(power_frechet_mean(matrix(letters[1:6], nrow = 3L)),
               regexp = "numeric")
})

test_that("error on missing values", {
  X <- matrix(rnorm(10L), nrow = 5L)
  X[2, 1] <- NA
  expect_error(power_frechet_mean(X, a = 2), regexp = "missing")
})

test_that("error for non-numeric a", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(power_frechet_mean(X, a = "two"), regexp = "numeric")
})
