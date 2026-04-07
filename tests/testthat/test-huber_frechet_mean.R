## Tests for huber_frechet_mean() and huber_objective_value()

library(PowerFrechetMeanR)


# ==============================================================================
# 1. Return type and structure
# ==============================================================================

test_that("return value is a tibble with correct columns", {
  X <- matrix(rnorm(20), nrow = 10, ncol = 2)
  res <- huber_frechet_mean(X, delta = 1)
  expect_s3_class(res, "tbl_df")
  expect_named(res, c("delta", "mean", "value", "convergence", "n", "d"))
})

test_that("mean column is a k x d matrix", {
  set.seed(10)
  X <- matrix(rnorm(30), nrow = 10, ncol = 3)
  res <- huber_frechet_mean(X, delta = c(0.5, 1, 2))
  expect_true(is.matrix(res$mean))
  expect_equal(nrow(res$mean), 3L)
  expect_equal(ncol(res$mean), 3L)
})

test_that("mean column is k x 1 for univariate data", {
  x <- rnorm(10L)
  res <- huber_frechet_mean(x, delta = c(1, 2))
  expect_true(is.matrix(res$mean))
  expect_equal(nrow(res$mean), 2L)
  expect_equal(ncol(res$mean), 1L)
})

test_that("n and d fields are correct", {
  X <- matrix(1:20, nrow = 5, ncol = 4)
  storage.mode(X) <- "double"
  res <- huber_frechet_mean(X, delta = c(1, 2))
  expect_true(all(res$n == 5L))
  expect_true(all(res$d == 4L))
})


# ==============================================================================
# 2. Large delta -> arithmetic mean
# ==============================================================================

test_that("large delta gives arithmetic mean (multivariate)", {
  set.seed(1L)
  X <- matrix(rnorm(60L), nrow = 20L, ncol = 3L)
  # With delta = 1e6, all ||x_i - x|| << delta, so loss ~ r^2/2 -> colMeans
  res <- huber_frechet_mean(X, delta = 1e6)
  expect_equal(res$mean[1, ], colMeans(X), tolerance = 1e-5)
})

test_that("large delta gives arithmetic mean (1D)", {
  set.seed(2L)
  x <- rnorm(20L)
  res <- huber_frechet_mean(x, delta = 1e6)
  expect_equal(res$mean[1, 1], mean(x), tolerance = 1e-5)
})


# ==============================================================================
# 3. Vector of delta values
# ==============================================================================

test_that("vector of delta values returns one row each, sorted ascending", {
  set.seed(20)
  X <- matrix(rnorm(40), nrow = 20, ncol = 2)
  d_vec <- c(3, 0.5, 1, 2)
  res <- huber_frechet_mean(X, delta = d_vec)
  expect_equal(nrow(res), 4L)
  expect_equal(res$delta, sort(unique(d_vec)))
})

test_that("duplicate delta values are silently deduplicated", {
  set.seed(21)
  X <- matrix(rnorm(20), nrow = 10, ncol = 2)
  res <- huber_frechet_mean(X, delta = c(1, 1, 2, 2))
  expect_equal(nrow(res), 2L)
  expect_equal(res$delta, c(1, 2))
})


# ==============================================================================
# 4. Convergence
# ==============================================================================

test_that("convergence codes are 0 for a range of delta values", {
  set.seed(5L)
  X <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  res <- huber_frechet_mean(X, delta = c(0.1, 0.5, 1, 5, 100))
  expect_true(all(res$convergence == 0L))
})


# ==============================================================================
# 5. Minimality: returned point achieves a lower objective than colMeans
# ==============================================================================

test_that("Huber mean achieves lower objective than arithmetic mean", {
  set.seed(6L)
  X <- matrix(rnorm(40L), nrow = 20L, ncol = 2L)
  for (d_i in c(0.5, 1, 2)) {
    res <- huber_frechet_mean(X, delta = d_i)
    val_opt <- huber_objective_value(res$mean[1, ], X, delta = d_i)
    val_cm  <- huber_objective_value(colMeans(X),   X, delta = d_i)
    expect_lte(val_opt, val_cm + 1e-6,
               label = sprintf("minimality for delta=%g", d_i))
  }
})


# ==============================================================================
# 6. Symmetry
# ==============================================================================

test_that("Huber mean of symmetric data is the centre", {
  X <- matrix(c(1, 0, -1, 0,
                0, 1,  0, -1), nrow = 4L)
  for (d_i in c(0.5, 1, 5)) {
    res <- huber_frechet_mean(X, delta = d_i, tol = 1e-12)
    expect_equal(res$mean[1, ], c(0, 0), tolerance = 1e-5,
                 label = sprintf("symmetry for delta=%g", d_i))
  }
})


# ==============================================================================
# 7. Objective consistency
# ==============================================================================

test_that("objective values match huber_objective_value for all rows", {
  set.seed(7L)
  X <- matrix(rnorm(30L), nrow = 10L, ncol = 3L)
  res <- huber_frechet_mean(X, delta = c(0.5, 1, 5))
  for (i in seq_len(nrow(res))) {
    computed <- huber_objective_value(res$mean[i, ], X, delta = res$delta[i])
    expect_lt(abs(computed - res$value[i]), 1e-8)
  }
})

test_that("huber_objective_value basic sanity check", {
  # x = arithmetic mean; large delta => Huber ~ 0.5 * sum(||x_i - x||^2)
  set.seed(8L)
  X <- matrix(rnorm(20L), nrow = 10L, ncol = 2L)
  mu <- colMeans(X)
  val_huber <- huber_objective_value(mu, X, delta = 1e6)
  val_power <- sum(rowSums((X - matrix(mu, nrow(X), ncol(X), byrow = TRUE))^2)) / 2
  expect_lt(abs(val_huber - val_power), 1e-4)
})


# ==============================================================================
# 8. Error handling
# ==============================================================================

test_that("error when delta <= 0", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(huber_frechet_mean(X, delta = 0),  regexp = "> 0")
  expect_error(huber_frechet_mean(X, delta = -1), regexp = "> 0")
})

test_that("error when delta vector contains a non-positive value", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(huber_frechet_mean(X, delta = c(1, 2, 0)), regexp = "> 0")
})

test_that("error for non-numeric X", {
  expect_error(huber_frechet_mean(matrix(letters[1:6], nrow = 3L), delta = 1),
               regexp = "numeric")
})

test_that("error on missing values in X", {
  X <- matrix(rnorm(10L), nrow = 5L)
  X[2, 1] <- NA
  expect_error(huber_frechet_mean(X, delta = 1), regexp = "missing")
})

test_that("error for non-numeric delta", {
  X <- matrix(rnorm(10L), nrow = 5L)
  expect_error(huber_frechet_mean(X, delta = "one"), regexp = "numeric")
})
