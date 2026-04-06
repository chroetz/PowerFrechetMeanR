# Internal helper functions for the power Frechet objective and its gradient.
# These are not exported; they are called by power_frechet_mean().
# The C++ equivalents in src/pfm_cpp.cpp are faster and used by default;
# these pure-R versions serve as a fallback (use_cpp = FALSE).

# pfm_objective(x, X, a)
#
# Evaluates f(x) = sum_{i=1}^n ||x_i - x||^a.
#
# x    : numeric vector of length d (current iterate)
# X    : numeric n x d matrix of data points
# a    : power parameter (a >= 1)
# Returns a scalar objective value.
pfm_objective <- function(x, X, a) {
  diffs <- sweep(X, 2L, x, `-`)          # n x d
  norms <- sqrt(.rowSums(diffs^2L, nrow(diffs), ncol(diffs)))  # n-vector
  sum(norms^a)
}

# pfm_gradient(x, X, a, eps)
#
# Evaluates grad_x f(x) = a * sum_{i=1}^n ||x_i - x||^{a-2} * (x - x_i).
#
# For a < 2 the weight ||x_i - x||^{a-2} diverges as x -> x_i.  A small
# smoothing constant eps is used to keep the gradient numerically finite.
#
# x    : numeric vector of length d (current iterate)
# X    : numeric n x d matrix of data points
# a    : power parameter (a >= 1)
# eps  : smoothing constant for the singular case a < 2 (default 1e-10)
# Returns a numeric vector of length d (gradient).
pfm_gradient <- function(x, X, a, eps = 1e-10) {
  diffs <- sweep(X, 2L, x, `-`)          # n x d: rows are (x_i - x)
  sq    <- .rowSums(diffs^2L, nrow(diffs), ncol(diffs))
  norms_smooth <- sqrt(sq + eps^2)        # n-vector, always > 0
  weights <- a * norms_smooth^(a - 2L)   # n-vector of scalar weights
  -.colSums(weights * diffs, nrow(diffs), ncol(diffs))
}

# weiszfeld_init_r(X, max_iter, eps)
#
# Pure-R implementation of Weiszfeld iteration for computing the geometric
# median (L1 mean).  Used as a warm-start when use_cpp = FALSE.
#
# X        : numeric n x d matrix
# max_iter : max Weiszfeld iterations
# eps      : smoothing constant to avoid zero division
# Returns a numeric vector of length d.
weiszfeld_init_r <- function(X, max_iter = 20L, eps = 1e-10) {
  m <- colMeans(X)
  n <- nrow(X)
  d <- ncol(X)
  for (iter in seq_len(max_iter)) {
    diffs  <- sweep(X, 2L, m, `-`)
    norms  <- sqrt(.rowSums(diffs^2L, n, d))
    w      <- 1.0 / pmax(norms, eps)
    wtot   <- sum(w)
    m_new  <- .colSums(w * X, n, d) / wtot
    if (sqrt(sum((m_new - m)^2)) < eps * 100) break
    m <- m_new
  }
  m
}
