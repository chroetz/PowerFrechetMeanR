#' Compute the Power Frechet Mean
#'
#' Given \eqn{n} data points \eqn{x_1, \dots, x_n \in \mathbb{R}^d} and a
#' power parameter \eqn{a \ge 1}, the **power Frechet mean** is the minimiser
#' of
#' \deqn{f(x) = \sum_{i=1}^{n} \|x_i - x\|^a, \quad x \in \mathbb{R}^d.}
#'
#' The function is strictly convex for \eqn{a > 1} (unique minimiser) and
#' convex for \eqn{a = 1} (geometric median, unique if the data are not
#' collinear).  Notable special cases:
#'
#' * **\eqn{a = 2}** -- arithmetic mean (`colMeans(X)`).
#' * **\eqn{a = 1}** -- geometric / spatial median (L1 mean).
#' * **\eqn{a \to \infty}** -- minimax / Chebyshev centre.
#'
#' Optimisation uses an analytical gradient and either the L-BFGS-B
#' quasi-Newton method from [stats::optim()] (`backend = "optim"`, default)
#' or the LBFGS algorithm from [nloptr::nloptr()] (`backend = "nloptr"`).
#'
#' @param X       Numeric matrix of data points, one row per observation
#'   (\eqn{n \times d}).  A numeric vector is treated as an \eqn{n \times 1}
#'   matrix.
#' @param a       Power parameter \eqn{a \ge 1} (default `2`).
#' @param init    Initial guess for the optimiser, a numeric vector of length
#'   \eqn{d}.  Defaults to `colMeans(X)` (arithmetic mean), which is the
#'   exact solution for `a = 2` and a good warm start otherwise.
#'   Overridden by `weiszfeld_init` when that is positive.
#' @param backend Character string selecting the optimisation backend:
#'   `"optim"` (default, uses [stats::optim()] with method `"L-BFGS-B"`) or
#'   `"nloptr"` (uses [nloptr::nloptr()] with algorithm `NLOPT_LD_LBFGS`).
#'   If `"nloptr"` is requested but the package is not installed, the function
#'   falls back to `"optim"` with a message.
#' @param tol     Convergence tolerance passed to the optimiser.  For
#'   `"optim"` (L-BFGS-B) this sets `control$factr = tol / .Machine$double.eps`
#'   and `control$pgtol = tol`; for `"nloptr"` it sets `xtol_rel`.
#'   Default `1e-8`.
#' @param maxit   Maximum number of iterations (default `1000`).
#' @param eps     Smoothing constant used in the gradient when \eqn{a < 2}
#'   to prevent division by zero at data points.  Default `1e-10`.
#' @param use_cpp Logical.  If `TRUE` (default), use the C++/RcppArmadillo
#'   implementations of the objective and gradient for speed.  Set to
#'   `FALSE` to force the pure-R fallback (useful for benchmarking).
#' @param weiszfeld_init  Integer number of Weiszfeld (iteratively
#'   re-weighted least-squares) iterations to run before passing the result
#'   as the initial point for L-BFGS-B.  Useful for \eqn{a} near 1 where
#'   the Weiszfeld algorithm converges quickly and produces a much better
#'   starting point than the arithmetic mean.  Default `0L` (disabled).
#'   Ignored when `init` is supplied explicitly.
#' @param ...     Additional named arguments forwarded to the `control` list
#'   of [stats::optim()] (for `backend = "optim"`) or to the `opts` list of
#'   [nloptr::nloptr()] (for `backend = "nloptr"`).
#'
#' @return A list with class `"power_frechet_mean"` containing:
#'   \describe{
#'     \item{`mean`}{Numeric vector of length \eqn{d}: the power Frechet mean.}
#'     \item{`value`}{Numeric scalar: the minimised objective value \eqn{f(x^*)}.}
#'     \item{`convergence`}{Integer convergence code from the optimiser.
#'       `0` indicates successful convergence for both backends.}
#'     \item{`a`}{The power parameter used.}
#'     \item{`n`}{Number of data points.}
#'     \item{`d`}{Dimension of the data.}
#'     \item{`backend`}{The backend actually used.}
#'   }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(30), nrow = 10, ncol = 3)
#'
#' # a = 2: should equal arithmetic mean
#' res2 <- power_frechet_mean(X, a = 2)
#' stopifnot(max(abs(res2$mean - colMeans(X))) < 1e-6)
#'
#' # a = 1: geometric median (with Weiszfeld warm-start)
#' res1 <- power_frechet_mean(X, a = 1, weiszfeld_init = 20L)
#' res1$mean
#'
#' # a = 3
#' res3 <- power_frechet_mean(X, a = 3)
#' res3$mean
#'
#' @seealso [pfm_objective_value()] to evaluate the objective at an arbitrary
#'   point.
#' @export
power_frechet_mean <- function(X,
                               a              = 2,
                               init           = NULL,
                               backend        = c("optim", "nloptr"),
                               tol            = 1e-8,
                               maxit          = 1000L,
                               eps            = 1e-10,
                               use_cpp        = TRUE,
                               weiszfeld_init = 0L,
                               ...) {
  # ---- input validation -------------------------------------------------------
  if (is.vector(X) && is.numeric(X)) {
    X <- matrix(X, ncol = 1L)
  } else {
    X <- as.matrix(X)
  }
  if (!is.numeric(X))  stop("`X` must be a numeric matrix.")
  storage.mode(X) <- "double"   # ensure double, not integer
  if (anyNA(X))        stop("`X` contains missing values.")
  if (!is.numeric(a) || length(a) != 1L || a <= 0)
    stop("`a` must be a single numeric value > 0.")
  if (!is.numeric(eps) || eps < 0)
    stop("`eps` must be a non-negative numeric scalar.")
  weiszfeld_init <- as.integer(weiszfeld_init)

  n <- nrow(X)
  d <- ncol(X)

  backend <- match.arg(backend)

  # Fall back gracefully if nloptr is not installed
  if (backend == "nloptr" &&
      !requireNamespace("nloptr", quietly = TRUE)) {
    message("Package 'nloptr' is not installed; falling back to stats::optim.")
    backend <- "optim"
  }

  # ---- shortcut for a = 2 (arithmetic mean is exact) -------------------------
  # NOTE: this must come BEFORE the Weiszfeld init block so we do not waste
  # O(n*d*weiszfeld_init) work when the answer is just colMeans(X).
  if (isTRUE(all.equal(a, 2))) {
    mu <- colMeans(X)
    if (use_cpp) {
      val <- pfm_objective_cpp(mu, X, a)
    } else {
      val <- pfm_objective(mu, X, a)
    }
    return(structure(
      list(mean = mu, value = val, convergence = 0L,
           a = a, n = n, d = d, backend = "exact"),
      class = "power_frechet_mean"
    ))
  }

  # ---- initial point (after a=2 shortcut to avoid wasted Weiszfeld work) -----
  if (is.null(init)) {
    if (weiszfeld_init > 0L && use_cpp) {
      init <- as.numeric(weiszfeld_init_cpp(X, max_iter = weiszfeld_init,
                                            eps = eps))
    } else if (weiszfeld_init > 0L) {
      init <- weiszfeld_init_r(X, max_iter = weiszfeld_init, eps = eps)
    } else {
      init <- colMeans(X)
    }
  } else {
    init <- as.numeric(init)
    if (length(init) != d)
      stop("`init` must have length equal to the number of columns of `X`.")
  }

  # ---- choose objective / gradient implementation ----------------------------
  if (use_cpp) {
    fn <- function(x) pfm_objective_cpp(x, X, a)
    gr <- function(x) pfm_gradient_cpp(x, X, a, eps)
  } else {
    fn <- function(x) pfm_objective(x, X, a)
    gr <- function(x) pfm_gradient(x, X, a, eps)
  }

  # ---- optimise ---------------------------------------------------------------
  extra <- list(...)

  if (backend == "optim") {
    ctrl <- c(
      list(maxit = as.integer(maxit),
           factr = tol / .Machine$double.eps,
           pgtol = tol),
      extra
    )
    fit <- stats::optim(
      par     = init,
      fn      = fn,
      gr      = gr,
      method  = "L-BFGS-B",
      control = ctrl
    )
    result <- list(
      mean        = fit$par,
      value       = fit$value,
      convergence = fit$convergence,
      a           = a,
      n           = n,
      d           = d,
      backend     = "optim"
    )

  } else {
    opts <- c(
      list(algorithm = "NLOPT_LD_LBFGS",
           xtol_rel  = tol,
           maxeval   = maxit),
      extra
    )
    fit <- nloptr::nloptr(
      x0          = init,
      eval_f      = fn,
      eval_grad_f = gr,
      opts        = opts
    )
    conv <- if (fit$status > 0L) 0L else fit$status
    result <- list(
      mean        = fit$solution,
      value       = fit$objective,
      convergence = conv,
      a           = a,
      n           = n,
      d           = d,
      backend     = "nloptr"
    )
  }

  structure(result, class = "power_frechet_mean")
}


#' Print method for power_frechet_mean objects
#'
#' @param x   An object of class `"power_frechet_mean"`.
#' @param ... Ignored.
#' @return Invisibly returns `x`.
#' @export
print.power_frechet_mean <- function(x, ...) {
  cat(sprintf(
    "Power Fr\u00e9chet mean  (a = %g,  n = %d,  d = %d,  backend = %s)\n",
    x$a, x$n, x$d, x$backend
  ))
  cat("Mean:\n")
  print(x$mean)
  cat(sprintf("Objective value: %g\n", x$value))
  if (x$convergence != 0L)
    warning(sprintf("Optimiser did not converge (code %d).", x$convergence),
            call. = FALSE)
  invisible(x)
}


#' Evaluate the Power Frechet Objective at a Given Point
#'
#' Convenience function that evaluates
#' \eqn{f(x) = \sum_{i=1}^{n} \|x_i - x\|^a}
#' at an arbitrary point `x`.
#'
#' @param x  Numeric vector of length \eqn{d}.
#' @param X  Numeric \eqn{n \times d} matrix of data points.
#' @param a  Power parameter \eqn{a \ge 1} (default `2`).
#' @param use_cpp Logical.  Use C++ implementation if `TRUE` (default).
#'
#' @return Scalar numeric objective value.
#'
#' @examples
#' X <- matrix(1:6, nrow = 3)
#' pfm_objective_value(c(2, 3), X, a = 2)
#'
#' @export
pfm_objective_value <- function(x, X, a = 2, use_cpp = TRUE) {
  if (is.vector(X) && is.numeric(X)) X <- matrix(X, ncol = 1L)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  x <- as.numeric(x)
  if (use_cpp) {
    pfm_objective_cpp(x, X, a)
  } else {
    pfm_objective(x, X, a)
  }
}
