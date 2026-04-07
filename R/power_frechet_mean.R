#' Compute the Power Frechet Mean
#'
#' Given \eqn{n} data points \eqn{x_1, \dots, x_n \in \mathbb{R}^d} and one
#' or more power parameters \eqn{a > 0}, the **power Frechet mean** for each
#' value of \eqn{a} is the minimiser of
#' \deqn{f(x) = \sum_{i=1}^{n} \|x_i - x\|^a, \quad x \in \mathbb{R}^d.}
#'
#' The objective is convex for \eqn{a \ge 1} (strictly convex with a unique
#' minimiser for \eqn{a > 1}); for \eqn{0 < a < 1} it is non-convex and the
#' optimiser may converge to a local minimum.  Notable special cases:
#'
#' * **\eqn{a = 2}** -- arithmetic mean (`colMeans(X)`), computed exactly.
#' * **\eqn{a = 1}** -- geometric / spatial median (L1 mean).
#' * **\eqn{a \to \infty}** -- minimax / Chebyshev centre.
#'
#' ## Warm-start strategy
#'
#' Starting points for the optimiser are chosen automatically.  The arithmetic
#' mean (`colMeans(X)`) is the exact solution for \eqn{a = 2} and serves as
#' the anchor.  The supplied \eqn{a} values are sorted and the sweep proceeds
#' outward in both directions from 2:
#'
#' * **\eqn{a \ge 2}**: solved left-to-right in increasing order; each solve
#'   is warm-started from the previous result.
#' * **\eqn{a < 2}**: solved right-to-left in decreasing order (i.e. from
#'   values closest to 2 downward); each solve is warm-started from the
#'   previous result.
#'
#' This continuity-based strategy typically reduces total iterations
#' substantially compared to always starting from `colMeans(X)`.
#'
#' ## Optimisation
#'
#' Uses an analytical gradient (C++/RcppArmadillo) and the L-BFGS-B
#' quasi-Newton method from [stats::optim()].
#'
#' @param X       Numeric matrix of data points, one row per observation
#'   (\eqn{n \times d}).  A numeric vector is treated as an \eqn{n \times 1}
#'   matrix.
#' @param a       Numeric vector of power parameters, each \eqn{> 0}.
#'   Duplicate values are silently removed; the returned tibble rows are
#'   sorted by `a`.  The objective is convex for \eqn{a \ge 1}; values in
#'   \eqn{(0, 1)} are accepted but the problem is non-convex.
#' @param tol     Convergence tolerance.  Sets `control$factr` and
#'   `control$pgtol` for L-BFGS-B.  Default `1e-10`.
#' @param maxit   Maximum number of L-BFGS-B iterations (default `1000`).
#' @param eps     Smoothing constant for the gradient when \eqn{a < 2},
#'   preventing division by zero at data points.  Default `1e-12`.
#' @param ...     Additional entries for the `control` list of [stats::optim()].
#'
#' @return A [tibble::tibble()] with one row per unique value of `a` (sorted),
#'   containing columns:
#'   \describe{
#'     \item{`a`}{Power parameter.}
#'     \item{`mean`}{Matrix column (\eqn{k \times d}); row \eqn{i} is the
#'       power Frechet mean for `a[i]`.  Access as `res$mean[i, ]`.}
#'     \item{`value`}{Minimised objective \eqn{f(x^*)}.}
#'     \item{`convergence`}{Integer code from [stats::optim()]; `0` = success.}
#'     \item{`n`}{Number of data points.}
#'     \item{`d`}{Dimension.}
#'   }
#'
#' @examples
#' set.seed(42)
#' X <- matrix(rnorm(30), nrow = 10, ncol = 3)
#'
#' # Single a: a = 2 returns the arithmetic mean exactly
#' res <- power_frechet_mean(X, a = 2)
#' stopifnot(max(abs(res$mean[1, ] - colMeans(X))) < 1e-6)
#'
#' # Vector of a values
#' res_multi <- power_frechet_mean(X, a = c(1, 1.5, 2, 3))
#' res_multi
#'
#' @seealso [pfm_objective_value()] to evaluate the objective at an arbitrary
#'   point.
#' @export
power_frechet_mean <- function(X,
                               a,
                               tol   = 1e-10,
                               maxit = 1000L,
                               eps   = 1e-12,
                               ...) {
  # ---- input validation -------------------------------------------------------
  if (is.vector(X) && is.numeric(X)) {
    X <- matrix(X, ncol = 1L)
  } else {
    X <- as.matrix(X)
  }
  if (!is.numeric(X))  stop("`X` must be a numeric matrix.")
  storage.mode(X) <- "double"
  if (anyNA(X))        stop("`X` contains missing values.")

  if (!is.numeric(a) || length(a) < 1L)
    stop("`a` must be a non-empty numeric vector.")
  if (any(a <= 0))
    stop("All values of `a` must be > 0.")

  if (!is.numeric(eps) || length(eps) != 1L || eps < 0)
    stop("`eps` must be a non-negative numeric scalar.")

  n <- nrow(X)
  d <- ncol(X)

  # ---- sort and deduplicate a -------------------------------------------------
  a_vals <- sort(unique(as.numeric(a)))

  # ---- anchor: a = 2 is the arithmetic mean (exact) ---------------------------
  mu2  <- colMeans(X)
  val2 <- pfm_objective_cpp(mu2, X, 2.0)

  # ---- L-BFGS-B control list --------------------------------------------------
  ctrl <- c(
    list(maxit = as.integer(maxit),
         factr = tol / .Machine$double.eps,
         pgtol = tol),
    list(...)
  )

  # ---- helper: solve for a single a value -------------------------------------
  solve_one <- function(a_i, init) {
    if (isTRUE(all.equal(a_i, 2))) {
      return(list(par = mu2, value = val2, convergence = 0L))
    }
    fit <- stats::optim(
      par     = init,
      fn      = function(x) pfm_objective_cpp(x, X, a_i),
      gr      = function(x) pfm_gradient_cpp(x, X, a_i, eps),
      method  = "L-BFGS-B",
      control = ctrl
    )
    fit
  }

  # ---- sweep right: a >= 2 (sorted ascending) --------------------------------
  a_right <- a_vals[a_vals >= 2]
  results_right <- vector("list", length(a_right))
  init_r <- mu2
  for (i in seq_along(a_right)) {
    fit <- solve_one(a_right[i], init_r)
    results_right[[i]] <- list(a = a_right[i], mean = fit$par,
                               value = fit$value,
                               convergence = as.integer(fit$convergence))
    init_r <- fit$par
  }

  # ---- sweep left: a < 2 (sorted descending, i.e. closest to 2 first) --------
  a_left <- rev(a_vals[a_vals < 2])
  results_left <- vector("list", length(a_left))
  init_l <- mu2
  for (i in seq_along(a_left)) {
    fit <- solve_one(a_left[i], init_l)
    results_left[[i]] <- list(a = a_left[i], mean = fit$par,
                              value = fit$value,
                              convergence = as.integer(fit$convergence))
    init_l <- fit$par
  }

  # ---- combine results, sorted by a ------------------------------------------
  all_results <- c(results_left, results_right)
  # sort by a value
  ord <- order(vapply(all_results, `[[`, numeric(1), "a"))
  all_results <- all_results[ord]

  # Build k x d matrix: each row is the mean vector for one a value.
  # do.call(rbind, ...) handles d=1 correctly (gives k x 1 matrix).
  mean_mat <- do.call(rbind, lapply(all_results, `[[`, "mean"))

  tibble::tibble(
    a           = vapply(all_results, `[[`, numeric(1), "a"),
    mean        = mean_mat,
    value       = vapply(all_results, `[[`, numeric(1), "value"),
    convergence = vapply(all_results, `[[`, integer(1), "convergence"),
    n           = n,
    d           = d
  )
}


#' Evaluate the Power Frechet Objective at a Given Point
#'
#' Evaluates \eqn{f(x) = \sum_{i=1}^{n} \|x_i - x\|^a} at an arbitrary
#' point `x`.
#'
#' @param x  Numeric vector of length \eqn{d}.
#' @param X  Numeric \eqn{n \times d} matrix of data points.
#' @param a  Power parameter \eqn{a > 0} (default `2`).
#'
#' @return Scalar numeric objective value.
#'
#' @examples
#' X <- matrix(1:6, nrow = 3)
#' pfm_objective_value(c(2, 3), X, a = 2)
#'
#' @export
pfm_objective_value <- function(x, X, a = 2) {
  if (is.vector(X) && is.numeric(X)) X <- matrix(X, ncol = 1L)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  pfm_objective_cpp(as.numeric(x), X, a)
}