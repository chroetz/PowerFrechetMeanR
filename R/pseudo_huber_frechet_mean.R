#' Compute the Pseudo-Huber Frechet Mean
#'
#' Given \eqn{n} data points \eqn{x_1, \dots, x_n \in \mathbb{R}^d} and one
#' or more threshold parameters \eqn{\delta > 0}, the **pseudo-Huber Frechet
#' mean** for each \eqn{\delta} is the minimiser of
#' \deqn{f(x) = \sum_{i=1}^{n} \rho_\delta(\|x_i - x\|),
#'   \quad x \in \mathbb{R}^d,}
#' where the pseudo-Huber loss is
#' \deqn{\rho_\delta(r) = \delta^2\!\left(\sqrt{1 + (r/\delta)^2} - 1\right).}
#'
#' The pseudo-Huber loss is a smooth (infinitely differentiable) approximation
#' to the Huber loss.  For large \eqn{\delta}, \eqn{\rho_\delta(r) \approx
#' r^2/2} and the minimiser approaches the arithmetic mean.  For small
#' \eqn{\delta}, \eqn{\rho_\delta(r) \approx \delta r} and the minimiser
#' approaches the geometric median.
#'
#' ## Warm-start strategy
#'
#' \eqn{\delta} values are sorted in **decreasing** order and solved
#' sequentially; each solve is warm-started from the previous result.  The
#' largest \eqn{\delta} is anchored at `colMeans(X)`, which is the exact
#' minimiser in the limiting case \eqn{\delta \to \infty}.
#'
#' ## Optimisation
#'
#' Uses an analytical gradient (C++/RcppArmadillo); the pseudo-Huber gradient
#' has no singularity and requires no smoothing constant.  Optimised with
#' L-BFGS-B from [stats::optim()].
#'
#' @param X      Numeric matrix of data points, one row per observation
#'   (\eqn{n \times d}).  A numeric vector is treated as an \eqn{n \times 1}
#'   matrix.
#' @param delta  Numeric vector of pseudo-Huber thresholds, each \eqn{> 0}.
#'   Duplicate values are silently removed; the returned tibble rows are
#'   sorted by `delta`.
#' @param tol    Convergence tolerance.  Sets `control$factr` and
#'   `control$pgtol` for L-BFGS-B.  Default `1e-10`.
#' @param maxit  Maximum number of L-BFGS-B iterations (default `1000`).
#' @param ...    Additional entries for the `control` list of [stats::optim()].
#'
#' @return A [tibble::tibble()] with one row per unique value of `delta`
#'   (sorted ascending), containing columns:
#'   \describe{
#'     \item{`delta`}{Pseudo-Huber threshold.}
#'     \item{`mean`}{Matrix column (\eqn{k \times d}); row \eqn{i} is the
#'       pseudo-Huber Frechet mean for `delta[i]`.  Access as
#'       `res$mean[i, ]`.}
#'     \item{`value`}{Minimised objective \eqn{f(x^*)}.}
#'     \item{`convergence`}{Integer code from [stats::optim()]; `0` = success.}
#'     \item{`n`}{Number of data points.}
#'     \item{`d`}{Dimension.}
#'   }
#'
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(30), nrow = 10, ncol = 3)
#'
#' # Single threshold: large delta -> arithmetic mean
#' res <- pseudo_huber_frechet_mean(X, delta = 1e6)
#' stopifnot(max(abs(res$mean[1, ] - colMeans(X))) < 1e-5)
#'
#' # Vector of thresholds
#' res_multi <- pseudo_huber_frechet_mean(X, delta = c(0.5, 1, 2, 5))
#' res_multi
#'
#' @seealso [pseudo_huber_objective_value()] to evaluate the objective at an
#'   arbitrary point; [huber_frechet_mean()] for the piecewise-linear Huber
#'   variant.
#' @export
pseudo_huber_frechet_mean <- function(X,
                                      delta,
                                      tol   = 1e-10,
                                      maxit = 1000L,
                                      ...) {
  # ---- input validation -------------------------------------------------------
  if (is.vector(X) && is.numeric(X)) {
    X <- matrix(X, ncol = 1L)
  } else {
    X <- as.matrix(X)
  }
  if (!is.numeric(X)) stop("`X` must be a numeric matrix.")
  storage.mode(X) <- "double"
  if (anyNA(X))       stop("`X` contains missing values.")

  if (!is.numeric(delta) || length(delta) < 1L)
    stop("`delta` must be a non-empty numeric vector.")
  if (any(delta <= 0))
    stop("All values of `delta` must be > 0.")

  n <- nrow(X)
  d <- ncol(X)

  # ---- sort unique delta values (descending for warm-start) -------------------
  delta_vals <- sort(unique(as.numeric(delta)), decreasing = TRUE)

  # ---- anchor: arithmetic mean is exact for delta -> infinity -----------------
  mu <- colMeans(X)

  # ---- L-BFGS-B control list --------------------------------------------------
  ctrl <- c(
    list(maxit = as.integer(maxit),
         factr = tol / .Machine$double.eps,
         pgtol = tol),
    list(...)
  )

  # ---- sweep from large delta to small, warm-starting each from previous ------
  results <- vector("list", length(delta_vals))
  init    <- mu
  for (i in seq_along(delta_vals)) {
    d_i <- delta_vals[i]
    fit <- stats::optim(
      par     = init,
      fn      = function(x) pseudo_huber_objective_cpp(x, X, d_i),
      gr      = function(x) pseudo_huber_gradient_cpp(x, X, d_i),
      method  = "L-BFGS-B",
      control = ctrl
    )
    results[[i]] <- list(delta       = d_i,
                         mean        = fit$par,
                         value       = fit$value,
                         convergence = as.integer(fit$convergence))
    init <- fit$par
  }

  # ---- return tibble sorted by delta (ascending) ------------------------------
  results  <- rev(results)   # reverse: was descending, now ascending
  mean_mat <- do.call(rbind, lapply(results, `[[`, "mean"))

  tibble::tibble(
    delta       = vapply(results, `[[`, numeric(1), "delta"),
    mean        = mean_mat,
    value       = vapply(results, `[[`, numeric(1), "value"),
    convergence = vapply(results, `[[`, integer(1), "convergence"),
    n           = n,
    d           = d
  )
}


#' Evaluate the Pseudo-Huber Frechet Objective at a Given Point
#'
#' Evaluates \eqn{f(x) = \sum_{i=1}^{n} \rho_\delta(\|x_i - x\|)} at an
#' arbitrary point `x`, where \eqn{\rho_\delta(r) = \delta^2(\sqrt{1 +
#' (r/\delta)^2} - 1)} is the pseudo-Huber loss.
#'
#' @param x      Numeric vector of length \eqn{d}.
#' @param X      Numeric \eqn{n \times d} matrix of data points.
#' @param delta  Pseudo-Huber threshold \eqn{\delta > 0}.
#'
#' @return Scalar numeric objective value.
#'
#' @examples
#' X <- matrix(1:6, nrow = 3)
#' pseudo_huber_objective_value(c(2, 3), X, delta = 1)
#'
#' @seealso [pseudo_huber_frechet_mean()]
#' @export
pseudo_huber_objective_value <- function(x, X, delta) {
  if (is.vector(X) && is.numeric(X)) X <- matrix(X, ncol = 1L)
  X <- as.matrix(X)
  storage.mode(X) <- "double"
  pseudo_huber_objective_cpp(as.numeric(x), X, delta)
}
