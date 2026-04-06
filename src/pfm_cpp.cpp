// Power Frechet Mean -- C++/RcppArmadillo implementations of the objective,
// gradient, combined obj+grad, and Weiszfeld warm-start.
//
// All matrix/vector operations are BLAS-backed via Armadillo so that large
// n x d problems benefit from optimised linear-algebra routines.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// ---------------------------------------------------------------------------
// Objective:  f(x) = sum_{i=1}^n  ||x_i - x||^a
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double pfm_objective_cpp(const arma::vec& x,
                         const arma::mat& X,
                         double a) {
  // diffs: n x d,  row i = x_i - x
  mat diffs = X.each_row() - x.t();
  vec norms = sqrt(sum(diffs % diffs, 1));   // n-vector of ||x_i - x||
  return accu(pow(norms, a));
}

// ---------------------------------------------------------------------------
// Gradient:  grad f(x) = -a * sum_i ||x_i - x||^{a-2} * (x_i - x)
//
// For a < 2, the weight ||x_i - x||^{a-2} diverges at coincident points.
// A smoothing constant eps keeps the gradient finite: we replace ||.|| by
// sqrt(||.||^2 + eps^2).
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
arma::vec pfm_gradient_cpp(const arma::vec& x,
                           const arma::mat& X,
                           double a,
                           double eps = 1e-10) {
  mat diffs    = X.each_row() - x.t();               // n x d
  vec norms_sq = sum(diffs % diffs, 1);               // n
  vec norms_sm = sqrt(norms_sq + eps * eps);           // smoothed norms
  vec weights  = a * pow(norms_sm, a - 2.0);          // n
  // grad = -sum_i w_i (x_i - x)  =  -diffs^T weights   (d x n)(n x 1) = d
  return -diffs.t() * weights;
}

// ---------------------------------------------------------------------------
// Combined objective + gradient (single pass over the data).
// Useful for optimisers that evaluate both in one call (e.g. nloptr
// with eval_f returning a list).
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
Rcpp::List pfm_obj_grad_cpp(const arma::vec& x,
                            const arma::mat& X,
                            double a,
                            double eps = 1e-10) {
  mat diffs    = X.each_row() - x.t();               // n x d
  vec norms_sq = sum(diffs % diffs, 1);               // n

  // objective (exact norms, no smoothing needed)
  vec norms    = sqrt(norms_sq);
  double obj   = accu(pow(norms, a));

  // gradient (smoothed norms for a < 2)
  vec norms_sm = sqrt(norms_sq + eps * eps);
  vec weights  = a * pow(norms_sm, a - 2.0);
  vec grad     = -diffs.t() * weights;

  return List::create(Named("objective") = obj,
                      Named("gradient")  = NumericVector(grad.begin(),
                                                         grad.end()));
}

// ---------------------------------------------------------------------------
// Weiszfeld iteration (iteratively re-weighted least squares) for the
// geometric-median problem (a = 1).  Returns a warm-start point that is
// already close to the L1-mean, so L-BFGS-B converges much faster for
// small a.
//
// The algorithm solves  min_m  sum_i ||x_i - m||  via
//   m_{k+1} = (sum_i w_i x_i) / (sum_i w_i),   w_i = 1 / ||x_i - m_k||.
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
arma::vec weiszfeld_init_cpp(const arma::mat& X,
                             int max_iter = 20,
                             double eps   = 1e-10) {
  vec m = mean(X, 0).t();                       // start at column means

  for (int iter = 0; iter < max_iter; ++iter) {
    mat diffs    = X.each_row() - m.t();
    vec norms    = sqrt(sum(diffs % diffs, 1));
    vec weights  = 1.0 / clamp(norms, eps, datum::inf);
    double wtot  = accu(weights);
    vec m_new    = X.t() * weights / wtot;       // d-vector

    if (norm(m_new - m, 2) < eps * 100.0) break;
    m = m_new;
  }
  return m;
}
