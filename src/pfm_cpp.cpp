// Power Frechet Mean -- C++/RcppArmadillo implementations.
// All matrix operations are BLAS-backed via Armadillo.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

// ---------------------------------------------------------------------------
// Objective:  f(x) = sum_{i=1}^n  ||x_i - x||^a
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
double pfm_objective_cpp(const arma::vec& x,
                         const arma::mat& X,
                         double a) {
  mat diffs = X.each_row() - x.t();           // n x d: row i = x_i - x
  vec norms = sqrt(sum(diffs % diffs, 1));     // n-vector of ||x_i - x||
  return accu(pow(norms, a));
}

// ---------------------------------------------------------------------------
// Gradient:  grad f(x) = -a * sum_i ||x_i - x||^{a-2} * (x_i - x)
//
// A smoothing constant eps prevents 0^(a-2) divergence when a < 2.
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
  return -diffs.t() * weights;                        // d
}
