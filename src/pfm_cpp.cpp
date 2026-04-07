// Power / Huber / Pseudo-Huber Frechet Mean -- C++/RcppArmadillo implementations.
// All matrix operations are BLAS-backed via Armadillo.

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

// ===========================================================================
// Power Frechet Mean
// f(x) = sum_{i=1}^n  ||x_i - x||^a
// ===========================================================================

// [[Rcpp::export]]
double pfm_objective_cpp(const arma::vec& x,
                         const arma::mat& X,
                         double a) {
  mat diffs = X.each_row() - x.t();           // n x d: row i = x_i - x
  vec norms = sqrt(sum(diffs % diffs, 1));     // n-vector of ||x_i - x||
  return accu(pow(norms, a));
}

// Gradient:  grad f(x) = -a * sum_i ||x_i - x||^{a-2} * (x_i - x)
//
// A smoothing constant eps prevents 0^(a-2) divergence when a < 2.

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

// ===========================================================================
// Huber Frechet Mean
//
// Loss:  rho_H(r; delta) = r^2 / 2          if r <= delta
//                        = delta*r - delta^2/2  if r > delta
//
// Gradient weight: w_i = rho_H'(r_i) / r_i = 1       if r_i <= delta
//                                           = delta/r_i  if r_i > delta
//
// No smoothing constant needed: the r_i <= delta branch gives w_i = 1
// (not r-dependent), and the r_i > delta branch has r_i > delta > 0.
// ===========================================================================

// [[Rcpp::export]]
double huber_objective_cpp(const arma::vec& x,
                           const arma::mat& X,
                           double delta) {
  mat diffs   = X.each_row() - x.t();
  vec norms   = sqrt(sum(diffs % diffs, 1));
  vec clamped = clamp(norms, 0.0, delta);
  // rho(r) = 0.5*min(r,delta)^2 + delta*(r - min(r,delta))
  //        = 0.5*r^2            for r <= delta
  //        = delta*r - delta^2/2  for r > delta
  return accu(0.5 * clamped % clamped + delta * (norms - clamped));
}

// [[Rcpp::export]]
arma::vec huber_gradient_cpp(const arma::vec& x,
                             const arma::mat& X,
                             double delta) {
  mat  diffs   = X.each_row() - x.t();
  vec  norms   = sqrt(sum(diffs % diffs, 1));
  vec  weights(norms.n_elem, fill::ones);
  uvec big = find(norms > delta);
  if (big.n_elem > 0) {
    weights.elem(big) = delta / norms.elem(big);
  }
  return -diffs.t() * weights;
}

// ===========================================================================
// Pseudo-Huber Frechet Mean
//
// Loss:  rho_PH(r; delta) = delta^2 * (sqrt(1 + (r/delta)^2) - 1)
//
// Gradient weight: w_i = rho_PH'(r_i) / r_i = 1 / sqrt(1 + (r_i/delta)^2)
//
// Smooth everywhere; no singularity, no smoothing constant needed.
// Interpolates between r^2/2 (large delta) and delta*r - delta^2 (small delta).
// ===========================================================================

// [[Rcpp::export]]
double pseudo_huber_objective_cpp(const arma::vec& x,
                                  const arma::mat& X,
                                  double delta) {
  mat  diffs    = X.each_row() - x.t();
  vec  norms_sq = sum(diffs % diffs, 1);
  double d2     = delta * delta;
  return d2 * accu(sqrt(1.0 + norms_sq / d2) - 1.0);
}

// [[Rcpp::export]]
arma::vec pseudo_huber_gradient_cpp(const arma::vec& x,
                                    const arma::mat& X,
                                    double delta) {
  mat diffs    = X.each_row() - x.t();
  vec norms_sq = sum(diffs % diffs, 1);
  vec weights  = 1.0 / sqrt(1.0 + norms_sq / (delta * delta));
  return -diffs.t() * weights;
}
