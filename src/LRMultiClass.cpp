// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// this function computes P and returns P as a matrix for all possible k
// [[Rcpp::export]]
arma::mat prob_c(const arma::mat& X, const arma::mat& beta) {
  arma::mat expm = exp(X * beta);
  return (expm.each_col() / sum(expm, 1)); // each column divided by rowSums
}

// this function returns the value of the objective function
// [[Rcpp::export]]
double obj_c(const arma::mat& X, const arma::uvec& y, double lambda, const arma::mat& beta) {
  arma::mat P = prob_c(X, beta); // calculate probabilities
  double log_sum = 0.0;
  // repeat for each sample to compute the log probabilities
  for (size_t i = 0; i < X.n_rows; ++i) {
    log_sum += log(P(i, y(i))); // probabilities for each class
  }
  // sums the log probabilities of the class for each sample plus the ridge penalty term
  return -log_sum + 0.5 * lambda * arma::accu(beta % beta);
}

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                               int numIter = 50, double eta = 0.1, double lambda = 1){
    // All input is assumed to be correct
    
    // Initialize some parameters
    std::set<int> unique_elements(y.begin(), y.end()); // stores unique elements
    int K = unique_elements.size(); // number of classes
    int p = X.n_cols;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    std::vector<double> objective; // to store objective values
    
    // Initialize anything else that you may need
    objective.push_back(obj_c(X, y, lambda, beta)); // initialize objective value
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    for (int i = 0; i < numIter; i++) { // repeats for every iteration until the total number of iterations is reached
      arma::mat P = prob_c(X, beta);
      for (int j = 0; j < K; j++) { // repeats for the K classes
        arma::vec P_k = P.col(j);
        arma::vec grad = X.t() * (P_k - (y == j)) + lambda * beta.col(j); // computes gradient
        arma::mat H = X.t() * diagmat(P_k % (1 - P_k)) * X + lambda * arma::eye(p, p); // computes Hessian
        beta.col(j) -= eta * (H.i() * grad); // updates beta according to the damped Newton's method
      }
      objective.push_back(obj_c(X, y, lambda, beta)); // store the objective value for this iteration
    }
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
