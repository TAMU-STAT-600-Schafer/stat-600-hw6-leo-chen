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
  return expm.each_row() / sum(expm, 1);
}

// this function returns the value of the objective function
// [[Rcpp::export]]
double obj_c(const arma::mat& X, const arma::uvec& y, double lambda, const arma::mat& beta) {
  arma::mat P = prob_c(X, beta);
  // sums the log probabilities of the class for each sample plus the ridge penalty term
  return -arma::accu(log(P.submat(arma::regspace<arma::uvec>(0, X.n_rows - 1), y))) + 0.5 * lambda * arma::accu(beta % beta);
}

// this function computes the classification error
// [[Rcpp::export]]
double error_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta) {
  arma::uvec predicted_class = arma::index_max(prob_c(X, beta), 1);
  return 100 * arma::mean(predicted_class != y); // returns the proportion of misclassified samples
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
    int K = max(y) + 1; // number of classes
    int p = X.n_cols;
    int n = X.n_rows;
    arma::mat beta = beta_init; // to store betas and be able to change them if needed
    arma::vec objective(numIter + 1); // to store objective values
    
    // Initialize anything else that you may need
    objective(0) = obj_c(X, y, lambda, beta); // initialise objective value
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    for (int i = 0; i < numIter; i++) { // repeats for every iteration until the total number of iterations is reached
      arma::mat P = prob_c(X, beta);
      for (int j = 0; j < K; j++) { // repeats for the K classes
        arma::vec P_k = P.col(j);
        arma::vec grad = X.t() * (P_k - (y == j)) + lambda * beta.col(j); // computes gradient
        arma::mat H = X.t() * diagmat(P_k % (1 - P_k)) * X + lambda * arma::eye(p, p); // computes Hessian
        beta.col(j) -= eta * (H.i() * grad); // updates beta according to the damped Newton's method
      }
      objective(i + 1) = obj_c(X, y, lambda, beta);
    }
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
