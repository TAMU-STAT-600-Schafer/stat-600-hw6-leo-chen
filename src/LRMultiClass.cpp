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
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::vec& y, 
                         int numIter = 50, 
                         double eta = 0.1, 
                         double lambda = 1, 
                         const arma::mat& beta_init = arma::mat()) {
    // Input validation
    if (X.n_rows != y.n_elem) {
        Rcpp::stop("Number of rows in X must match length of y");
    }
    
    // Get dimensions
    int n = X.n_rows;
    int p = X.n_cols;
    int K = arma::max(y) + 1;
    
    // Initialize beta if not provided
    arma::mat beta;
    if (beta_init.is_empty()) {
        beta = arma::zeros(p, K);
    } else {
        beta = beta_init;
    }
    
    // Initialize objective values vector
    std::vector<double> objective;
    objective.push_back(obj_c(X, arma::conv_to<arma::uvec>::from(y), lambda, beta));
    
    // Newton's method cycle
    for (int iter = 0; iter < numIter; iter++) {
        arma::mat P = prob_c(X, beta);
        
        for (int k = 0; k < K; k++) {
            arma::vec P_k = P.col(k);
            arma::vec y_k = arma::conv_to<arma::vec>::from(y == k);
            
            // Compute gradient and Hessian
            arma::vec grad = X.t() * (P_k - y_k) + lambda * beta.col(k);
            arma::mat H = X.t() * arma::diagmat(P_k % (1 - P_k)) * X + 
                         lambda * arma::eye(p, p);
            
            // Update beta using damped Newton's method
            beta.col(k) -= eta * arma::solve(H, grad);
        }
        
        // Store objective value
        objective.push_back(obj_c(X, arma::conv_to<arma::uvec>::from(y), lambda, beta));
    }
    
    return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("objective") = objective
    );
}
