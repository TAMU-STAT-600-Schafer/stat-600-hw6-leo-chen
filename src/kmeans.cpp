// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                      const arma::mat& M, int numIter = 100){
  // Input parameters:
  // X: Data matrix where each row is an observation and each column is a feature
  // K: Number of clusters
  // M: Initial cluster centers matrix (K x p)
  // numIter: Maximum number of iterations
  
  // Initialize dimensions
  int n = X.n_rows;  // number of observations
  int p = X.n_cols;  // number of features
  arma::uvec Y(n);   // cluster assignments vector (0-based indexing)
  
  // Initialize algorithm variables
  arma::uvec old_Y = arma::uvec(n).fill(n+1);  // previous cluster assignments
  arma::mat M_cal(M);                           // current cluster centers
  arma::mat cluster_sums(K, p);                 // sum of points in each cluster
  arma::uvec cluster_sizes(K);                  // number of points in each cluster
  arma::rowvec row_sum(K);                      // squared norms of centers
  arma::mat distances(n, K);                    // distance matrix between points and centers

  // Main k-means iteration loop
  for(int iter = 0; iter < numIter; iter++) {
    // Calculate squared L2 norms of cluster centers
    row_sum = sum(square(M_cal), 1).t();
    
    // Compute squared Euclidean distances between points and centers
    // Using matrix multiplication for efficiency: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x^T y
    distances = -2 * X * M_cal.t();
    distances.each_row() += row_sum;
    
    // Find nearest center for each point
    Y = arma::index_min(distances, 1);
    
    // Check convergence
    if(arma::all(Y == old_Y)) {
      Rcpp::Rcout << "Converged in " << iter + 1 << " iterations.\n";
      break;
    }
    
    // Calculate cluster sizes and check for empty clusters
    cluster_sizes.zeros();
    for(arma::uword i = 0; i < n; i++) {
      cluster_sizes(Y(i))++;
    }
    
    if(arma::any(cluster_sizes == 0)) {
      arma::uvec empty_clusters = arma::find(cluster_sizes == 0);
      std::string msg = "Cluster(s) " + std::to_string(empty_clusters(0));
      for(arma::uword i = 1; i < empty_clusters.n_elem; i++) {
        msg += ", " + std::to_string(empty_clusters(i));
      }
      msg += " disappeared. Try changing the value of M.";
      Rcpp::stop(msg);
    }
    
    // Update centers
    cluster_sums.zeros();
    for(arma::uword i = 0; i < n; i++) {
      cluster_sums.row(Y(i)) += X.row(i);
    }
    
    // Calculate new centers
    for(int k = 0; k < K; k++) {
      M_cal.row(k) = cluster_sums.row(k) / cluster_sizes(k);
    }
    
    old_Y = Y;
  }
  
  // Returns the vector of cluster assignments
  return(Y);
}

