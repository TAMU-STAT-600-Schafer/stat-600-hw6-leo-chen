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
    // Initialize parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n);  // cluster assignments
    arma::mat centroids = M;  // copy initial centroids
    
    // Pre-compute squared norms of data points
    arma::vec sq_X = sum(square(X), 1);
    
    // Main k-means loop
    for(int iter = 0; iter < numIter; iter++) {
        // Compute squared norms of centroids
        arma::vec sq_M = sum(square(centroids), 1);
        
        // Compute distances matrix
        arma::mat D = repmat(sq_X, 1, K) - 
                     2 * X * centroids.t() + 
                     repmat(sq_M.t(), n, 1);
        
        // computes the cluster assignments for each observation as the indices of the minimum values across each row
        Y = index_min(D, 1);
        
        // Check for empty clusters
        arma::uvec cluster_counts = arma::zeros<arma::uvec>(K);
        for(unsigned int i = 0; i < n; i++) {
            cluster_counts(Y(i))++;
        }
        if(min(cluster_counts) == 0) {
            Rcpp::stop("A cluster has disappeared");
        }
        
        // re-computes the means of each cluster
        arma::mat new_centroids = arma::zeros(K, p);
        for(int i = 0; i < n; i++) {
            new_centroids.row(Y(i)) += X.row(i);
        }
        
        for(int k = 0; k < K; k++) {
            new_centroids.row(k) /= cluster_counts(k);
        }
        
        // stops if the convergence criterion is met (the mean values do not change for every cluster)
        if(accu(abs(new_centroids - centroids)) < 1e-6) {
            break;
        }
        centroids = new_centroids; // updates centroids
    }
    return Y;
}

