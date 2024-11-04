#' K-means Clustering Algorithm
#'
#' @description
#' Implements K-means clustering algorithm using C++ for efficient computation.
#' The algorithm partitions n observations into K clusters by minimizing the
#' within-cluster sum of squares. Uses Euclidean distance as the distance metric.
#'
#' @param X A numeric matrix of n observations (rows) and p variables (columns).
#'          Each row represents an observation and each column represents a feature.
#' @param K Integer specifying the number of clusters (must be positive and less
#'          than the number of observations).
#' @param M Optional initial K x p matrix of cluster centers. If NULL (default),
#'          K rows will be randomly sampled from X as initial centers.
#' @param numIter Maximum number of iterations for the algorithm (default: 100).
#'                The algorithm may converge earlier.
#'
#' @return A vector of cluster assignments (integers from 1 to K) where the i-th
#'         element indicates the cluster assignment for the i-th observation.
#'
#' @details
#' The algorithm implements the standard k-means clustering using the following steps:
#' 1. Initialize cluster centers (randomly or with provided centers)
#' 2. Assign each point to the nearest center
#' 3. Update centers as the mean of assigned points
#' 4. Repeat steps 2-3 until convergence or maximum iterations
#' 
#' The implementation includes checks for empty clusters and uses efficient
#' matrix operations through RcppArmadillo.
#'
#' @examples
#' # Generate sample data with two well-separated clusters
#' set.seed(123)
#' n_per_cluster <- 50
#' X <- rbind(
#'   matrix(rnorm(n_per_cluster * 2, mean = 0, sd = 0.3), ncol = 2),
#'   matrix(rnorm(n_per_cluster * 2, mean = 2, sd = 0.3), ncol = 2)
#' )
#' 
#' # Run clustering
#' clusters <- MyKmeans(X, K = 2)
#' 
#' # Plot results
#' plot(X, col = clusters, pch = 16,
#'      main = "K-means Clustering Results",
#'      xlab = "Feature 1", ylab = "Feature 2")
#' legend("topright", legend = paste("Cluster", 1:2),
#'        col = 1:2, pch = 16)
#' 
#' # Example with custom initial centers
#' M <- matrix(c(0, 0, 2, 2), nrow = 2, byrow = TRUE)
#' clusters_custom <- MyKmeans(X, K = 2, M = M)
#'
#' @export
#' @importFrom stats rnorm
MyKmeans <- function(X, K, M = NULL, numIter = 100) {
  # Input validation
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if (!is.numeric(X)) {
    stop("X must be a numeric matrix")
  }
  if (!is.numeric(K) || K <= 0 || K != round(K)) {
    stop("K must be a positive integer")
  }
  if (K > nrow(X)) {
    stop("K cannot be larger than the number of observations")
  }
  
  n <- nrow(X)
  p <- ncol(X)
  
  # Initialize M if NULL
  if (is.null(M)) {
    # Randomly select K rows from X as initial centroids
    idx <- sample(n, K)
    M <- X[idx, , drop = FALSE]
  } else {
    # Validate M if provided
    if (!is.matrix(M)) {
      M <- as.matrix(M)
    }
    if (!is.numeric(M)) {
      stop("M must be a numeric matrix")
    }
    if (nrow(M) != K || ncol(M) != p) {
      stop(sprintf("M must be a %d x %d matrix", K, p))
    }
  }
  
  # Validate numIter
  if (!is.numeric(numIter) || numIter <= 0 || numIter != round(numIter)) {
    stop("numIter must be a positive integer")
  }
  
  # Call C++ implementation
  tryCatch({
    Y <- MyKmeans_c(X, K, M, numIter)
    # Add 1 to cluster assignments since C++ returns 0-based indices
    Y <- Y + 1
    return(Y)
  }, error = function(e) {
    stop(paste("K-means clustering failed:", e$message))
  })
}
