#' K-means Clustering Algorithm
#'
#' @description
#' Implements K-means clustering algorithm using C++ for improved performance.
#' The algorithm partitions n observations into K clusters where each observation
#' belongs to the cluster with the nearest mean.
#'
#' @param X A numeric matrix of n observations (rows) and p variables (columns)
#' @param K Integer specifying the number of clusters
#' @param M Optional initial K x p matrix of cluster centers. If NULL, K rows will be randomly sampled from X
#' @param numIter Maximum number of iterations (default: 100)
#'
#' @return A vector of cluster assignments (integers from 1 to K)
#' @export
#'
#' @examples
#' \donttest{
#' # Generate sample data with two clusters
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
#' # Plot results if in interactive session
#' if(interactive()) {
#'   plot(X, col = clusters, pch = 16,
#'        main = "K-means Clustering Results",
#'        xlab = "X1", ylab = "X2")
#'   legend("topright", legend = paste("Cluster", 1:2),
#'          col = 1:2, pch = 16)
#' }
#' 
#' # Example with custom initial centers
#' M <- matrix(c(0, 0, 2, 2), nrow = 2, byrow = TRUE)
#' clusters_custom <- MyKmeans(X, K = 2, M = M)
#' }
#'
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
