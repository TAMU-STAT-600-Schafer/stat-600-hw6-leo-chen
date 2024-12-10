#' Multiclass Logistic Regression with Ridge Regularization
#'
#' Implements multiclass logistic regression using damped Newton's method with ridge regularization.
#' The implementation uses C++ for efficient computation and handles multiple classes through
#' a one-vs-all approach.
#'
#' @param X A numeric matrix of predictors where the first column must be all 1s (intercept term).
#'          Each row represents an observation and each column represents a feature.
#' @param y A numeric vector of class labels. If classes start from 1, they will be
#'          automatically converted to 0-based indexing (0 to K-1).
#' @param numIter An integer specifying the number of iterations for Newton's method.
#'                Default is 100.
#' @param eta A positive number specifying the damping parameter for Newton's method.
#'           Controls the step size of parameter updates. Default is 0.1.
#' @param lambda A non-negative number specifying the ridge regularization parameter.
#'              Larger values result in stronger regularization. Default is 1.
#' @param beta_init Optional initial values for the coefficient matrix. If NULL (default),
#'                 initializes with a matrix of zeros of size p × K, where p is the
#'                 number of features and K is the number of classes.
#'
#' @return A list containing:
#'         \item{beta}{The final coefficient matrix (p × K) where p is the number
#'               of features (including intercept) and K is the number of classes}
#'         \item{objective}{A numeric vector containing the objective function values
#'               at each iteration, showing the convergence behavior}
#'
#' @details The function implements multinomial logistic regression using damped Newton's
#' method for optimization. The objective function includes a ridge penalty term
#' (λ/2)||β||²) for regularization. The algorithm uses a C++ implementation for
#' efficient computation of gradients and Hessians.
#'
#' @examples
#' # Generate example data
#' set.seed(123)
#' n <- 50  # number of observations
#' p <- 2   # number of features (excluding intercept)
#' K <- 3   # number of classes
#' 
#' # Create feature matrix with intercept
#' X <- cbind(1, matrix(rnorm(n * p), n, p))
#' 
#' # Generate class labels (0, 1, 2)
#' y <- sample(0:(K-1), n, replace = TRUE)
#' 
#' # Fit the model
#' result <- LRMultiClass(X, y, numIter = 50)
#'
#' @export
LRMultiClass <- function(X, y, numIter = 100, eta = 0.1, lambda = 1, beta_init = NULL) {
  # Input validation
  if (!is.matrix(X)) {
    stop("X must be a matrix")
  }
  if (!is.numeric(y)) {
    stop("y must be numeric")
  }
  if (nrow(X) != length(y)) {
    stop("Number of rows in X must match length of y")
  }
  if (!all(X[,1] == 1)) {
    stop("First column of X must be 1 (intercept)")
  }
  
  # Parameter validation
  if (lambda < 0) {
    stop("lambda must be non-negative")
  }
  if (eta <= 0) {
    stop("eta must be positive")
  }
  if (numIter <= 0) {
    stop("numIter must be positive")
  }
  
  # Convert y to 0-based index if needed
  if (min(y) > 0) {
    y <- y - min(y)
  }
  
  # Get dimensions
  n <- nrow(X)
  p <- ncol(X)
  K <- length(unique(y))
  
  # Initialize beta if not provided
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  }
  
  # Call C++ function
  result <- LRMultiClass_c(X = X, 
                          y = y, 
                          numIter = as.integer(numIter), 
                          eta = as.numeric(eta), 
                          lambda = as.numeric(lambda), 
                          beta_init = beta_init)
  
  return(result)
}


