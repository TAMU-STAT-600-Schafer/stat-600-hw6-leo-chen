
#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  
  # Compatibility checks from HW3 and initialization of beta_init
  n = nrow(X); p = ncol(X); K = length(unique(y)) # K determined based on the supplied input
  # Check that the first column of X are 1s, if not - display appropriate message and stop execution.
  if (!all(X[, 1] == 1)) {
    stop("The first column of X should contain all 1s.") # the first column of X is for the intercept
  }
  # Check for compatibility of dimensions between X and Y
  if (n != length(y)) {
    stop("The number of rows of X should be the same as the length of y.") # returns error message if the dimensions of X and y do not match
  }
  # Check eta is positive
  if (eta <= 0) {
    stop("Eta should be positive.") # ensures that the learning rate is positive in order to proceed
  }
  # Check lambda is non-negative
  if (lambda < 0) {
    stop("Lambda should be non-negative") # ensures that the ridge regulariser is non-negative in order to proceed
  }
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)) {
    beta = matrix(0, p, K)
  } else {
    if (nrow(beta_init) != p | ncol(beta_init) != K) {
      stop("The dimensions of beta_init supplied are not correct.") # returns error message if the dimensions of beta_init are not p times K
    }
    beta = beta_init # initialises beta_init if it passes the compatibility check
  }
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, numIter, eta, lambda, beta_init)
  
  # Return the class assignments
  return(out)
}