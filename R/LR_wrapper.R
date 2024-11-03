#' Multiclass Logistic Regression Classification
#'
#' Implements multiclass logistic regression using gradient descent optimization with ridge regularization.
#'
#' @param X A numeric matrix of predictors where the first column must be all 1s (intercept term).
#'          Each row represents an observation and each column represents a feature.
#' @param y A numeric vector of class labels (response variable). Classes should be coded as 1, 2, ..., K
#'          where K is the number of classes.
#' @param numIter An integer specifying the number of iterations for gradient descent optimization.
#'                Default is 50.
#' @param eta A positive number specifying the learning rate for gradient descent.
#'           Default is 0.1.
#' @param lambda A non-negative number specifying the ridge regularization parameter.
#'              Default is 1.
#' @param beta_init Optional initial values for the coefficient matrix. If NULL (default),
#'                 initializes with a matrix of zeros. If provided, must be a p x K matrix
#'                 where p is the number of predictors and K is the number of classes.
#'
#' @return A list containing the fitted model parameters and classification results:
#'         \item{beta}{The final coefficient matrix (p x K)}
#'         \item{classes}{Predicted class labels for the training data}
#'         \item{probs}{Matrix of predicted probabilities for each class}
#' @export
#'
#' @examples
#' # Create sample data
#' X <- cbind(1, matrix(rnorm(100 * 3), 100, 3))
#' y <- sample(1:3, 100, replace = TRUE)
#' 
#' # Fit multiclass logistic regression
#' model <- LRMultiClass(X, y, numIter = 100, eta = 0.1, lambda = 0.5)
#' 
#' # View predicted classes
#' head(model$classes)
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