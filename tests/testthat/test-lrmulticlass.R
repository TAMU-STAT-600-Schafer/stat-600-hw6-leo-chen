library(testthat)
library(microbenchmark)

# Helper function to create test data
create_test_data <- function(seed = 936004902) {
  set.seed(seed)
  n <- 16
  p <- 19
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X <- cbind(1, X)  # Add intercept
  Y <- sample(0:4, n, replace = TRUE)  # 5 classes
  K <- length(unique(Y))
  beta_init <- matrix(0, nrow = ncol(X), ncol = K)
  list(X = X, Y = Y, K = K, beta_init = beta_init)
}

test_that("LRMultiClass basic functionality works", {
  data <- create_test_data()
  out <- LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = 1)
  
  expect_true(is.matrix(out$beta))
  expect_type(out$objective, "double")
  expect_equal(dim(out$beta), c(ncol(data$X), data$K))
})

test_that("LRMultiClass handles parameter variations correctly", {
  data <- create_test_data()
  
  # Test different parameter combinations
  params_list <- list(
    list(iter = 500, eta = 0.1, lambda = 1),
    list(iter = 50, eta = 0.5, lambda = 1),
    list(iter = 50, eta = 0.01, lambda = 1),
    list(iter = 50, eta = 0.5, lambda = 5),
    list(iter = 50, eta = 0.5, lambda = 0.5)
  )
  
  for(params in params_list) {
    out <- LRMultiClass(data$X, data$Y, 
                       numIter = params$iter, 
                       eta = params$eta, 
                       lambda = params$lambda)
    expect_type(out$beta, "double")
    expect_type(out$objective, "double")
    expect_equal(dim(out$beta), c(20, data$K))
  }
})

test_that("LRMultiClass works with larger datasets", {
  set.seed(936004902)
  n <- 200
  p <- 99
  Y <- rbinom(n, size = 10, prob = 0.5) - 1
  X <- matrix(rbinom(n*p, 10, 0.5), n)
  X <- cbind(1, X)
  K <- length(unique(Y))
  
  out <- LRMultiClass(X, Y, numIter = 50, eta = 0.1, lambda = 1)
  expect_equal(dim(out$beta), c(p + 1, K))
  expect_true(length(out$objective) > 0)
})

test_that("LRMultiClass handles edge cases and errors appropriately", {
  data <- create_test_data()
  
  # Test with incorrect first column
  X_bad <- matrix(rnorm(16*20), 16, 20)
  expect_error(LRMultiClass(X_bad, data$Y, numIter = 50, eta = 0.1, lambda = 1))
  
  # Test dimension mismatch
  expect_error(LRMultiClass(data$X, c(0, 1, 2, 3), numIter = 50, eta = 0.1, lambda = 1))
  
  # Test negative learning rate
  expect_error(LRMultiClass(data$X, data$Y, numIter = 50, eta = -0.1, lambda = 1))
  
  # Test negative regularization
  expect_error(LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = -1))
  
  # Test incorrect beta_init dimensions
  expect_error(
    LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = 1,
                 beta_init = matrix(0, 20, 2))
  )
})

test_that("LRMultiClass produces consistent results", {
  data <- create_test_data()
  
  # Run twice with same parameters
  out1 <- LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = 1)
  out2 <- LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = 1)
  
  # Results should be identical
  expect_equal(out1$beta, out2$beta)
  expect_equal(out1$objective, out2$objective)
})

test_that("LRMultiClass objective function decreases", {
  data <- create_test_data()
  
  out <- LRMultiClass(data$X, data$Y, numIter = 50, eta = 0.1, lambda = 1)
  
  # Check that objective values are monotonically decreasing
  expect_true(all(diff(out$objective) <= 1e-10))
  
  # Check final objective is less than initial
  expect_lt(tail(out$objective, 1), out$objective[1])
})