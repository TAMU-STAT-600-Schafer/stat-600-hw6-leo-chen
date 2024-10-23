# This is a script to save your own tests for the function
source("FunctionsLR.R")

library(Rcpp)
library(RcppArmadillo)

# Source your C++ funcitons
sourceCpp("LRMultiClass.cpp")

# first test case
Y = c(0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 3, 4, 1, 2, 0, 4)
X = matrix(rnorm(16*19), 16)
Yt = c(1, 0, 3, 2)
Xt = matrix(rnorm(4*19), 4)
X = cbind(1, X)
Xt = cbind(1, Xt)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 500, eta = 0.1, lambda = 1) # more iterations
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 1) # larger eta
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.01, lambda = 1) # smaller eta
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 5) # larger lambda
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 0.5) # smaller lambda
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')

# compatibility checks to see if error messages will be returned appropriately
out = LRMultiClass(matrix(rnorm(17*20), 17), Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
out = LRMultiClass(X, Y, matrix(rnorm(5*20), 5), Yt, numIter = 50, eta = 0.1, lambda = 1)
out = LRMultiClass(X, c(0, 1, 2, 3), Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
out = LRMultiClass(X, Y, Xt, c(1, 2), numIter = 50, eta = 0.1, lambda = 1)
out = LRMultiClass(X, Y, cbind(1, Xt), Yt, numIter = 50, eta = 0.1, lambda = 1)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = -0.1, lambda = 1)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = -1)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(1, 20, 5))
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(10, 20, 5))
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(0, 20, 2))
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(0, 2, 5))

# 2nd test case
Y = rbinom(200, size = 10, prob = 0.5) - 1
X = matrix(rbinom(200*99, 10, 0.5), 200)
Yt = rbinom(30, size = 10, prob = 0.5) - 1
Xt = matrix(rbinom(30*99, 10, 0.5), 30)
X = cbind(1, X)
Xt = cbind(1, Xt)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 500, eta = 0.1, lambda = 1) # more iterations
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 1) # larger eta
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.01, lambda = 1) # smaller eta
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 5) # larger lambda
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.5, lambda = 0.5) # smaller lambda
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')

# tests relating to the letter example
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])
X = cbind(1, X)
Xt = cbind(1, Xt)
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(1, ncol(X), length(unique(Y))))
plot(out$objective, type = 'o', xlab = "number of iterations", ylab = "objective values")
plot(out$error_train, type = 'o', xlab = "number of iterations", ylab = "training error")
plot(out$error_test, type = 'o')
