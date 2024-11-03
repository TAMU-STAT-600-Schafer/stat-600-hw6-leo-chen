# setwd("/Users/apple/AM-Stat/Courses/Stat600/stat-600-hw6-leo-chen/src")
# This is a script to save your own tests for the function
source("FunctionsLR.R")

library(Rcpp)
library(RcppArmadillo)
library(fossil)  # For calculating Rand Index

# Source your C++ functions
sourceCpp("LRMultiClass.cpp")
sourceCpp("kmeans.cpp")

# ====================== Tests for LRMultiClass ======================

# Test Case 1: Small Dataset with Multiple Classes
set.seed(936004902)
Y = c(0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 3, 4, 1, 2, 0, 4)
X = matrix(rnorm(16*19), 16)
Yt = c(1, 0, 3, 2)
Xt = matrix(rnorm(4*19), 4)
X = cbind(1, X)
Xt = cbind(1, Xt)
K = length(unique(Y))
beta_init = matrix(0, 20, K)

print("Test Case 1 - Testing basic functionality and parameter variations:")
# Basic functionality test
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
out_c = LRMultiClass_c(X, Y, beta_init, numIter = 50, eta = 0.1, lambda = 1)
print(paste("Basic test - Beta matrices equal:", all.equal(out$beta, out_c$beta)))
print(paste("Basic test - Objectives equal:", all.equal(out$objective, out_c$objective)))

# Parameter variation tests
print("\nTesting different parameter combinations:")
test_params = list(
  list(iter = 500, eta = 0.1, lambda = 1, desc = "More iterations"),
  list(iter = 50, eta = 0.5, lambda = 1, desc = "Larger eta"),
  list(iter = 50, eta = 0.01, lambda = 1, desc = "Smaller eta"),
  list(iter = 50, eta = 0.5, lambda = 5, desc = "Larger lambda"),
  list(iter = 50, eta = 0.5, lambda = 0.5, desc = "Smaller lambda")
)

for(params in test_params) {
  out = LRMultiClass(X, Y, Xt, Yt, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  out_c = LRMultiClass_c(X, Y, beta_init, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  print(paste(params$desc, "- Beta matrices equal:", all.equal(out$beta, out_c$beta)))
  print(paste(params$desc, "- Objectives equal:", all.equal(out$objective, out_c$objective)))
}

# Performance comparison
print("\nPerformance Comparison:")
library(microbenchmark)
print(microbenchmark(
  R_version = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1),
  Cpp_version = LRMultiClass_c(X, Y, beta_init, numIter = 50, eta = 0.1, lambda = 1),
  times = 20
))

# Test Case 2: Larger Dataset with Binary Classification
print("\nTest Case 2 - Testing with larger binary classification dataset:")
set.seed(936004902)
Y = rbinom(200, size = 10, prob = 0.5) - 1
X = matrix(rbinom(200*99, 10, 0.5), 200)
Yt = rbinom(30, size = 10, prob = 0.5) - 1
Xt = matrix(rbinom(30*99, 10, 0.5), 30)
X = cbind(1, X)
Xt = cbind(1, Xt)
K = length(unique(Y))
beta_init = matrix(0, 100, K)

# Run same tests with new dataset
out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
out_c = LRMultiClass_c(X, Y, beta_init, numIter = 50, eta = 0.1, lambda = 1)
print(paste("Basic test - Beta matrices equal:", all.equal(out$beta, out_c$beta)))
print(paste("Basic test - Objectives equal:", all.equal(out$objective, out_c$objective)))

# Parameter variations for larger dataset
for(params in test_params) {
  out = LRMultiClass(X, Y, Xt, Yt, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  out_c = LRMultiClass_c(X, Y, beta_init, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  print(paste(params$desc, "- Beta matrices equal:", all.equal(out$beta, out_c$beta)))
  print(paste(params$desc, "- Objectives equal:", all.equal(out$objective, out_c$objective)))
}

# Performance comparison for larger dataset
print("\nPerformance Comparison (Larger Dataset):")
print(microbenchmark(
  R_version = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1),
  Cpp_version = LRMultiClass_c(X, Y, beta_init, numIter = 50, eta = 0.1, lambda = 1),
  times = 20
))

# Test Case 3: Letter Recognition Dataset
print("\nTest Case 3 - Testing with letter recognition dataset:")
letter_train <- read.table("Data/letter-train.txt", header = F, colClasses = "numeric")
Y <- letter_train[, 1]
X <- as.matrix(letter_train[, -1])
letter_test <- read.table("Data/letter-test.txt", header = F, colClasses = "numeric")
Yt <- letter_test[, 1]
Xt <- as.matrix(letter_test[, -1])
X = cbind(1, X)
Xt = cbind(1, Xt)
K = length(unique(Y))
beta_init = matrix(0, 17, K)

# Run same tests with letter dataset
for(params in test_params) {
  out = LRMultiClass(X, Y, Xt, Yt, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  out_c = LRMultiClass_c(X, Y, beta_init, numIter = params$iter, eta = params$eta, lambda = params$lambda)
  print(paste(params$desc, "- Beta matrices equal:", all.equal(out$beta, out_c$beta)))
  print(paste(params$desc, "- Objectives equal:", all.equal(out$objective, out_c$objective)))
}

# Performance comparison for letter dataset
print("\nPerformance Comparison (Letter Dataset):")
print(microbenchmark(
  R_version = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1),
  Cpp_version = LRMultiClass_c(X, Y, beta_init, numIter = 50, eta = 0.1, lambda = 1),
  times = 20
))

# Test Case 4: Edge Cases and Error Handling
print("\nTest Case 4 - Testing error handling:")

# Test with incorrect first column
print("Testing with non-intercept first column:")
X_bad = matrix(rnorm(16*20), 16, 20)
tryCatch({
  out = LRMultiClass(X_bad, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
}, error = function(e) {
  print(paste("Error caught as expected:", e$message))
})

# Test dimension mismatch
print("\nTesting dimension mismatch:")
tryCatch({
  out = LRMultiClass(X, c(0, 1, 2, 3), Xt, Yt, numIter = 50, eta = 0.1, lambda = 1)
}, error = function(e) {
  print(paste("Error caught as expected:", e$message))
})

# Test negative learning rate
print("\nTesting negative learning rate:")
tryCatch({
  out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = -0.1, lambda = 1)
}, error = function(e) {
  print(paste("Error caught as expected:", e$message))
})

# Test negative regularization
print("\nTesting negative regularization:")
tryCatch({
  out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = -1)
}, error = function(e) {
  print(paste("Error caught as expected:", e$message))
})

# Test incorrect beta_init dimensions
print("\nTesting incorrect beta_init dimensions:")
tryCatch({
  out = LRMultiClass(X, Y, Xt, Yt, numIter = 50, eta = 0.1, lambda = 1, 
                     beta_init = matrix(0, 20, 2))
}, error = function(e) {
  print(paste("Error caught as expected:", e$message))
})

# compatibility checks to see if error messages will be returned appropriately
# Y = c(0, 1, 2, 3, 4, 3, 2, 1, 0, 2, 3, 4, 1, 2, 0, 4)
# X = matrix(rnorm(16*19), 16)
# X = cbind(1, X)
# out = LRMultiClass(matrix(rnorm(17*20), 17), Y, numIter = 50, eta = 0.1, lambda = 1)
# out = LRMultiClass(X, c(0, 1, 2, 3), numIter = 50, eta = 0.1, lambda = 1)
# out = LRMultiClass(X, Y, numIter = 50, eta = -0.1, lambda = 1)
# out = LRMultiClass(X, Y, numIter = 50, eta = 0.1, lambda = -1)
# out = LRMultiClass(X, Y, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(0, 20, 2))
# out = LRMultiClass(X, Y, numIter = 50, eta = 0.1, lambda = 1, beta_init = matrix(0, 2, 5))


# ====================== Tests for MyKmeans ======================
# Test Case 1: Two Distinct Clusters in 2D
set.seed(123)
cluster1 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
cluster2 <- matrix(rnorm(50, mean = 5, sd = 0.5), ncol = 2)
X_test1 <- rbind(cluster1, cluster2)
true_labels1 <- c(rep(1, 25), rep(2, 25))

Y_pred1 <- MyKmeans(X_test1, K = 2)
rand_index1 <- rand.index(Y_pred1, true_labels1)
print(paste("Test Case 1 - Rand Index:", rand_index1))

# Visual inspection
plot(X_test1, col = Y_pred1, pch = 16, main = "Test Case 1: K-means Clustering Results")
legend("topright", legend = paste("Cluster", 1:2), col = 1:2, pch = 16)

# Test Case 2: Three Overlapping Clusters in 2D
set.seed(456)
cluster1 <- matrix(rnorm(30, mean = 0, sd = 0.8), ncol = 2)
cluster2 <- matrix(rnorm(30, mean = 3, sd = 0.8), ncol = 2)
cluster3 <- matrix(rnorm(30, mean = 6, sd = 0.8), ncol = 2)
X_test2 <- rbind(cluster1, cluster2, cluster3)
true_labels2 <- c(rep(1, 15), rep(2, 15), rep(3, 15))

Y_pred2 <- MyKmeans(X_test2, K = 3)
rand_index2 <- rand.index(Y_pred2, true_labels2)
print(paste("Test Case 2 - Rand Index:", rand_index2))

# Visual inspection
plot(X_test2, col = Y_pred2, pch = 16, main = "Test Case 2: K-means Clustering Results")
legend("topright", legend = paste("Cluster", 1:3), col = 1:3, pch = 16)

# Test Case 3: Single Cluster
set.seed(789)
X_test3 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
Y_pred3 <- MyKmeans(X_test3, K = 1)
unique_clusters3 <- length(unique(Y_pred3))
print(paste("Test Case 3 - Number of clusters found:", unique_clusters3))

# Test Case 4: Invalid M
set.seed(101112)
X_test4 <- matrix(rnorm(40, mean = 0, sd = 0.5), ncol = 2)
M_invalid <- matrix(0, nrow = 3, ncol = ncol(X_test4) + 1)
print("Test Case 4 - Testing with invalid M:")
tryCatch({
  Y_pred4 <- MyKmeans(X_test4, K = 2, M = M_invalid)
}, error = function(e) {
  print("Error caught as expected:")
  print(e$message)
})

# Test Case 5: Cluster Disappearance
set.seed(131415)
X_test5 <- matrix(rnorm(60, mean = 0, sd = 1), ncol = 2)
M_bad <- X_test5[1:2, ]
print("Test Case 5 - Testing cluster disappearance:")
tryCatch({
  Y_pred5 <- MyKmeans(X_test5, K = 3, M = M_bad)
}, error = function(e) {
  print("Cluster disappearance handled:")
  print(e$message)
})

# Test Case 6: High-dimensional Data
set.seed(161718)
X_test6_cluster1 <- matrix(rnorm(500, mean = 0), ncol = 50)
X_test6_cluster2 <- matrix(rnorm(500, mean = 3), ncol = 50)
X_test6 <- rbind(X_test6_cluster1, X_test6_cluster2)
true_labels6 <- c(rep(1, 10), rep(2, 10))

Y_pred6 <- MyKmeans(X_test6, K = 2)
rand_index6 <- rand.index(Y_pred6, true_labels6)
print(paste("Test Case 6 - Rand Index:", rand_index6))

# Performance Testing
library(microbenchmark)
print(microbenchmark(
  MyKmeans(X_test6, K = 2),
  times = 10
))

# Test Case 7-10: Edge Cases and Consistency
set.seed(192021)
X_test7 <- matrix(rnorm(10, mean = 0, sd = 1), ncol = 2)
print("Test Case 7 - Testing with K larger than number of points:")
tryCatch({
  Y_pred7 <- MyKmeans(X_test7, K = 15)
}, error = function(e) {
  print("Error caught as expected:")
  print(e$message)
})

# Test identical data points
X_test8 <- matrix(rep(1, 20), ncol = 2)
print("Test Case 8 - Testing with identical data points:")
tryCatch({
  Y_pred8 <- MyKmeans(X_test8, K = 3)
}, error = function(e) {
  print("Error caught as expected:")
  print(e$message)
})

# Test consistency
set.seed(222324)
X_test9 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
set.seed(252627)
Y_pred9a <- MyKmeans(X_test9, K = 2)
set.seed(252627)
Y_pred9b <- MyKmeans(X_test9, K = 2)
consistency <- identical(Y_pred9a, Y_pred9b)
print(paste("Test Case 9 - Consistency across runs:", consistency))
