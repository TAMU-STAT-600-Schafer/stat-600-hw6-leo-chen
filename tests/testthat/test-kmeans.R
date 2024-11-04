library(testthat)
library(fossil)

test_that("kmeans clusters well-separated data correctly", {
  set.seed(123)
  cluster1 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
  cluster2 <- matrix(rnorm(50, mean = 5, sd = 0.5), ncol = 2)
  X_test1 <- rbind(cluster1, cluster2)
  true_labels1 <- c(rep(1, 25), rep(2, 25))
  
  Y_pred1 <- MyKmeans(X_test1, K = 2)
  rand_index1 <- rand.index(Y_pred1, true_labels1)
  expect_gt(rand_index1, 0.9)
})

test_that("kmeans handles overlapping clusters", {
  set.seed(456)
  cluster1 <- matrix(rnorm(30, mean = 0, sd = 0.8), ncol = 2)
  cluster2 <- matrix(rnorm(30, mean = 3, sd = 0.8), ncol = 2)
  cluster3 <- matrix(rnorm(30, mean = 6, sd = 0.8), ncol = 2)
  X_test2 <- rbind(cluster1, cluster2, cluster3)
  true_labels2 <- c(rep(1, 15), rep(2, 15), rep(3, 15))
  
  Y_pred2 <- MyKmeans(X_test2, K = 3)
  rand_index2 <- rand.index(Y_pred2, true_labels2)
  expect_gt(rand_index2, 0.7)
})

test_that("kmeans handles edge cases", {
  # Single cluster
  set.seed(789)
  X_test3 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
  Y_pred3 <- MyKmeans(X_test3, K = 1)
  expect_equal(length(unique(Y_pred3)), 1)
  
  # Invalid K
  X_test7 <- matrix(rnorm(10, mean = 0, sd = 1), ncol = 2)
  expect_error(MyKmeans(X_test7, K = 15))
  
  # Identical points
  X_test8 <- matrix(rep(1, 20), ncol = 2)
  expect_error(MyKmeans(X_test8, K = 3))
})

test_that("kmeans is consistent with same seed", {
  set.seed(222324)
  X_test9 <- matrix(rnorm(50, mean = 0, sd = 0.5), ncol = 2)
  
  set.seed(252627)
  Y1 <- MyKmeans(X_test9, K = 2)
  set.seed(252627)
  Y2 <- MyKmeans(X_test9, K = 2)
  
  expect_equal(Y1, Y2)
})

test_that("kmeans performance is acceptable", {
  set.seed(161718)
  X_test6 <- rbind(
    matrix(rnorm(500, mean = 0), ncol = 50),
    matrix(rnorm(500, mean = 3), ncol = 50)
  )
  
  time <- system.time(MyKmeans(X_test6, K = 2))[3]
  expect_lt(time, 1)
}) 