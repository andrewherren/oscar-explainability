#################################################
# Quick and dirty OSCAR implementation in R
#################################################

### Load necessary packages
pkg.list <- c("CVXR")
lapply(pkg.list, require, character.only = TRUE)

### Set random seed
set.seed(1234)

### CVXR threshold
thresh = 1e-12

### OSCAR Solution
# Implemented as convex optimization problem using CXVR
oscar_fit <- function(X, y, c, t){
  
  # Data dimensions
  p = ncol(X)
  n = nrow(X)
  
  # Set the CVX variable to be optimized
  # beta_plus = max(beta, 0)
  # beta_minus = max(-beta, 0)
  # beta = beta_plus - beta_minus
  # abs(beta) = beta_plus + beta_minus
  
  # we're interested in beta, but for computational reasons, 
  # it's easier to split into beta_plus and beta_minus
  beta_plus <- Variable(p)
  beta_minus <- Variable(p)
  beta_soln <- beta_plus - beta_minus
  beta_abs <- beta_plus + beta_minus

  # Implement the pairwise maximum of absolute values of beta
  a <- upper_tri((rep(1, p))%*%t(beta_abs))
  b <- upper_tri(((beta_abs)%*%t(rep(1, p))))
  nu <- max_entries(hstack(a, b), axis = 1)
  
  # Minimize sum of squared errors, subject to `(sum(beta_abs) + c*sum(nu)) <= t`
  obj <- (sum_squares(y - X%*%beta_soln)/n)
  constraint1 <- ((sum(beta_abs) + c*sum(nu)) <= t)
  constraint2 <- beta_plus >= 0
  constraint3 <- beta_minus >= 0
  prob <- Problem(Minimize(obj), constraints = list(constraint1, constraint2, constraint3))
  result <- solve(prob, FEASTOL = thresh, RELTOL = thresh, ABSTOL = thresh, verbose = TRUE)
  
  # Return estimated model coefficients
  return(result$getValue(beta_soln))
}

# K-fold prediction error for a given set of OSCAR hyperparameters
oscar_cv <- function(hyper=c(1,1), k = 5, X, y){
  # unpack hyperparameters
  c = hyper[1]
  t = hyper[2]
  
  # shuffle dataset
  p = ncol(X)
  n = nrow(X)
  sample_ind <- sample(1:n, replace = F)
  X_shuffle = X[sample_ind, ]
  y_shuffle = y[sample_ind]
  
  # split into k parts
  subset_inds <- sapply(1:n, function(x) (x %% k) + 1)
  
  # iterate through each of k subsets
  pred_error <- 0
  for (i in 1:k){
    # Split up data
    X_train <- X_shuffle[subset_inds != k, ]
    X_test <- X_shuffle[subset_inds == k, ]
    y_train <- y_shuffle[subset_inds != k]
    y_test <- y_shuffle[subset_inds == k]
    
    # solve OSCAR problem on x_train
    beta_hat_OSCAR <- oscar_fit(X_train, y_train, c, t)
    
    # estimate prediction error on x_test
    pred_error <- pred_error + mean((y_test - X_test%*%beta_hat_OSCAR)^2)
  }
  
  # Average prediction error
  return(pred_error/k)
}

# Test/train split prediction error for a given set of OSCAR hyperparameters
oscar_test_train <- function(hyper=c(1,1), X_train, y_train, X_test, y_test){
  # unpack hyperparameters
  c = hyper[1]
  t = hyper[2]
  
  # solve OSCAR problem on x_train
  beta_hat_OSCAR <- oscar_fit(X_train, y_train, c, t)
  
  # estimate prediction error on x_test
  pred_error <- mean((y_test - X_test%*%beta_hat_OSCAR)^2)
  return(pred_error)
}

### Simulation 1
# Simulate the first DGP in the OSCAR paper
n = 20
p = 8
cov_exp <- abs((1:p)%*%t(rep(1,p))-rep(1,p)%*%t(1:p))
cov <- 0.7^cov_exp
chol_cov <- chol(cov)
z = matrix(rnorm(2*n*p, 0, 1), ncol = p)
X = (z%*%chol_cov)
X_train = X[1:n, ]
X_test = X[(n+1):(2*n), ]
colnames(X) <- colnames(X_train) <- colnames(X_test) <- paste0("x", seq(1:p))
sigma = 3
eps = rnorm(2*n, 0, sigma)
beta = c(3, 2, 1.5, 0, 0, 0, 0, 0)
y = X%*%beta + eps
y_train = y[1:n]
y_test = y[(n+1):(2*n)]

# MSE function for quick computation
MSE_calc <- function(beta_est, b=beta, V=cov){
  t(beta_est-b)%*%V%*%(beta_est-b)
}

# Run quick OLS estimate and calculate MSE
beta_hat_ols <- summary(lm(y_train ~ 0 + X_train))$coeff[, 1]
MSE_OLS <- MSE_calc(beta_hat_ols)

# Run quick OSCAR estimate for fixed hyperparameters
oscar_test <- oscar_fit(X_train, y_train, c = 1, t = 40)

# Tune hyperparameters c and t using test/train split
hyperparam_opt <- optim(par = c(1,1), fn = oscar_test_train, X_train = X_train, y_train = y_train, 
                        X_test = X_test, y_test = y_test, lower = 0, upper = Inf)

# Estimate OSCAR betas
oscar_beta_opt <- oscar_fit(X_train, y_train, c = hyperparam_opt$par[1], t = hyperparam_opt$par[2])

# Calculate MSE, compare to the distribution of MSEs in Table 1 of OSCAR paper
MSE_OSCAR <- MSE_calc(oscar_beta_opt)

### Simulation 2
# Binary features with one cluster of identical coefficients
n = 20
p = 8
X = apply(matrix(rbinom(2*n*p, 1, p = 0.5), ncol = p), 2, as.numeric)
X_train = X[1:n, ]
X_test = X[(n+1):(2*n), ]
sigma = sqrt(0.1)
eps = rnorm(2*n, 0, sigma)
beta = c(10, 0, 0, 0, 10, 0, 0, 10)
y = X%*%beta + eps
y_train = y[1:n]
y_test = y[(n+1):(2*n)]

# Run quick OLS estimate, see how wrong it is
beta_hat_ols <- summary(lm(y_train ~ 0 + X_train))$coeff[, 1]

# Tune model hyperparameters
hyperparam_opt <- optim(par = c(1,1), fn = oscar_test_train, X_train = X_train, y_train = y_train, 
                        X_test = X_test, y_test = y_test, lower = 0, upper = Inf)

# Estimate model betas, see how far off they are
oscar_beta_opt <- oscar_fit(X_train, y_train, c = hyperparam_opt$par[1], t = hyperparam_opt$par[2])

### Real data example 1
# Load soil data from the OSCAR paper
soil <- read.csv("data/raw/soil.csv")
X = as.matrix(soil[, 1:15])
y = as.numeric(soil[, 16])

# Get OLS solution
beta_hat_ols <- summary(lm(y ~ 0 + X))$coeff[, 1]

# Tune model hyperparameters
hyperparam_opt <- optim(par = c(1,1), fn = oscar_cv, k = 5, X = X, y = y, lower = 0, upper = Inf)

# Estimate model betas, see how far off they are
oscar_beta_opt <- oscar_fit(X, y, c = hyperparam_opt$par[1], t = hyperparam_opt$par[2])
oscar_beta_paper <- oscar_fit(X, y, c = 4, t = 0.15*sum(abs(beta_hat_ols)))
