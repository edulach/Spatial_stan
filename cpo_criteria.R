library(mvtnorm)

cpo_covariance <- function(stan_fit, y, H, cc,
                           cov_type = c("exponential", "gaussian", "cubic", "spherical", "matern"),
                           kappa = NULL) {
  
  cov_type <- match.arg(cov_type)
  samples  <- rstan::extract(stan_fit)
  M        <- length(samples$beta0)
  N_total  <- length(y)
  
  # ── Storage ──────────────────────────────────────────────────────────────
  results <- list(
    log_cpo  = numeric(N_total),
    cpo      = numeric(N_total),
    inv_lik  = matrix(NA, nrow = M, ncol = N_total),
    mu_cond  = numeric(N_total),   # posterior-mean conditional mean
    var_cond = numeric(N_total),   # posterior-mean conditional variance
    lpml     = NULL,
    ver = numeric(N_total),        # Likelihood at posterior means
    Loglikaux = matrix(NA, nrow = M, ncol = N_total)  # Log-likelihood across iterations
  )
  
  cat("Starting CPO calculation:", N_total, "observations,", M, "posterior samples.\n")
  
  # FIRST: Calculate posterior means for ver
  cat("\n======= STEP 1: Calculating 'ver' (likelihood at posterior means) =======\n")
  beta0_mean <- mean(samples$beta0)
  phi_mean <- mean(samples$phi)
  sigmasq_mean <- mean(samples$sigmasq)
  nugget_mean <- mean(samples$nugget)
  
  for (i in 1:N_total) {
    cat("  Processing ver for observation", i, "of", N_total, "\n")
    
    other_idx <- setdiff(1:N_total, i)
    is_cens_i <- cc[i] == 1
    
    # Calculate covariance matrix with posterior means
    ratio <- H / phi_mean
    
    
    if (cov_type == "exponential") {
      cov_mat <- exp(-ratio)
    } else if (cov_type == "gaussian") {
      cov_mat <- exp(-ratio^2)
    } else if (cov_type == "cubic") {
      cov_mat <- (1 - 7 * ratio^2 + 8.75 * ratio^3 - 3.5 * ratio^5 + 0.75 * ratio^7)
      cov_mat[H >= phi_mean] <- 0
    } else if (cov_type == "spherical") {
      cov_mat <- (1 - 1.5 * ratio + 0.5 * ratio^3)
      cov_mat[H >= phi_mean] <- 0
    } else if (cov_type == "matern") {
      cov_mat <- (2^(1 - kappa) / gamma(kappa)) *
        ratio^kappa * besselK(ratio, nu = kappa)
      cov_mat[ratio==0] <- 1
    }
    
    
    Sigma_full <- nugget_mean * diag(N_total) + sigmasq_mean * cov_mat
    
    # Partitions
    Sigma_oo <- Sigma_full[other_idx, other_idx, drop = FALSE] 
    Sigma_io <- Sigma_full[i, other_idx, drop = FALSE]
    Sigma_ii <- Sigma_full[i, i, drop = FALSE] 
    
    mu_oo <- rep(beta0_mean, length(other_idx))
    mu_i <- beta0_mean
    
    # Conditional mean and variance
    Sigma_oo_inv <- solve(Sigma_oo)
    mu_cond <- as.numeric(mu_i + Sigma_io %*% Sigma_oo_inv %*% (y[other_idx] - mu_oo))
    var_cond <- as.numeric(Sigma_ii - Sigma_io %*% Sigma_oo_inv %*% t(Sigma_io))
    sd_cond <- sqrt(max(var_cond, 1e-10))
    # Calculate ver (likelihood at posterior means)
    if (is_cens_i) {
      results$ver[i] <- pmvnorm(
        lower = -Inf,
        upper = y[i],
        mean = mu_cond,
        sigma = matrix(sd_cond^2, nrow = 1, ncol = 1),
        abseps = 1e-10,
        releps = 1e-10,
        maxpts = 1e6
      )[1]
    } else {
      results$ver[i] <- dnorm(y[i], mean = mu_cond, sd = sd_cond, log = FALSE)
    }
    
    if (is.na(results$ver[i])) results$ver[i] <- 1e-20
    cat(sprintf("    ver[%d] = %.6e\n", i, results$ver[i]))
  }
  
  # SECOND: Calculate Loglikaux (log-likelihood across iterations)
  cat("\n======= STEP 2: Calculating 'Loglikaux' (log-likelihood across MCMC iterations) =======\n")
  
  
  
  
  for (i in seq_len(N_total)) {
    
    cat("\n--------------------------------------\n")
    cat("Processing observation", i, "of", N_total, "\n")
    
    other_idx  <- setdiff(seq_len(N_total), i)
    is_cens_i  <- cc[i] == 1
  
    
    for (j in 1:M) {
      
      # Get parameters from this MCMC draw
      mu_j <- samples$beta0[j]
      phi_j <- samples$phi[j]
      sigmasq_j <- samples$sigmasq[j]
      nugget_j <- samples$nugget[j]
      
      ratio <- H / phi_j
      
      # ── Covariance function ─────────────────────────────────────────────
      
      if (cov_type == "exponential") {
        cov_mat <- exp(-ratio)
      } else if (cov_type == "gaussian") {
        cov_mat <- exp(-ratio^2)
      } else if (cov_type == "cubic") {
        cov_mat <- (1 - 7 * ratio^2 + 8.75 * ratio^3 - 3.5 * ratio^5 + 0.75 * ratio^7)
        cov_mat[H >= phi_j] <- 0
      } else if (cov_type == "spherical") {
        cov_mat <- (1 - 1.5 * ratio + 0.5 * ratio^3)
        cov_mat[H >= phi_j] <- 0
      } else if (cov_type == "matern") {
        cov_mat <- (2^(1 - kappa) / gamma(kappa)) *
          ratio^kappa * besselK(ratio, nu = kappa)
        cov_mat[ratio==0] <- 1
      }
      
      
      # ── Full covariance matrix (single jitter applied once) ─────────────
      Sigma_full <- nugget_j * diag(N_total) + sigmasq_j * cov_mat
      Sigma_full <- Sigma_full + 1e-6 * diag(N_total)   # single consistent jitter
      
      # ── Partition ───────────────────────────────────────────────────────
      Sigma_oo <- Sigma_full[other_idx, other_idx, drop = FALSE]
      Sigma_io <- Sigma_full[i, other_idx, drop = FALSE]   # 1 × (N-1)
      Sigma_ii <- Sigma_full[i, i]
      
      mu_oo <- rep(mu_j, length(other_idx))
      mu_i  <- mu_j
      
      # ── Conditional mean & variance via Cholesky ─────────────────────── 
      L         <- tryCatch(chol(Sigma_oo), error = function(e) {
        chol(Sigma_oo + 1e-6 * diag(nrow(Sigma_oo)))
      })
      Loo_inv   <- chol2inv(L)                                    # (N-1)×(N-1)
      resid_oo  <- y[other_idx] - mu_oo                           # (N-1)×1
      
      mu_cond  <- as.numeric(mu_i + Sigma_io %*% Loo_inv %*% resid_oo)
      var_cond <- as.numeric(Sigma_ii - Sigma_io %*% Loo_inv %*% t(Sigma_io))
      var_cond <- max(var_cond, 1e-10)
      sd_cond  <- sqrt(var_cond)
      
      # ── Likelihood ──────────────────────────────────────────────────────
      # cc == 1  → censored: use survival probability P(Y > y_i) for
      #            right-censoring, or P(Y ≤ y_i) for left-censoring.
      # Adjust lower.tail below to match your censoring convention.edu
      if (is_cens_i) {
        lik_i <- pmvnorm(
          lower = -Inf,
          upper = y[i],
          mean = mu_cond,
          sigma = matrix(sd_cond^2, nrow = 1, ncol = 1),
          abseps = 1e-10,
          releps = 1e-10,
          maxpts = 1e6
        )[1]
        if (is.na(lik_i)) lik_i <- 1e-20
      } else {
        lik_i <- dnorm(y[i], mean = mu_cond, sd = sd_cond, log = FALSE)
      }
    
      # Store Loglikaux (log-likelihood, NOT inverse)
      results$Loglikaux[j, i] <- log(max(lik_i, 1e-20))
      
      # Store inverse likelihood for CPO (existing functionality)
      results$inv_lik[j, i] <- 1 / max(lik_i, 1e-20)
      
    }
    
      # Store conditional parameters (from last iteration)
      results$mu_cond[i] <- mu_cond
      results$var_cond[i] <- var_cond
      
      # CPO calculation (existing functionality)
      cpo_i <- 1 / mean(results$inv_lik[, i])
      results$cpo[i] <- cpo_i
      results$log_cpo[i] <- log(cpo_i)
      
      cat(sprintf("  - CPO[%d] = %.4e, log(CPO) = %.4f\n", i, cpo_i, log(cpo_i)))
    }
    
    # Compute LPML (existing functionality)
    results$lpml <- sum(results$log_cpo)
    
    # ADDITIONAL: compute DIC components
    cat("\n======= ADDITIONAL MODEL DIAGNOSTICS =======\n")
    
    # Log-likelihood at posterior means (from ver)
    log_ver <- sum(log(results$ver))
    cat(sprintf("Log-likelihood at posterior means: %.4f\n", log_ver))
    
    # Mean deviance (from Loglikaux)
    D_bar <- -2 * mean(rowSums(results$Loglikaux))
    cat(sprintf("Mean deviance (D_bar): %.4f\n", D_bar))
    
    # Deviance at posterior means
    D_theta_bar <- -2 * log_ver
    cat(sprintf("Deviance at posterior means (D_theta_bar): %.4f\n", D_theta_bar))
    
    # Effective number of parameters
    p_D <- D_bar - D_theta_bar
    cat(sprintf("Effective parameters (p_D): %.4f\n", p_D))
    
    # DIC
    DIC <- D_bar + p_D
    cat(sprintf("DIC: %.4f\n", DIC))
    
    # WAIC components
    lppd <- sum(log(colMeans(exp(results$Loglikaux))))
    p_waic <- sum(apply(results$Loglikaux, 2, var))
    WAIC <- -2 * (lppd - p_waic)
    cat(sprintf("WAIC: %.4f\n", WAIC))
    cat(sprintf("LPML: %.4f\n", results$lpml))
    
    results$DIC         <- DIC
    results$WAIC        <- WAIC
    results$p_D         <- p_D
    results$D_bar       <- D_bar
    results$D_theta_bar <- D_theta_bar
    
    print(results$Loglikaux)
    return(results)
}