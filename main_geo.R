source("cpo_criteria.R")
#library(geoR)
#library(CensSpatial)
library(rstan)


options(mc.cores = parallel::detectCores())  # use all available CPU cores
rstan_options(auto_write = TRUE)              # avoid recompiling Stan files
rstan_options(threads_per_chain = 1)         # threads per chain (keep at 1

script_dir <- tryCatch(
  dirname(normalizePath(sys.frames()[[1]]$ofile)),
  error = function(e) getwd()
)
set.seed(123)
file_path  <- file.path(script_dir, "data", "Missouri.txt")
data <- read.table(file_path, header = TRUE, sep = "\t")


x_axis <- data$x
y_axis <- data$y
coords <- cbind(x_axis, y_axis)
cc     <- data$cc
y      <- data$log_v3
H      <- as.matrix(dist(coords))

y_obs  <- y[cc == 0];  N_obs  <- length(y_obs)
y_cens <- y[cc == 1];  N_cens <- length(y_cens)
N      <- length(cc)
kappa<-2
stan_data <- list(N = N, N_obs = N_obs, N_cens = N_cens, cc = cc,
                  coords = coords, y_obs = y_obs, y_cens = y_cens, H = H,kappa=kappa)

# ── Output folder (same directory as script) ──────────────────────────────────
out_dir <- file.path(script_dir, "results")
dir.create(out_dir, showWarnings = FALSE)

# ── Helper: save a text summary to file ───────────────────────────────────────
save_summary <- function(object, filename) {
  sink(file.path(out_dir, filename))
  print(summary(object))
  sink()
}

# ═════════════════════════════════════════════════════════════════════════════
# STAN MODELS
# ═════════════════════════════════════════════════════════════════════════════

stan_exp <- stan(file = 'exp.stan', data = stan_data,
                          init = list(list(phi = 0.1, sigmasq = 0.1),
                                      list(phi = 0.1, sigmasq = 0.1),
                                      list(phi = 0.1, sigmasq = 0.1)),
                          chain = 3, iter = 1000)
  
stan_gaus <- stan(file = 'gaussian.stan', data = stan_data,
                           init = list(list(phi = 0.1, sigmasq = 0.1),
                                       list(phi = 0.1, sigmasq = 0.1),
                                       list(phi = 0.1, sigmasq = 0.1)),
                           chain = 3, iter = 1000)

stan_cubic <- stan(file = 'cubic.stan', data = stan_data,
                            init = list(list(phi = 0.1, sigmasq = 0.1),
                                        list(phi = 0.1, sigmasq = 0.1),
                                        list(phi = 0.1, sigmasq = 0.1)),
                            chain = 3, iter = 1000)

stan_spherical <- stan(file = 'spherical.stan', data = stan_data,
                                init = list(list(phi = 0.1, sigmasq = 0.1),
                                            list(phi = 0.1, sigmasq = 0.1),
                                            list(phi = 0.1, sigmasq = 0.1)),
                                chain = 3, iter = 1000)

stan_matern <- stan(file = 'matern.stan', data = stan_data,
                             init = list(list(phi = 0.1, sigmasq = 0.1),
                                         list(phi = 0.1, sigmasq = 0.1),
                                         list(phi = 0.1, sigmasq = 0.1)),
                                chain = 3, iter = 1000)

# Save Stan model objects (.rds) and summaries (.txt)
stan_models <- list(
  exponential = stan_exp,
  gaussian    = stan_gaus,
  cubic       = stan_cubic,
  spherical   = stan_spherical,
  matern   = stan_matern
)

for (name in names(stan_models)) {
  sum_df <- as.data.frame(summary(stan_models[[name]])$summary)
  write.csv(sum_df,
            file.path(out_dir, paste0("stan_", name, "_summary.csv")))
}

# ═════════════════════════════════════════════════════════════════════════════
# CPO
# ═════════════════════════════════════════════════════════════════════════════

cpo_results <- list(
  exponential = cpo_covariance(stan_exp,      y, H, cc, cov_type = "exponential"),
  gaussian    = cpo_covariance(stan_gaus,     y, H, cc, cov_type = "gaussian"),
  cubic       = cpo_covariance(stan_cubic,    y, H, cc, cov_type = "cubic"),
  spherical   = cpo_covariance(stan_spherical,y, H, cc, cov_type = "spherical"),
  matern   = cpo_covariance(stan_matern,y, H, cc, cov_type = "matern",kappa)
)

save_criteria <- function(cpo_results, output_file = "model_criteria.csv") {
  # Initialize empty data frame to store results
  criteria_df <- data.frame(
    Model = character(),
    LPML = numeric(),
    DIC = numeric(),
    WAIC = numeric(),
    p_D = numeric(),
    Log_likelihood = numeric(),
    D_bar = numeric(),
    D_theta_bar = numeric(),
    stringsAsFactors = FALSE
  )
  
  # Extract criteria for each model
  for(model_name in names(cpo_results)) {
    model_result <- cpo_results[[model_name]]
    
    # Check if the required components exist
    if(!is.null(model_result$lpml) && !is.null(model_result$DIC) && !is.null(model_result$WAIC)) {
      criteria_df <- rbind(criteria_df, data.frame(
        Model = model_name,
        LPML = model_result$lpml,
        DIC = model_result$DIC,
        WAIC = model_result$WAIC,
        p_D = model_result$p_D,
        Log_likelihood = sum(log(model_result$ver)),
        D_bar = model_result$D_bar,
        D_theta_bar = model_result$D_theta_bar,
        stringsAsFactors = FALSE
      ))
    } else {
      warning(paste("Model", model_name, "missing required criteria components"))
    }

  }
  
  
  write.csv(criteria_df, file = output_file, row.names = FALSE)
  cat("\n======= CRITERIA SAVED TO CSV =======\n")
  print(criteria_df)
  cat("\nFile saved as:", output_file, "\n")
  
  return(criteria_df)
}


criteria_table <- save_criteria(cpo_results, file.path(out_dir, "model_comparison_criteria.csv"))
