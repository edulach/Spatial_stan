functions {
  // Function for the exponential method

matrix covariance_matern(int N, matrix H, real sigmasq, real phi,
                          real nugget, int kappa) {
  matrix[N, N] Sigma;

  for (i in 1:N) {
    for (j in i:N) {
      if (i == j) {
        Sigma[i, j] = sigmasq + nugget;

      } else {
        real h_phi = H[i, j] / phi;   // scaled distance, matches geoR
        real s     = sqrt(2 * kappa) * h_phi;

        // General Matern:
        // C(h) = sigmasq * 2^(1-kappa)/Gamma(kappa) * (sqrt(2*kappa)*h/phi)^kappa * K_kappa(sqrt(2*kappa)*h/phi)
        real cov_val = sigmasq
                       * pow(2.0, 1 - kappa)
                       / tgamma(kappa)
                       * pow(s, kappa)
                       * modified_bessel_second_kind(kappa, s);

        Sigma[i, j] = cov_val;
        Sigma[j, i] = cov_val;
      }
    }
  }
  return Sigma;
}

}

data {
  int<lower=1> N;
  int<lower=0> N_obs;
  int<lower=0> N_cens;
  matrix[N,N] H;
  vector[N] cc;
  vector[N_obs] y_obs;
  vector[N_cens] y_cens;
  int<lower=0> kappa;
}
 
transformed data {
  int obs_idx[N_obs];
  int cens_idx[N_cens];
  
  {
    int o = 1;
    int c = 1;
    for (n in 1:N) {
      if (cc[n] == 0) {
        obs_idx[o] = n;
        o += 1;
      } else {
        cens_idx[c] = n;
        c += 1;
      }
    }
  }
}

parameters {
  real beta0;
  real<lower=0> sigmasq;
  real<lower=0.00001, upper=50> phi;
  vector<upper=min(y_cens)>[N_cens] ycen;
  real<lower=0> nugget;
  
}
 
transformed parameters {
  matrix[N, N] Sigma = covariance_matern(N, H, sigmasq, phi, nugget,kappa);
  
  // Partition covariance matrices
  matrix[N_obs, N_obs] Sigma_oo;
  matrix[N_cens, N_cens] Sigma_cc;
  matrix[N_cens, N_obs]  Sigma_co;
  
  // Conditional mean and covariance — computed once, reused everywhere
  vector[N_cens]          mu_cond;
  matrix[N_cens, N_cens]  Sigma_cond;

  for (i in 1:N_obs)
    for (j in 1:N_obs)
      Sigma_oo[i,j] = Sigma[obs_idx[i], obs_idx[j]];

  for (i in 1:N_cens)
    for (j in 1:N_cens)
      Sigma_cc[i,j] = Sigma[cens_idx[i], cens_idx[j]];

  for (i in 1:N_cens)
    for (j in 1:N_obs)
      Sigma_co[i,j] = Sigma[cens_idx[i], obs_idx[j]];

  {
    // ── Numerically stable solve via inverse_spd ──────────────────────────
    // inverse_spd uses Cholesky internally — safer than inverse() for SPD matrices
    matrix[N_obs, N_obs] Sigma_oo_jit = add_diag(Sigma_oo, 1e-6);
    matrix[N_obs, N_obs] Sigma_oo_inv = inverse_spd(Sigma_oo_jit);

    mu_cond    = rep_vector(beta0, N_cens)
                 + Sigma_co * Sigma_oo_inv
                 * (y_obs - rep_vector(beta0, N_obs));

    // Symmetrize + jitter to guarantee positive-definiteness
    Sigma_cond = Sigma_cc - Sigma_co * Sigma_oo_inv * Sigma_co';
    Sigma_cond = 0.5 * (Sigma_cond + Sigma_cond');
    Sigma_cond = add_diag(Sigma_cond, 1e-6);
    
    
  }
}

model {
  
  
  // Priors
  beta0 ~ normal(0, 50);
  sigmasq ~ student_t(4, 0, 5);
  kappa ~ student_t(4, 0, 5);
  //sigmasq ~ inv_gamma(2.1,6.6);
  phi ~ student_t(4, 0, 5);
  //phi ~ gamma(2,0.1);
  nugget ~ student_t(4, 0, 5);
 // nugget ~inv_gamma(2.1,0.55);
// Likelihood — reuse mu_cond and Sigma_cond from transformed parameters
 y_obs ~ multi_normal(rep_vector(beta0, N_obs), add_diag(Sigma_oo, 1e-6));
  ycen  ~ multi_normal(mu_cond, Sigma_cond);
}

generated quantities {
  // ── Log-likelihoods ──────────────────────────────────────────────────────
  real log_lik_obs  = multi_normal_lpdf(y_obs | rep_vector(beta0, N_obs),
                                        add_diag(Sigma_oo, 1e-6));
  real log_lik_cens = multi_normal_lpdf(ycen  | mu_cond, Sigma_cond);
  real log_lik      = log_lik_obs + log_lik_cens;

  // ── Deviance (used for DIC in R) ─────────────────────────────────────────
  // DIC = D_bar + p_D
  //   D_bar        = mean(-2 * log_lik)  over all draws   [computed in R]
  //   D(theta_bar) = -2 * log_lik at posterior mean       [computed in R]
  //   p_D          = D_bar - D(theta_bar)
  real deviance = -2 * log_lik;
}



