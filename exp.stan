functions {
  matrix covariance_exponential(int N, matrix H, real sigmasq, real phi, real nugget) {
    matrix[N, N] Sigma;
    for (i in 1:N) {
      for (j in i:N) {
        if (i == j) {
          Sigma[i, j] = sigmasq + nugget;
        } else {
          Sigma[i, j] = sigmasq * exp(-H[i,j]/phi);
          Sigma[j, i] = Sigma[i, j];
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
  matrix[N, N] Sigma = covariance_exponential(N, H, sigmasq, phi, nugget);
  
  // Partition covariance matrices
  matrix[N_obs, N_obs] Sigma_oo;
  matrix[N_cens, N_cens] Sigma_cc;
  matrix[N_cens, N_obs] Sigma_co;
  
  for (i in 1:N_obs) {
    for (j in 1:N_obs) {
      Sigma_oo[i, j] = Sigma[obs_idx[i], obs_idx[j]];
    }
  }
  
  for (i in 1:N_cens) {
    for (j in 1:N_cens) {
      Sigma_cc[i, j] = Sigma[cens_idx[i], cens_idx[j]];
    }
  }
  
  for (i in 1:N_cens) {
    for (j in 1:N_obs) {
      Sigma_co[i, j] = Sigma[cens_idx[i], obs_idx[j]];
    }
  }
}

model {
  matrix[N_obs, N_obs] Sigma_oo_inv = inverse(Sigma_oo);
  vector[N_cens] mu_cens = rep_vector(beta0, N_cens) + 
                           Sigma_co * Sigma_oo_inv * (y_obs - rep_vector(beta0, N_obs));
  matrix[N_cens, N_cens] Sigma_cond = Sigma_cc - Sigma_co * Sigma_oo_inv * Sigma_co';
  
  // Priors
  beta0 ~ normal(0, 50);
  //sigmasq ~ student_t(4, 0, 5);
  sigmasq ~ inv_gamma(2.1,6.6);
  //phi ~ student_t(4, 0, 5);
  phi ~ gamma(2,0.1);
  //nugget ~ student_t(4, 0, 5);
  nugget ~inv_gamma(2.1,0.55);
  // Likelihood
  y_obs ~ multi_normal(rep_vector(beta0, N_obs), Sigma_oo);
  ycen ~ multi_normal(mu_cens, Sigma_cond);
}

generated quantities {
  vector[N_obs] log_lik_obs;
  vector[N_cens] log_lik_cens;
  real log_lik;

  {
    matrix[N_obs, N_obs] Sigma_oo_inv = inverse(Sigma_oo);
    vector[N_cens] mu_cens = rep_vector(beta0, N_cens) + 
                             Sigma_co * Sigma_oo_inv * (y_obs - rep_vector(beta0, N_obs));
    matrix[N_cens, N_cens] Sigma_cond = Sigma_cc - Sigma_co * Sigma_oo_inv * Sigma_co';
    
    for (i in 1:N_obs) {
      log_lik_obs[i] = normal_lpdf(y_obs[i] | beta0, sqrt(Sigma_oo[i,i]));
    }
    
    for (i in 1:N_cens) {
      log_lik_cens[i] = normal_lpdf(ycen[i] | mu_cens[i], sqrt(Sigma_cond[i,i]));
    }
    // Total log-likelihood = sum of both parts
    log_lik = sum(log_lik_obs) + sum(log_lik_cens);

  }
}

