


// closed-form posterior for x, but not y

data {
  
  int<lower = 1> n; // number of observations
  int<lower = 1> p; // numer of predictors
  int<lower = 1> d; // dimension of outcome
  
  //int<lower = 1, upper = n> ind_matched;
  //int<lower = 1, upper = n> ind_svyd;
  int<lower = 1, upper = n> ind_training;
  //int<lower = 1, upper = n> ind_holdout;
  
  //int<lower = 1, upper = p> ind_reg_vars;
  //int<lower = 1, upper = p> ind_bag_vars;
  
  matrix[n, p] X;
  matrix[n, d] Y;
}


transformed data {
  int n_svy;
  n_svy = sum(in_svyd);
  
  //int n_p_reg;
  //n_p_reg = sum(ind_reg_vars);
  
  //int n_p_bag;
  //n_p_bag = sum(n_p_bag);
  
  // make matrix just for bagging
  matrix[n_svy, n_p_bag] X_bagging;
  X_bagging = X[ind_svyd, ind_bag_vars];
  
  
}


parameters {
  
  // int<lower = 1, upper = n_svy> k; // number of clusters
  // int<lower = 1, upper = n_file> m; // numer of landmarks
  
  // length-scale parameter
  real<lower = 0> sigma;
  

}


model {
  
  
}





