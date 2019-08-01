


// closed-form posterior for x, but not y
// fixed number of bags and landmarks

data {
  
  //int<lower = 1> n; // number of observations
  int<lower = 1> n_bags;
  int<lower = 1> p; // numer of predictors
  int<lower = 1> d; // dimension of outcome
  int<lower = 1> b; // number of bags
  
  int<lower = 1, upper = b> bags; // bag assignments for observations in X
  
  matrix[b, p] mu; // empirical mean embeddings for each bag
  matrix[b, d] Y; // outcome by bag
  
  //matrix[n, p] X; // for prediction at individ level
}



parameters {
  
  // intercept
  real[d] alpha;
  
  vector[d] beta[p]; //
  

}


model {
  
  for(j in 1:n_bags)
    y[j] ~ multinomial(alpha + mu[j,] * beta)
}





