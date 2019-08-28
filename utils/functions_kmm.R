
doKMM = function(X_trn, X_tst
                 , B = 1
                 , kernel_params
){
  require(Matrix)
  require(kernlab)
  require(fields)
  require(quadprog)
  
  if(is.null(ncol(X_trn))){
    n_trn = length(X_trn)
    n_tst = length(X_tst)
  }else{
    n_trn = nrow(X_trn)
    n_tst = nrow(X_tst)
  }
  
  eps = B/sqrt(n_trn)  # set epsilon based on B and suggested value from Gretton chapter; this constraint ensures that  Beta * the training dist is close to a probability dist
  
  K = getCustomKern(X_trn, kernel_params)
  # fix to make sure we can use Cholesky decomp
  newK = nearPD(K)$mat
  
  kappa = getCustomKern(X_trn, X_tst, kernel_params)
  kappa = (n_trn/n_tst) * rowSums(kappa)
  
  G = as.matrix(rbind(- rep(1, n_trn)
                      , rep(1, n_trn)
                      , - diag(n_trn)
                      , diag(n_trn)
  ))
  h = c(- n_trn * (1 + eps)
        , n_trn * (1 - eps)
        , - B * rep(1, n_trn)
        , rep(0, n_trn)
  )
  
  sol = solve.QP(Dmat = newK, dvec = kappa, Amat = t(G), bvec = h)
  
  return(sol)
}



getWeights = function(data, vars, train_ind, target_ind, weight_col = NULL
                      , B = 1
                      , kernel_type = 'linear'
                      , sigma = 1  #rbf params
                      , theta = 1.0, smoothness = 0.5, scale=1 #matern params
){
  
  fmla = as.formula(paste0('~ -1 +', paste(vars, collapse = '+')))
  X = model.matrix(object = fmla, data = data)  #not all levels
  
  # if weights are supplied, multiply them through the observations of X
  if(!is.null(weight_col)){
    X = diag(data[, get(weight_col)]) %*% X
  }
  
  X_train = X[which(data[, get(train_ind)] == 1),]    # data to weight
  X_target = X[which(data[, get(target_ind)] == 1),]  # target
  
  weighted = doKMM(X_trn = X_train
                   , X_tst = X_target
                   , B = B
                   , kernel_type = kernel_type
                   , sigma = sigma
                   , theta = theta, smoothness = smoothness, scale = scale
  )  # the smaller sigma is, the more weighting that happens 
  
  # calculate weights
  if(is.null(nrow(X_train))){
    n_train = length(X_train)
  }else{
    n_train = nrow(X_train)
  }
  weighted$weights = (n_train/sum(weighted$solution)) * weighted$solution
  
  return(weights = weighted$weights)
}
