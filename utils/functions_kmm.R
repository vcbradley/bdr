
doKMM = function(X_trn, X_tst
                 , kernel_params
                 , B = c(1,1)
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
  
  eps = B[2]/sqrt(n_trn)  # set epsilon based on B and suggested value from Gretton chapter; this constraint ensures that  Beta * the training dist is close to a probability dist
  
  K = getCustomKern(X_trn, kernel_params = kernel_params)
  # fix to make sure we can use Cholesky decomp
  newK = nearPD(K)$mat
  
  kappa = getCustomKern(X_trn, X_tst, kernel_params = kernel_params)
  kappa = (n_trn/n_tst) * rowSums(kappa)
  
  G = as.matrix(rbind(- rep(1, n_trn)
                      , rep(1, n_trn)
                      , - diag(n_trn)
                      , diag(n_trn)
  ))
  h = c(- n_trn * (1 + eps)
        , n_trn * (1 - eps)
        , - B[2] * rep(1, n_trn)
        , rep(B[1], n_trn)  # lower bound on the weights
  )
  
  sol = solve.QP(Dmat = newK, dvec = kappa, Amat = t(G), bvec = h)
  
  return(sol)
}



getWeights = function(data, vars, train_ind, target_ind
                      , kernel_type
                      , weight_col = NULL
                      , B = c(1,1)
                      , sigma = NULL
){
  
  fmla = as.formula(paste0('~ -1 +', paste(vars, collapse = '+')))
  X = model.matrix(object = fmla, data = data)  #not all levels
  
  # if weights are supplied, multiply them through the observations of X
  if(!is.null(weight_col)){
    X = diag(data[, get(weight_col)]) %*% X
  }
  
  X_train = as.matrix(X[which(data[, get(train_ind)] == 1),])    # data to weight
  X_target = as.matrix(X[which(data[, get(target_ind)] == 1),])  # target
  
  
  ## SET KERNEL
  kernel_params = getKernParams(X = X_train, sigma = sigma)
  
  # GET WEIGHTS
  weighted = doKMM(X_trn = X_train
                   , X_tst = X_target
                   , B = B
                   , kernel_params = kernel_params
                   )
  
  # calculate weights
  if(is.null(nrow(X_train))){
    n_train = length(X_train)
  }else{
    n_train = nrow(X_train)
  }
  weighted$weights = (n_train/sum(weighted$solution)) * weighted$solution
  
  return(weights = weighted$weights)
}


### replacing with nearPD from the Matrix package
# ridgeMat = function(origMat){
#   cholStatus <- try(u <- chol(origMat), silent = TRUE)
#   cholError <- ifelse(class(cholStatus) == "try-error", TRUE, FALSE)
#   
#   newMat <- origMat
#   iter <- 0
#   while (cholError) {
#     
#     iter <- iter + 1
#     cat("iteration ", iter, "\n")
#     
#     # replace -ve eigen values with small +ve number
#     newEig <- eigen(newMat)
#     newEig2 <- newEig$values + 2 *abs(min(newEig$values))# + 1e-10  # add in the min eigenvalue
#     #newEig2 <- ifelse(newEig$values < 1e-10, 1e-10, newEig$values)
#     
#     # create modified matrix eqn 5 from Brissette et al 2007, inv = transp for
#     # eig vectors
#     newMat <- newEig$vectors %*% diag(newEig2) %*% t(newEig$vectors)
#     
#     # normalize modified matrix eqn 6 from Brissette et al 2007
#     newMat <- newMat/sqrt(diag(newMat) %*% t(diag(newMat)))
#     
#     # try chol again
#     cholStatus <- try(u <- chol(newMat), silent = TRUE)
#     cholError <- ifelse(class(cholStatus) == "try-error", TRUE, FALSE)
#   }
#   return(newMat)
# }

