getKernParams = function(X, kernel_type, sigma = NULL){
  kernel_params = list()
  
  if(is.null(sigma) & kernel_type %in% c('rbf', 'rbf_age')){
    stop("You must supply sigma ")
  } 
  
  if(kernel_type %in% c('rbf', 'rbf_age')){
    kernel_params[['sigma']] = sigma
  }
  
  if(kernel_type == 'linear'){
    kernel_params[['linear_ind']] = 1:ncol(X)
  }else if(kernel_type == 'rbf'){
    kernel_params[['rbf_ind']] = 1:ncol(X)
  }else if(kernel_type == 'rbf_age'){
    #kernel_params[['lin_ind']] = which(!colnames(X) == 'age_scaled')
    kernel_params[['linear_ind']] = 1:ncol(X)
    kernel_params[['rbf_ind']] = which(colnames(X) == 'age_scaled')
  }else{
    stop("Kernel type not supported")
  }
  
  return(kernel_params)
}


getSigma = function(X, Y, sigma_type){
  kern_rbf_1 = rbfdot(sigma = 1)
  K_rbf_1 = kernelMatrix(kern_rbf_1
                         , as.matrix(X)
                         , as.matrix(Y)
  )
  
  # rbf kernel is k(x,x') = \exp(-σ \|x - x'\|^2) , we want sigma to be approx 1/median(\|x - x'\|^2)
  K_rbf_1 = -log(K_rbf_1)
  if(sigma_type == 'mean'){
    sigma = 1/mean(K_rbf_1)
  }else{
    sigma = 1/median(K_rbf_1)
  }
  
  cat('sigma from ',sigma_type,':', sigma, '\n')
  return(sigma)
}

# create a custom kernel that is a combination of linear, matern and RBF kernels
getCustomKern = function(X, Y = NULL, kernel_params, comb.fun = 'sum'){
  require(kernlab)
  require(Matrix)
  require(fields)
  
  if(is.null(Y)){
    Y = X
  }
  
  k_rows = nrow(X)
  k_cols = nrow(Y)
  
  sigmas = list()
  
  if(comb.fun == 'sum'){
    filler = 0
  }else{
    filler = 1
  }
  
  ## Linear kernel
  if(!is.null(kernel_params$linear_ind)){
    kern_lin = vanilladot()
    K_lin = kernelMatrix(kern_lin
                         , as.matrix(X[, kernel_params$linear_ind])
                         , as.matrix(Y[, kernel_params$linear_ind]))
  }else{
    K_lin = matrix(filler, k_rows, k_cols)
  }
  
  ## RBF Kernel
  if(!is.null(kernel_params$rbf_ind)){
    
    # set sigma using median heuristic
    if(kernel_params$sigma %in% c('median', 'mean')){
      sigma = getSigma(X = X[, kernel_params$rbf_ind]
                       , Y[, kernel_params$rbf_ind]
                       , sigma_type = kernel_params$sigma)
    }else{
      sigma = kernel_params$sigma
    }
    sigmas[['rbf']] = sigma
    
    # create kernel
    kern_rbf = rbfdot(sigma = sigma)
    
    # get kernel matrix
    K_rbf = kernelMatrix(kern_rbf
                         , as.matrix(X[, kernel_params$rbf_ind])
                         , as.matrix(Y[, kernel_params$rbf_ind])
                         )
  }else{
    K_rbf = matrix(filler, k_rows, k_cols)
  }
  
  ## Matern kernel
  if(!is.null(kernel_params$matern_ind)){
    K_matern = matern.cov(as.matrix(X[, kernel_params$matern_ind])
                          , as.matrix(Y[, kernel_params$matern_ind])
                          , theta = kernel_params$theta
                          , smoothness = kernel_params$smoothness
                          , scale = kernel_params$scale
    )
  }else{
    K_matern = matrix(filler, k_rows, k_cols)
  }
  
  ## ARD RBF Kernel
  if(!is.null(kernel_params$rbf_ard_ind)){
    
    sigmas_ard = c()
    K_rbf_ard = lapply(kernel_params$rbf_ard_ind, function(i){
      # get sigma
      sigma = getSigma(X = X[, i], Y[, i], sigma_type = kernel_params$sigma)
      #store
      sigmas_ard <<- c(sigmas_ard, sigma)
      
      # create kernel
      rbf = rbfdot(sigma = sigma)
      
      # get kernel matrix
      k = kernelMatrix(rbf, X[, i], Y[, i])
      
      # fill with fillers if needed
      k[is.nan(k)] <- filler
      k
    })
    sigmas[['rbf_ard']] <- sigmas_ard
    
    K_rbf_ard = Reduce(K_rbf_ard, f = ifelse(comb.fun == 'prod','*','+'), accumulate = F)
      
  }else{
    K_rbf_ard = matrix(filler, k_rows, k_cols)
  }
  
  if(comb.fun == 'prod'){
    K_full = K_lin * K_rbf * K_matern * K_rbf_ard
  }else{
    K_full = K_lin + K_rbf + K_matern + K_rbf_ard
  }
  
  
  return(list(K = K_full, sigma = sigmas))
}


