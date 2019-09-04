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

# create a custom kernel that is a combination of linear, matern and RBF kernels
getCustomKern = function(X, Y = NULL, kernel_params){
  require(kernlab)
  require(Matrix)
  require(fields)
  
  if(is.null(Y)){
    Y = X
  }
  
  k_rows = nrow(X)
  k_cols = nrow(Y)
  
  if(!is.null(kernel_params$linear_ind)){
    kern_lin = vanilladot()
    K_lin = kernelMatrix(kern_lin
                         , as.matrix(X[, kernel_params$linear_ind])
                         , as.matrix(Y[, kernel_params$linear_ind]))
  }else{
    K_lin = matrix(0, k_rows, k_cols)
  }
  
  if(!is.null(kernel_params$rbf_ind)){
    
    # set sigma using median heuristic
    if(kernel_params$sigma == 'median'){
      kern_rbf_1 = rbfdot(sigma = 1)
      K_rbf_1 = kernelMatrix(kern_rbf_1
                             , as.matrix(X[, kernel_params$rbf_ind])
                             , as.matrix(Y[, kernel_params$rbf_ind])
                             )
      
      # rbf kernel is k(x,x') = \exp(-σ \|x - x'\|^2) , we want sigma to be approx 1/median(\|x - x'\|^2)
      K_rbf_1 = -log(K_rbf_1)
      sigma_med = 1/median(K_rbf_1)
      cat('sigma from median:', sigma_med, '\n')
      
      kern_rbf = rbfdot(sigma = sigma_med)
    }else{
      kern_rbf = rbfdot(sigma = kernel_params$sigma)
    }

    
    K_rbf = kernelMatrix(kern_rbf
                         , as.matrix(X[, kernel_params$rbf_ind])
                         , as.matrix(Y[, kernel_params$rbf_ind])
                         )
  }else{
    K_rbf = matrix(0, k_rows, k_cols)
  }
  
  if(!is.null(kernel_params$matern_ind)){
    K_matern = matern.cov(as.matrix(X[, kernel_params$matern_ind])
                          , as.matrix(Y[, kernel_params$matern_ind])
                          , theta = kernel_params$theta
                          , smoothness = kernel_params$smoothness
                          , scale = kernel_params$scale
    )
  }else{
    K_matern = matrix(0, k_rows, k_cols)
  }
  
  K_full = K_lin + K_rbf + K_matern
  
  return(K_full)
}


