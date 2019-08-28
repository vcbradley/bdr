
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
    K_lin = kernelMatrix(kern_lin, matrix(X[, kernel_params$linear_ind]), matrix(Y[, kernel_params$linear_ind]))
  }else{
    K_lin = matrix(0, k_rows, k_cols)
  }
  
  if(!is.null(kernel_params$rbf_ind)){
    kern_rbf = rbfdot(sigma = kernel_params$sigma)
    K_rbf = kernelMatrix(kern_lin, matrix(X[, kernel_params$rbf_ind]), matrix(Y[, kernel_params$rbf_ind]))
  }else{
    K_rbf = matrix(0, k_rows, k_cols)
  }
  
  if(!is.null(kernel_params$matern_ind)){
    K_matern = matern.cov(matrix(X[, kernel_params$matern_ind])
                          , matrix(Y[, kernel_params$matern_ind])
                          , theta = kernel_params$theta
                          , smoothness = kernel_params$smoothness
                          , scale = kernel_params$scale
    )
  }else{
    K_matern = matrix(0, k_rows, k_cols)
  }
  
  K_full = K_lin + K_rbf + K_matern
  
  # this takes a while ugh
  K_new = nearPD(K_full)$mat
  
  return(K_new)
}


