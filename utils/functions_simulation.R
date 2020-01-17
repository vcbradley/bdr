
# function to try models up to 5 times if they hit an error on an iteration (i.e. Lasso doesn't converge)
tryN = function(expr, max_attempts = 5){
  attempts = 1
  error = T
  while(error & (attempts <= max_attempts)){
    cat(attempts, '\n')
    out = try(expr)
    error = (class(out) == 'try-error')
    attempts = attempts + 1
  }
  return(out)
}


runModels = function(data, dist_reg_params, max_attempts = 5){
  
  
  #-----------------------------------------
  ## Run models
  results = list()
  
  lasso_frmla = as.formula(paste0("~", paste(dist_reg_params$regression_vars, collapse = '+')))
  X_lasso = modmat_all_levs(data, formula = lasso_frmla)
  
  ## LASSO - ALL DATA
  if(is.null(dist_reg_params$model_list) | 'logit_alldata' %in% dist_reg_params$model_list){
    cat("\tFitting LASSO - ALL DATA\t")
    lasso_alldata_fit = tryN(fitLasso(mu_hat = X_lasso
                                      , Y_bag = as.matrix(data[, .SD, .SDcols = dist_reg_params$outcome])
                                      , phi_x = X_lasso
                                      , family = dist_reg_params$outcome_family
    ))
    setnames(lasso_alldata_fit$Y_hat, gsub('y_','y_hat_',dist_reg_params$outcome))
    
    # cat(calcMSE(Y = as.numeric(unlist(data[holdout == 1, dist_reg_params$outcome, with = F]))
    #         , as.numeric(unlist(lasso_alldata_fit$Y_hat[data$holdout == 1,]))), '\n')
    
    results[['logit_alldata']] = lasso_alldata_fit$Y_hat
    
  }
  
  
  ## Basic LASSO
  if(is.null(dist_reg_params$model_list) | 'logit' %in% dist_reg_params$model_list){
    cat("\tFitting LOGIT\t")
    
    lasso_fit = tryN(fitLasso(mu_hat = X_lasso[which(data$matched == 1), ]
                              , Y_bag = as.matrix(data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                              , phi_x = X_lasso
                              , family = dist_reg_params$outcome_family
    ))
    
    setnames(lasso_fit$Y_hat, gsub('y_','y_hat_',dist_reg_params$outcome))
    
    # cat(calcMSE(Y = as.numeric(unlist(data[holdout == 1, dist_reg_params$outcome, with = F]))
    #         , as.numeric(unlist(lasso_fit$Y_hat[data$holdout == 1,]))), '\n')
    
    results[['logit']] = lasso_fit$Y_hat
  }
  
  
  ### fit DR -- Linear
  if(is.null(dist_reg_params$model_list) | 'dr_linear' %in% dist_reg_params$model_list){
    cat("\tFitting DR - Linear\t")
    dist_reg_params$kernel_type = 'linear'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag'
    
    fit_dr_linear = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr_linear$mse_test, '\n')
    results[['dr_linear']] = fit_dr_linear$y_hat
  }
  
  ### fit DR - NO WEIGHTING
  if(is.null(dist_reg_params$model_list) | 'dr' %in% dist_reg_params$model_list){
    cat("\tFitting DR - RBF\t")
    dist_reg_params$kernel_type = 'rbf'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag'
    
    fit_dr = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr$mse_test, '\n')
    results[['dr']] = fit_dr$y_hat
  }
  
  ### fit DR - NO WEIGHTING -- CUSTOM KERNEL
  if(is.null(dist_reg_params$model_list) | 'dr_cust' %in% dist_reg_params$model_list){
    cat("\tFitting DR - Custom\t")
    dist_reg_params$kernel_type = 'rbf_age'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag'
    
    fit_dr_cust = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr_cust$mse_test, '\n')
    results[['dr_cust']] = fit_dr_cust$y_hat
  }
  
  
  ### fit DR -- weighted & Linear
  if(is.null(dist_reg_params$model_list) | 'wdr_linear' %in% dist_reg_params$model_list){
    cat("\tFitting WDR - Linear\t")
    
    dist_reg_params$kernel_type = 'linear'
    dist_reg_params$weight_col = 'kmm_weight'
    dist_reg_params$which_bag = 'bag'
    
    fit_wdr_linear = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_wdr_linear$mse_test, '\n')
    results[['wdr_linear']] = fit_wdr_linear$y_hat
  }
  
  ### fit DR -- WEIGHTED
  if(is.null(dist_reg_params$model_list) | 'wdr' %in% dist_reg_params$model_list){
    cat("\tFitting WDR - RBF\t")
    dist_reg_params$kernel_type = 'rbf'
    dist_reg_params$weight_col = 'kmm_weight'
    dist_reg_params$which_bag = 'bag'
    
    fit_wdr = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_wdr$mse_test, '\n')
    results[['wdr']] = fit_wdr$y_hat
  }
  
  
  ### fit DR - SEP BAGS
  if(is.null(dist_reg_params$model_list) | 'dr_sepbags' %in% dist_reg_params$model_list){
    cat("\tFitting SEP DR - RBF\t")
    dist_reg_params$kernel_type = 'rbf'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag_sep'
    
    # fit model with sigma
    fit_dr_sepbags = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr_sepbags$mse_test, '\n')
    results[['dr_sepbags']] = fit_dr_sepbags$y_hat
  }
  
  
  ### fit DR - SEP BAGS - WEIGHTED
  if(is.null(dist_reg_params$model_list) | 'wdr_sepbags' %in% dist_reg_params$model_list){
    cat("\tFitting SEP WDR - RBF\t")
    dist_reg_params$kernel_type = 'rbf'
    dist_reg_params$weight_col = 'kmm_weight'
    dist_reg_params$which_bag = 'bag_sep'
    
    fit_wdr_sepbags = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_wdr_sepbags$mse_test, '\n')
    results[['wdr_sepbags']] = fit_wdr_sepbags$y_hat
  }
  
  ### fit DR - SEP BAGS - LINEAR
  if(is.null(dist_reg_params$model_list) | 'dr_sepbags_lin' %in% dist_reg_params$model_list){
    cat("\tFitting SEP DR - Linear\t")
    dist_reg_params$kernel_type = 'linear'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag_sep'
    
    fit_dr_sepbags_lin = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr_sepbags_lin$mse_test, '\n')
    results[['dr_sepbags_lin']] = fit_dr_sepbags_lin$y_hat
  }
  
  
  ### fit DR - SEP BAGS - CUSTOM KERNEL
  if(is.null(dist_reg_params$model_list) | 'dr_sepbags_cust' %in% dist_reg_params$model_list){
    cat("\tFitting SEP DR - Custom\t")
    dist_reg_params$kernel_type = 'rbf_age'
    dist_reg_params$weight_col = NULL
    dist_reg_params$which_bag = 'bag_sep'
    
    fit_dr_sepbags_cust = tryN(doBasicDR(data = data, dist_reg_params))
    #cat(fit_dr_sepbags_cust$mse_test, '\n')
    results[['dr_sepbags_cust']] = fit_dr_sepbags_cust$y_hat
  }
  
  #### make ecological features
  dist_reg_params$kernel_type = 'rbf_age'
  dist_reg_params$weight_col = NULL
  dist_reg_params$which_bag = 'bag'
  
  if(is.null(dist_reg_params$weight_col)){
    pew_data[, weight := 1]
  }else{
    pew_data[, weight := get(dist_reg_params$weight_col)]
  }
  
  kernel_params = getKernParams(X = dist_reg_params$landmarks$X
                                , kernel_type = dist_reg_params$kernel_type
                                , sigma = dist_reg_params$sigma)
  
  # get features
  cat(paste0(Sys.time(), "\t Making features\n"))
  ecol_features = getFeatures(data = dist_reg_params$landmarks$X
                              , bag = as.numeric(pew_data[, get(dist_reg_params$which_bag)])
                              , train_ind = as.numeric(pew_data[, get(dist_reg_params$train_ind)])
                              , landmarks = dist_reg_params$landmarks$landmarks
                              , kernel_params = kernel_params
                              , weight = as.numeric(pew_data[, weight]))
  
  
  if(is.null(dist_reg_params$model_list) | 'logit_bagfeat' %in% dist_reg_params$model_list){
    cat("\tFitting Logit w/ ecol features - bags\t")
    
    X_bag = merge(data.table('bag' = ecol_features$phi_x$bag), ecol_features$mu_hat, by = 'bag', sort = F, all.x = T)
    X_bag = scale(X_bag)
    X_bag_mat = cbind(X_lasso, as.matrix(X_bag[, 2:ncol(X_bag)]))
    
    fit_logit_bagfeat = tryN(fitLasso(mu_hat = X_bag_mat[which(pew_data$matched == 1), ]
                              , Y_bag = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                              , phi_x = X_bag_mat
                              , family = dist_reg_params$outcome_family
    ))
    
    setnames(fit_logit_bagfeat$Y_hat, gsub('y_','y_hat_', dist_reg_params$outcome))
    results[['logit_bagfeat']] = fit_logit_bagfeat$Y_hat
  }
  
  
  if(is.null(dist_reg_params$model_list) | 'logit_regfeat' %in% dist_reg_params$model_list){
    cat("\tFitting Logit w/ ecol features - region\t")
    
    mu_hat_reg = aggregate(ecol_features$phi_x, by = list('demo_region' = pew_data$demo_region), FUN = mean)
    X_reg = merge(data.table('demo_region' = pew_data$demo_region), mu_hat_reg, by = 'demo_region', sort = F, all.x = T)
    X_reg = scale(as.matrix(X_reg[, 2:ncol(X_reg), with = F]))
    X_reg_mat = cbind(X_lasso, X_reg)
    
    fit_logit_bagfeat = tryN(fitLasso(mu_hat = X_reg_mat[which(pew_data$matched == 1), ]
                                      , Y_bag = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                                      , phi_x = X_reg_mat
                                      , family = dist_reg_params$outcome_family
    ))
    
    setnames(fit_logit_bagfeat$Y_hat, gsub('y_','y_hat_', dist_reg_params$outcome))
    results[['logit_regfeat']] = fit_logit_bagfeat$Y_hat
  }
  
  
  #------------------------------------------------------------------------
  ###### calculate group means - re-run after running DR because bags will be re-fit
  cat("\tGroup means\t")
  Y_grp_means = data[surveyed == 1, lapply(.SD, mean), .SDcols = dist_reg_params$outcome, by = bag]
  setnames(Y_grp_means, c('bag',gsub('_','_hat_',dist_reg_params$outcome)))
  
  Y_grp_means = merge(data[, .(bag)], Y_grp_means, by = 'bag', all.x = T)
  
  Y_grp_means[is.na(Y_grp_means)] <- 0
  # Y_grp_means[is.na(y_hat_dem), y_hat_dem := 0]
  # Y_grp_means[is.na(y_hat_rep), y_hat_rep := 0]
  # Y_grp_means[is.na(y_hat_oth), y_hat_oth := 0]
  
  # add to results
  results[['grpmean']] = Y_grp_means[, -1]
  
  return(results)
}



