
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
  ## fix bags
  cat("\tGetting bags\n")
  
  bags = getBags(data = data[surveyed == 1,]
                 , vars = vars$file_and_survey
                 , n_bags = dist_reg_params$n_bags
                 , newdata = data[, vars$file_and_survey, with = F])
  
  # if we're refitting bags, create new bag just using the unmatched survey data
  if(dist_reg_params$refit_bags){
    bags_unm = getBags(data = data[unmatched == 1,]
                       , vars = vars$file_and_survey
                       , n_bags = dist_reg_params$n_bags
                       , newdata = data[, vars$file_and_survey, with = F])
  }else{
    bags_unm = bags
  }
  # give each matched data point its own bag
  bags_unm$bags_newdata[data$matched == 1] <- seq(dist_reg_params$n_bags + 1, length = sum(data$matched))
  
  #-----------------------------------------
  ## Run models
  results = list()
  
  ## Basic LASSO
  cat("\tFitting LOGIT\t")
  lasso_frmla = as.formula(paste0("~", paste(dist_reg_params$regression_vars, collapse = '+')))
  X_lasso = modmat_all_levs(data, formula = lasso_frmla)
  
  lasso_fit = tryN(fitLasso(mu_hat = X_lasso[which(data$matched == 1), ]
                            , Y_bag = as.matrix(data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                            , phi_x = X_lasso
                            , family = 'multinomial'
  ))
  
  setnames(lasso_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
  
  # cat(calcMSE(Y = as.numeric(unlist(data[holdout == 1, dist_reg_params$outcome, with = F]))
  #         , as.numeric(unlist(lasso_fit$Y_hat[data$holdout == 1,]))), '\n')
  
  results[['logit']] = lasso_fit$Y_hat
  
  
  
  ## LASSO - ALL DATA
  cat("\tFitting LASSO - ALL DATA\t")
  lasso_alldata_fit = tryN(fitLasso(mu_hat = X_lasso
                                    , Y_bag = as.matrix(data[, .SD, .SDcols = dist_reg_params$outcome])
                                    , phi_x = X_lasso
                                    , family = 'multinomial'
  ))
  setnames(lasso_alldata_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
  
  # cat(calcMSE(Y = as.numeric(unlist(data[holdout == 1, dist_reg_params$outcome, with = F]))
  #         , as.numeric(unlist(lasso_alldata_fit$Y_hat[data$holdout == 1,]))), '\n')
  
  results[['logit_alldata']] = lasso_alldata_fit$Y_hat
  
  
  ### fit DR -- Linear
  cat("\tFitting DR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr_linear = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr_linear$mse_test, '\n')
  results[['dr_linear']] = fit_dr_linear$y_hat
  
  
  
  ### fit DR -- weighted & Linear
  cat("\tFitting WDR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags
  
  fit_wdr_linear = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_wdr_linear$mse_test, '\n')
  results[['wdr_linear']] = fit_wdr_linear$y_hat
  
  
  
  ### fit DR - NO WEIGHTING
  cat("\tFitting DR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr$mse_test, '\n')
  results[['dr']] = fit_dr$y_hat
  
  
  
  ### fit DR -- WEIGHTED
  cat("\tFitting WDR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags
  
  fit_wdr = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_wdr$mse_test, '\n')
  results[['wdr']] = fit_wdr$y_hat
  
  
  ### fit DR - NO WEIGHTING -- CUSTOM KERNEL
  cat("\tFitting DR - Custom\t")
  dist_reg_params$kernel_type = 'rbf_age'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr_cust = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr_cust$mse_test, '\n')
  results[['dr_cust']] = fit_dr_cust$y_hat
  
  
  
  ### fit DR - SEP BAGS
  cat("\tFitting SEP DR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  
  # fit model with sigma
  fit_dr_sepbags = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr_sepbags$mse_test, '\n')
  results[['dr_sepbags']] = fit_dr_sepbags$y_hat
  
  
  
  ### fit DR - SEP BAGS - WEIGHTED
  cat("\tFitting SEP WDR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags_unm
  
  fit_wdr_sepbags = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_wdr_sepbags$mse_test, '\n')
  results[['wdr_sepbags']] = fit_wdr_sepbags$y_hat
  
  
  
  ### fit DR - SEP BAGS - LINEAR
  cat("\tFitting SEP DR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  fit_dr_sepbags_lin = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr_sepbags_lin$mse_test, '\n')
  results[['dr_sepbags_lin']] = fit_dr_sepbags_lin$y_hat
  
  
  ### fit DR - SEP BAGS - CUSTOM KERNEL
  cat("\tFitting SEP DR - Custom\t")
  dist_reg_params$kernel_type = 'rbf_age'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  fit_dr_sepbags_cust = tryN(doBasicDR(data = data, dist_reg_params))
  #cat(fit_dr_sepbags_cust$mse_test, '\n')
  results[['dr_sepbags_cust']] = fit_dr_sepbags_cust$y_hat
  
  
  #------------------------------------------------------------------------
  ###### calculate group means - re-run after running DR because bags will be re-fit
  cat("\tGroup means\t")
  data[, bag := bags$bags_newdata]
  Y_grp_means = data[surveyed == 1, lapply(.SD, mean), .SDcols = dist_reg_params$outcome, by = bag]
  setnames(Y_grp_means, c('bag',gsub('_','_hat_',dist_reg_params$outcome)))
  
  Y_grp_means = merge(data[, .(bag)], Y_grp_means, by = 'bag', all.x = T)
  
  Y_grp_means[is.na(y_hat_dem), y_hat_dem := 0]
  Y_grp_means[is.na(y_hat_rep), y_hat_rep := 0]
  Y_grp_means[is.na(y_hat_oth), y_hat_oth := 0]
  
  # add to results
  results[['grpmean']] = Y_grp_means[, -1]
  
  return(results)
}



