rm(list = ls())
library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)
library(kernlab)
library(R.utils)
library(glmnet)
library(doMC)
library(parallel)

detectCores(all.tests = FALSE, logical = TRUE)
registerDoMC(8)
getDoParWorkers()

setwd('~/github/bdr')
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

results_file = '~/github/bdr/pew-experiment/results/pew_simulation_results.csv'

#-----------------------------------------------
### Read in data
pew_data = fread('data/data_recoded.csv')


n_holdout = 1000
n_surveyed = 2000

#-----------------------------------------------
### MAKE PARAM GRID
party_list = c('insurvey', 'onfile')
match_rate_list = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1)
n_bags_list = c(50, 75, 100, 125)
n_landmarks_list = c(75, 100, 150, 200, 300, 400)
refit_bags_list = c(F, T)


sim_params = expand.grid(match_rate = match_rate_list
                         , n_bags = n_bags_list
                         , n_landmarks = n_landmarks_list
                         , refit_bags = refit_bags_list
                         , party = party_list
                         )
sim_params

# #-----------------------------------------------
# # Make test and train sets - FIXED FOR SIMULATION
# n_holdout = 1000
# n_surveyed = 2000
# 
# ## holdout
# holdout_ind = sample.int(size = n_holdout, nrow(pew_data))
# pew_data[, holdout := 0]
# pew_data[holdout_ind, holdout := 1]
# 
# ## surveyed
# surveyed_ind = sample(which(pew_data$holdout == 0)
#                       , size = n_surveyed
#                       , prob = pew_data[holdout == 0, p_surveyed])
# pew_data[, surveyed := 0]
# pew_data[surveyed_ind, surveyed := 1]
# 
# ## matched
# matched_ind = c()
# possible_ind = surveyed_ind
# while(length(possible_ind) > 0){
#   # append another 10
#   matched_ind = c(matched_ind, sample(possible_ind, size = 10, prob = pew_data[possible_ind, p_matched]))
#   
#   #eliminate those chosen
#   possible_ind = surveyed_ind[!surveyed_ind %in% matched_ind]
# }
# length(matched_ind)
# 
# #voterfile
# pew_data[surveyed == 0 & holdout == 0, voterfile := 1]


#---------------------------------------------------------------------------
##categorize variables
vars = list()
vars$all = names(pew_data)[grepl('demo|^age_scaled', names(pew_data))]
vars$all = vars$all[-which(grepl('demo_state|month_called', vars$all))]

vars$file_and_survey = c('demo_sex', 'demo_age_bucket', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')

# make two of each of the following that we'll switch to given the particular settings
vars$survey_partyonfile = c('demo_mode', 'demo_education', 'demo_phonetype', 'demo_ideology')
vars$survey_partyinsurvey = c(vars$survey_partyonfile, 'demo_party')
vars$file_only_partyonfile = vars$all[!vars$all %in% c(vars$survey_partyonfile, vars$file_and_survey)]
vars$file_only_partyinsurvey = vars$all[!vars$all %in% c(vars$survey_partyinsurvey, vars$file_and_survey)]



#---------------------------------------------------------------------------
## RUN SIMULATION
results_mses = foreach(i=1:10) %dopar%{
  
  #run_settings = data.frame(match_rate = 0.1, n_bags = 125, n_landmarks = 100, refit_bags = F, party = 'onfile')
  # SET PARAMS
  run_settings = sim_params[i,]
  cat(i, '\n')
  cat(paste(names(run_settings),run_settings), '\n')
  
  results_id = paste0('party',run_settings$party
                      , '_match', round(run_settings$match_rate * 100)
                      , '_bags', run_settings$n_bags
                      , '_lmks', run_settings$n_landmarks
                      , '_refitbags', run_settings$refit_bags)
  

  
  #-----------------------------------------
  # Set vars
  if(run_settings$party == 'insurvey'){
    vars$survey = vars$survey_partyinsurvey
    vars$file_only = vars$file_only_partyinsurvey
  }else{
    vars$survey = vars$survey_partyonfile
    vars$file_only = vars$file_only_partyonfile
  }
  
  
  #-----------------------------------------
  ## get test train sets
  cat("\tGetting test/train sets\n")
  testtrain = getTestTrain(data = pew_data
                           , n_holdout = n_holdout
                           , n_surveyed = n_surveyed
                           , n_matched = n_surveyed * run_settings$match_rate
                           , p_surveyed = pew_data$p_surveyed
                           , p_matched = pew_data$p_matched
  )
  pew_data[, unmatched := as.numeric(surveyed == 1 & matched == 0)]
  pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]
  
  
  
  #-----------------------------------------
  ## WEIGHT DATA
  cat("\tWeighting data\n")
  
  pew_data[, kmm_weight := 1]
  
  # weight non-age file vars
  w_matched = getWeights(data = pew_data
                         , vars = c(vars$file_and_survey, vars$file_only)
                         , train_ind = 'matched'
                         , target_ind = 'voterfile'
                         , kernel_type = 'rbf_age' 
                         , sigma = 0.1
                         , B = c(0.1, 5))
  # set weights
  pew_data[matched == 1, kmm_weight := w_matched]
  
  if(run_settings$match_rate < 1){
    w_unmatched = getWeights(data = pew_data
                             , vars = c(vars$file_and_survey, vars$survey)
                             , train_ind = 'unmatched'
                             , target_ind = 'matched'
                             , weight_col = 'kmm_weight'
                             , kernel_type = 'linear' #important for weighting so that marginals match
                             , B = c(0.01, 7))
    
    pew_data[unmatched == 1, kmm_weight := w_unmatched]
  }
  
  
  
  #-----------------------------------------
  ## fix landmarks
  cat("\tGetting landmarks\n")
  n_landmarks = run_settings$n_landmarks
  regression_vars = c(vars$file_and_survey, vars$file_only)
  
  landmarks = getLandmarks(data = pew_data
                           , vars = regression_vars
                           , n_landmarks = n_landmarks
                           , subset_ind = (pew_data[, voterfile] == 1))
  
  #-----------------------------------------
  ## fix bags
  cat("\tGetting bags\n")
  n_bags = run_settings$n_bags
  
  bags = getBags(data = pew_data[surveyed == 1,]
                 , vars = vars$file_and_survey
                 , n_bags = n_bags
                 , newdata = pew_data[, vars$file_and_survey, with = F])
  
  # give each matched data point its own bag
  if(run_settings$refit_bags){
    bags_unm = getBags(data = pew_data[unmatched == 1,]
                       , vars = vars$file_and_survey
                       , n_bags = n_bags
                       , newdata = pew_data[, vars$file_and_survey, with = F])
  }else{
    bags_unm = bags
  }
  bags_unm$bags_newdata[pew_data$matched == 1] <- seq(n_bags + 1, length = sum(pew_data$matched))
  
  
  #-----------------------------------------
  #### SET PARAMETERS ####
  dist_reg_params = list(sigma = 'median' # use median heuristic
                         , bags = bags
                         , landmarks = landmarks
                         , make_bags_vars = vars$file_and_survey
                         , score_bags_vars = vars$file_and_survey
                         , regression_vars = regression_vars
                         , outcome = c('y_dem', 'y_rep', 'y_oth')
                         , n_bags = n_bags
                         , outcome_family = 'multinomial'
                         , train_ind = 'voterfile'
                         , test_ind = 'holdout'
                         , bagging_ind = 'surveyed'
                         , kernel_type = 'rbf'
                         , weight_col = NULL)
  
  
  #--------------------------------------------------
  # RUN MODELS
  results_temp = runModels(dist_reg_params, max_attempts = 5)
  
  results = rbindlist(lapply(names(results_temp), function(r){
    cbind(results_id
          , match_rate = run_settings$match_rate
          , n_bags = run_settings$n_bags
          , n_landmarks = run_settings$n_landmarks
          , refit_bags = run_settings$refit_bags
          , party = run_settings$party
          , model = r
          , results_temp[[r]]
          , holdout = pew_data$holdout)
  }))
  
  # write results out to file
  write.table(results, file = paste0('~/github/bdr/pew-experiment/results/simulation/', results_id, '.csv')
              , row.names = F, sep = ',')
  
  # calculate MSEs
  mses = unlist(lapply(results_temp, function(r){
    calcMSE(Y = as.numeric(unlist(pew_data[, dist_reg_params$outcome, with = F]))
            , Y_pred = as.numeric(unlist(r)))
  }))
  
  data.table(results_id
             , match_rate = run_settings$match_rate
             , n_bags = run_settings$n_bags
             , n_landmarks = run_settings$n_landmarks
             , refit_bags = run_settings$refit_bags
             , model = names(mses)
             , mses
             )
  
}


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




runModels = function(dist_reg_params, max_attempts = 5){
  results = list()
  
  ## Basic LASSO
  cat("\tFitting LOGIT\t")
  lasso_frmla = as.formula(paste0("~", paste(dist_reg_params$regression_vars, collapse = '+')))
  X_lasso = modmat_all_levs(pew_data, formula = lasso_frmla)
  
  lasso_fit = tryN(fitLasso(mu_hat = X_lasso[which(pew_data$matched == 1), ]
                                 , Y_bag = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                                 , phi_x = X_lasso
                                 , family = 'multinomial'
  ))

  setnames(lasso_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
  
  # cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
  #         , as.numeric(unlist(lasso_fit$Y_hat[pew_data$holdout == 1,]))), '\n')
  
  results[['logit']] = lasso_fit$Y_hat
  
  
  
  ## LASSO - ALL DATA
  cat("\tFitting LASSO - ALL DATA\t")
  lasso_alldata_fit = tryN(fitLasso(mu_hat = X_lasso
                                   , Y_bag = as.matrix(pew_data[, .SD, .SDcols = dist_reg_params$outcome])
                                   , phi_x = X_lasso
                                   , family = 'multinomial'
  ))
  setnames(lasso_alldata_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
  
  # cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
  #         , as.numeric(unlist(lasso_alldata_fit$Y_hat[pew_data$holdout == 1,]))), '\n')
  
  results[['logit_alldata']] = lasso_alldata_fit$Y_hat
  
  
  ### fit DR -- Linear
  cat("\tFitting DR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr_linear = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr_linear$mse_test, '\n')
  results[['dr_linear']] = fit_dr_linear$y_hat
  
  
  
  ### fit DR -- weighted & Linear
  cat("\tFitting WDR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags
  
  fit_wdr_linear = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_wdr_linear$mse_test, '\n')
  results[['wdr_linear']] = fit_wdr_linear$y_hat
  
  
  
  ### fit DR - NO WEIGHTING
  cat("\tFitting DR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr$mse_test, '\n')
  results[['dr']] = fit_dr$y_hat
  
  
  
  ### fit DR -- WEIGHTED
  cat("\tFitting WDR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags
  
  fit_wdr = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_wdr$mse_test, '\n')
  results[['wdr']] = fit_wdr$y_hat
  
  
  ### fit DR - NO WEIGHTING -- CUSTOM KERNEL
  cat("\tFitting DR - Custom\t")
  dist_reg_params$kernel_type = 'rbf_age'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags
  
  fit_dr_cust = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr_cust$mse_test, '\n')
  results[['dr_cust']] = fit_dr_cust$y_hat
  
  
  
  ### fit DR - SEP BAGS
  cat("\tFitting SEP DR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  
  # fit model with sigma
  fit_dr_sepbags = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr_sepbags$mse_test, '\n')
  results[['dr_sepbags']] = fit_dr_sepbags$y_hat
  
  
  
  ### fit DR - SEP BAGS - WEIGHTED
  cat("\tFitting SEP WDR - RBF\t")
  dist_reg_params$kernel_type = 'rbf'
  dist_reg_params$weight_col = 'kmm_weight'
  dist_reg_params$bags = bags_unm
  
  fit_wdr_sepbags = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_wdr_sepbags$mse_test, '\n')
  results[['wdr_sepbags']] = fit_wdr_sepbags$y_hat
  
  
  
  ### fit DR - SEP BAGS - LINEAR
  cat("\tFitting SEP DR - Linear\t")
  dist_reg_params$kernel_type = 'linear'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  fit_dr_sepbags_lin = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr_sepbags_lin$mse_test, '\n')
  results[['dr_sepbags_lin']] = fit_dr_sepbags_lin$y_hat
  
  
  ### fit DR - SEP BAGS - CUSTOM KERNEL
  cat("\tFitting SEP DR - Custom\t")
  dist_reg_params$kernel_type = 'rbf_age'
  dist_reg_params$weight_col = NULL
  dist_reg_params$bags = bags_unm
  
  fit_dr_sepbags_cust = tryN(doBasicDR(data = pew_data, dist_reg_params))
  #cat(fit_dr_sepbags_cust$mse_test, '\n')
  results[['dr_sepbags_cust']] = fit_dr_sepbags_cust$y_hat
  
  
  #------------------------------------------------------------------------
  ###### calculate group means - re-run after running DR because bags will be re-fit
  cat("\tGroup means\t")
  pew_data[, bag := bags$bags_newdata]
  Y_grp_means = pew_data[surveyed == 1, lapply(.SD, mean), .SDcols = dist_reg_params$outcome, by = bag]
  setnames(Y_grp_means, c('bag',gsub('_','_hat_',dist_reg_params$outcome)))
  
  Y_grp_means = merge(pew_data[, .(bag)], Y_grp_means, by = 'bag', all.x = T)
  
  Y_grp_means[is.na(y_hat_dem), y_hat_dem := 0]
  Y_grp_means[is.na(y_hat_rep), y_hat_rep := 0]
  Y_grp_means[is.na(y_hat_oth), y_hat_oth := 0]
  
  # add to results
  results[['grpmean']] = Y_grp_means[, -1]
  
  return(results)
}



