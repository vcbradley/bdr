rm(list = ls())
library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)
library(kernlab)
library(R.utils)
library(glmnet)

setwd('~/github/bdr')
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

# read in *recoded* data
pew_data = fread('data/data_recoded.csv')

run_settings = list(party = 'onfile'   #'onfile' or 'insurvey'
                    , n_surveyed = 2000
                    , match_rate = 0.5
                    , n_bags = 75
                    , n_landmarks = 200
                    , refit_bags = F)

results_id = paste0('party',run_settings$party
                    , '_match', round(run_settings$match_rate*100)
                    , '_bags', run_settings$n_bags
                    , '_lmks', run_settings$n_landmarks
                    , '_refitbags', run_settings$refit_bags)


#categorize variables
vars = list()
vars$all = names(pew_data)[grepl('demo|^age_scaled', names(pew_data))]
vars$survey = c('demo_mode', 'demo_education', 'demo_phonetype', 'demo_ideology')#, 'demo_party')
# add party to survey variables
if(run_settings$party == 'insurvey'){
  vars$survey = c(vars$survey, 'demo_party')
}
vars$file_and_survey = c('demo_sex', 'demo_age_bucket', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')
vars$all = vars$all[-which(grepl('demo_state|month_called', vars$all))]
vars$file_only = vars$all[!vars$all %in% c(vars$survey, vars$file_and_survey)]





## get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000
                         , n_surveyed = run_settings$n_surveyed
                         , n_matched = run_settings$n_surveyed * run_settings$match_rate
                         , p_surveyed = pew_data$p_surveyed
                         , p_matched = pew_data$p_matched
)
pew_data[, unmatched := as.numeric(surveyed == 1 & matched == 0)]
pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]



#------------------------------------------------------------------------
################################
######### WEIGHT DATA ##########
################################

# marginals won't match with RBF kernel, but weights are too extreme with linear

#B = 5.0  # upper bound; B = 1 is the unweighted solution

pew_data[, kmm_weight := 1]

# weight non-age file vars
w_matched = getWeights(data = pew_data
                       , vars = c(vars$file_and_survey, vars$file_only)
                       , train_ind = 'matched'
                       , target_ind = 'voterfile'
                       , kernel_type = 'rbf_age' 
                       , sigma = 0.1
                       , B = c(0.1, 5))
hist(w_matched)
summary(w_matched)

# set weights
pew_data[matched == 1, kmm_weight := w_matched]

w_unmatched = getWeights(data = pew_data
                         , vars = c(vars$file_and_survey, vars$survey)
                         , train_ind = 'unmatched'
                         , target_ind = 'matched'
                         , weight_col = 'kmm_weight'
                         , kernel_type = 'linear' #important for weighting so that marginals match
                         , B = c(0.01, 7))
hist(w_unmatched)
summary(w_unmatched)

pew_data[unmatched == 1, kmm_weight := w_unmatched]

hist(pew_data$kmm_weight)
summary(pew_data$kmm_weight)

# check that distributions match ok
rbindlist(lapply(c(vars$file_and_survey), function(v){
  cbind(v, pew_data[, .(dist_vf = sum(voterfile * kmm_weight)/sum(pew_data$voterfile)
                        , dist_matched = sum(matched * kmm_weight)/sum(pew_data$matched)
                        , dist_unmatched = sum(unmatched * kmm_weight)/sum(pew_data$unmatched)
                        , dist_both = sum((matched + unmatched) * kmm_weight)/sum(pew_data$unmatched + pew_data$matched)
  ), get(v)])
}))[order(v, get)]





#------------------------------------------------------------------------
###################################
############ FIT MODELS ###########
###################################
results = list()

regression_vars = c(vars$file_and_survey, vars$file_only)
n_landmarks = run_settings$n_landmarks
n_bags = run_settings$n_bags

## fix landmarks
landmarks = getLandmarks(data = pew_data
                         , vars = regression_vars
                         , n_landmarks = n_landmarks
                         , subset_ind = (pew_data[, voterfile] == 1))

## fix bags
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



#### SET PARAMETERS ####
dist_reg_params = list(sigma = 0.16 #from quick CV
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


## Basic LASSO
lasso_frmla = as.formula(paste0("~", paste(c(vars$file_and_survey, vars$file_only), collapse = '+')))
X_lasso = modmat_all_levs(pew_data, formula = lasso_frmla)

lasso_fit = fitLasso(mu_hat = X_lasso[which(pew_data$matched == 1), ]
                     , Y_bag = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                     , phi_x = X_lasso
                     , family = 'multinomial'
)
setnames(lasso_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))

# lasso_fit = cv.glmnet(x = X_lasso[which(pew_data$matched == 1), ]
#                    , y = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
#                    , family = 'multinomial')
# lasso_fit$Y_hat = data.table(data.frame(predict(lasso_fit, newx = X_lasso, type = 'response')))
 setnames(lasso_fit$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))

calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
        , as.numeric(unlist(lasso_fit$Y_hat[pew_data$holdout == 1,])))

results[['logit']] = lasso_fit$Y_hat

# ggplot(lasso_fit$Y_hat) + geom_density(aes(x = y_dem.1), color = 'blue') +
#   geom_density(aes(x = y_rep.1), color = 'red') +
#   geom_density(aes(x = y_oth.1), color = 'green') +
#   facet_grid(~pew_data$support)
# 
# nonzero_ind = sort(unique(unlist(lapply(coef(lasso_fit, s = 'lambda.min'), function(c){
#   which(c[-c(1,2)] != 0) # because of two intercepts
# })))) + 1
# nonzero_ind = c(1, nonzero_ind) # always include one intercept
# 
# lasso_fit2 = cv.glmnet(x = X_lasso[which(pew_data$matched == 1), nonzero_ind]
#                       , y = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
#                       , family = 'multinomial'
#                     , lambda = 0)
# lasso_fit2$Y_hat = data.table(data.frame(predict(lasso_fit2, newx = X_lasso[, nonzero_ind], type = 'response')))
# setnames(lasso_fit2$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
# 
# calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
#         , as.numeric(unlist(lasso_fit2$Y_hat[pew_data$holdout == 1,])))

#------------------------------------------------------------------------
### DIST REG MODELS


# # quick CV
# sigmas = exp(seq(log(0.000001), log(0.01), length.out = 10))
# #sigmas = seq(0.01, 0.2, length.out = 20)
# dist_reg_params$kernel_type = 'rbf'
# dist_reg_params$bags = bags_unm
# 
# cust_mse = unlist(lapply(sigmas, function(s){
#   dist_reg_params$sigma = s
#   try(doBasicDR(data = pew_data, dist_reg_params)$mse_test)
# }))
# 
# plot(x = sigmas[1:length(cust_mse)], y = cust_mse)
# sigmas[which.min(cust_mse)]


### fit DR -- Linear
dist_reg_params$kernel_type = 'linear'
dist_reg_params$weight_col = NULL
dist_reg_params$bags = bags

fit_dr_linear = doBasicDR(data = pew_data, dist_reg_params)
fit_dr_linear$mse_test
results[['dr_linear']] = fit_dr_linear$y_hat



### fit DR -- weighted & Linear
dist_reg_params$kernel_type = 'linear'
dist_reg_params$weight_col = 'kmm_weight'
dist_reg_params$bags = bags

fit_wdr_linear = doBasicDR(data = pew_data, dist_reg_params)
fit_wdr_linear$mse_test
results[['wdr_linear']] = fit_wdr_linear$y_hat



### fit DR - NO WEIGHTING
dist_reg_params$kernel_type = 'rbf'
dist_reg_params$weight_col = NULL
dist_reg_params$bags = bags
dist_reg_params$sigma = 0.16

fit_dr = doBasicDR(data = pew_data, dist_reg_params)
fit_dr$mse_test
results[['dr']] = fit_dr$y_hat



### fit DR -- WEIGHTED
dist_reg_params$kernel_type = 'rbf'
dist_reg_params$weight_col = 'kmm_weight'
dist_reg_params$bags = bags
dist_reg_params$sigma = 0.16

fit_wdr = doBasicDR(data = pew_data, dist_reg_params)
fit_wdr$mse_test
results[['wdr']] = fit_wdr$y_hat



### fit DR - SEP BAGS
dist_reg_params$kernel_type = 'rbf'
dist_reg_params$weight_col = NULL
dist_reg_params$bags = bags_unm
dist_reg_params$sigma = 0.003

fit_dr_sepbags = doBasicDR(data = pew_data, dist_reg_params)
fit_dr_sepbags$mse_test
results[['dr_sepbags']] = fit_dr_sepbags$y_hat



### fit DR - SEP BAGS - WEIGHTED
dist_reg_params$kernel_type = 'rbf'
dist_reg_params$weight_col = 'kmm_weight'
dist_reg_params$bags = bags_unm
dist_reg_params$sigma = 0.003

fit_wdr_sepbags = doBasicDR(data = pew_data, dist_reg_params)
fit_wdr_sepbags$mse_test
results[['wdr_sepbags']] = fit_wdr_sepbags$y_hat



### fit DR - SEP BAGS - LINEAR
dist_reg_params$kernel_type = 'linear'
dist_reg_params$weight_col = NULL
dist_reg_params$bags = bags_unm

fit_dr_sepbags_lin = doBasicDR(data = pew_data, dist_reg_params)
fit_dr_sepbags_lin$mse_test
results[['dr_sepbags_lin']] = fit_dr_sepbags_lin$y_hat


### fit DR - SEP BAGS - CUSTOM KERNEL
dist_reg_params$kernel_type = 'rbf_age'
dist_reg_params$sigma = 2.6
dist_reg_params$weight_col = NULL
dist_reg_params$bags = bags_unm

fit_dr_sepbags_cust = doBasicDR(data = pew_data, dist_reg_params)
fit_dr_sepbags_cust$mse_test
results[['dr_sepbags_cust']] = fit_dr_sepbags_cust$y_hat

# # quick CV
# 
# #sigmas = exp(seq(log(0.005), log(5), length.out = 20))
# sigmas = seq(2,3, length.out = 10)
# 
# cust_mse = unlist(lapply(sigmas, function(s){
#   dist_reg_params$sigma = s
#   try(doBasicDR(data = pew_data, dist_reg_params)$mse_test)
# }))
# 
# plot(x = sigmas, y = cust_mse)


#------------------------------------------------------------------------
###### calculate group means - re-run after running DR because bags will be re-fit

pew_data[, bag := bags$bags_newdata]
Y_grp_means = pew_data[surveyed == 1, lapply(.SD, mean), .SDcols = dist_reg_params$outcome, by = bag]
setnames(Y_grp_means, c('bag',gsub('_','_hat_',dist_reg_params$outcome)))

Y_grp_means = merge(pew_data[, .(bag)], Y_grp_means, by = 'bag', all.x = T)

Y_grp_means[is.na(y_hat_dem), y_hat_dem := 0]
Y_grp_means[is.na(y_hat_rep), y_hat_rep := 0]
Y_grp_means[is.na(y_hat_oth), y_hat_oth := 0]

# add to results
results[['grpmean']] = Y_grp_means[, -1]


#------------------------------------------------------------------------
## Write out results
save(results, vars, pew_data, run_settings
     , file = paste0('~/github/bdr/pew-experiment/results/results_',results_id,'.RData'))


rmarkdown::render(input = '~/github/bdr/pew-experiment/results/results_analysis.Rmd',
                  output_file = paste0('results_analysis_', results_id, '.html')
                  )
