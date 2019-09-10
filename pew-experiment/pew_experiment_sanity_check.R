# stress tests

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
library(randomForest)


setwd('~/github/bdr')
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

#-----------------------------------------------
### Read in data
pew_data = fread('data/data_recoded_v2.csv')


n_holdout = 1000
n_surveyed = 2000



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




run_settings = list()
#run_settings = data.frame(match_rate = 0.1, n_bags = 125, n_landmarks = 100, refit_bags = F, party = 'onfile')
# SET PARAMS
#run_settings = sim_params[i,]

# # randomly choose settings

run_settings$party = 'insurvey'


#run_settings$party = sample(c('insurvey', 'onfile'), size = 1)
run_settings$refit_bags = sample(c(T, F), prob = c(0.4, 0.6), size = 1)
#run_settings$match_rate = runif(min = 0.01, max = 1, n = 1)
run_settings$match_rate = rbeta(1, 3, n = 1)
#run_settings$n_bags = round(runif(min = 40, max = 150, n = 1))
run_settings$n_bags = round(runif(min = 40, max = 200, n = 1))
#run_settings$n_landmarks = round(exp(rnorm(mean = 4.7, sd = 0.6, n = 1)))
#run_settings$n_landmarks = round(exp(rnorm(mean = 3.5, sd = 0.35, n = 1)))
#run_settings$n_landmarks = rnorm(mean = 300, sd = 75, n = 100)
run_settings$n_landmarks = runif(min = 3, max = 500, n = 1)
# run_settings = list(party='onfile'
#                     ,refit_bags=TRUE
#                     ,match_rate=0.319947048882023
#                     ,n_bags=94 
#                     ,n_landmarks=145 )
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
regression_vars = c(vars$file_and_survey, vars$file_only)


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



dist_reg_params = list(sigma = 'median' # use median heuristic
                       #, bags = bags
                       #, landmarks = landmarks
                       , make_bags_vars = vars$file_and_survey
                       , score_bags_vars = vars$file_and_survey
                       , regression_vars = regression_vars
                       , outcome = c('y_dem', 'y_rep', 'y_oth')
                       #, n_bags = n_bags
                       , outcome_family = 'multinomial'
                       , train_ind = 'voterfile'
                       , test_ind = 'holdout'
                       , bagging_ind = 'surveyed'
                       , kernel_type = 'rbf'
                       , weight_col = NULL)



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
cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
        , as.numeric(unlist(lasso_fit$Y_hat[pew_data$holdout == 1,]))), '\n')


### only white people
lasso_fit_white = tryN(fitLasso(mu_hat = X_lasso[which(pew_data$matched == 1 & pew_data$demo_race == 'W'), ]
                          , Y_bag = as.matrix(pew_data[matched == 1  & pew_data$demo_race == 'W', .SD, .SDcols = dist_reg_params$outcome])
                          , phi_x = X_lasso
                          , family = 'multinomial'
))
setnames(lasso_fit_white$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
            , as.numeric(unlist(lasso_fit_white$Y_hat[pew_data$holdout == 1,]))), '\n')


lasso_fit_men = tryN(fitLasso(mu_hat = X_lasso[which(pew_data$matched == 1 & pew_data$demo_sex == '02-male'), ]
                                , Y_bag = as.matrix(pew_data[matched == 1  & pew_data$demo_sex == '02-male', .SD, .SDcols = dist_reg_params$outcome])
                                , phi_x = X_lasso
                                , family = 'multinomial'
))
setnames(lasso_fit_men$Y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
            , as.numeric(unlist(lasso_fit_men$Y_hat[pew_data$holdout == 1,]))), '\n')



rf_fit = randomForest(x = X_lasso[which(pew_data$matched == 1), ]
             , y = factor(pew_data$support[pew_data$matched == 1]))
summary(rf_fit)
rf_fit$Y_hat = predict(rf_fit, X_lasso, type = 'prob')
cat(calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, dist_reg_params$outcome, with = F]))
            , as.numeric(unlist(rf_fit$Y_hat[pew_data$holdout == 1,]))), '\n')


