# ecological features


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

results_dir = '~/github/bdr/pew-experiment/results/sim_randparams_v2/'

if(!dir.exists(results_dir)){
  dir.create(results_dir)
}

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
run_settings$party = sample(c('insurvey', 'onfile'), size = 1)
#run_settings$refit_bags = sample(c(T, F), prob = c(0.4, 0.6), size = 1)
run_settings$refit_bags = sample(c(T, F), size = 1)
#run_settings$match_rate = runif(min = 0.01, max = 1, n = 1)
run_settings$match_rate = rbeta(1, 2, n = 1)
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



#-----------------------------------------
## fix landmarks
cat("\tGetting landmarks\n")
n_landmarks = run_settings$n_landmarks

landmarks = getLandmarks(data = pew_data
                         , vars = regression_vars
                         , n_landmarks = n_landmarks
                         , subset_ind = (pew_data[, voterfile] == 1))



#-----------------------------------------
#### SET PARAMETERS ####
dist_reg_params = list(sigma = 'median' # use median heuristic
                       #, bags = bags  #moveing to withiin the runModel code
                       , refit_bags = run_settings$refit_bags
                       , landmarks = landmarks
                       , make_bags_vars = vars$file_and_survey
                       , score_bags_vars = vars$file_and_survey
                       , regression_vars = regression_vars
                       , outcome = c('y_dem', 'y_rep', 'y_oth')
                       , n_bags = run_settings$n_bags
                       , outcome_family = 'multinomial'
                       , train_ind = 'voterfile'
                       , test_ind = 'holdout'
                       , bagging_ind = 'surveyed'
                       , kernel_type = 'rbf'
                       , weight_col = NULL)



bags = getBags(data = pew_data[surveyed == 1,]
               , vars = vars$file_and_survey
               , n_bags = dist_reg_params$n_bags
               , newdata = pew_data[, vars$file_and_survey, with = F])

# if we're refitting bags, create new bag just using the unmatched survey data
if(dist_reg_params$refit_bags){
  bags_unm = getBags(data = pew_data[unmatched == 1,]
                     , vars = vars$file_and_survey
                     , n_bags = dist_reg_params$n_bags
                     , newdata = pew_data[, vars$file_and_survey, with = F])
}else{
  bags_unm = bags
}
# give each matched data point its own bag
bags_unm$bags_newdata[pew_data$matched == 1] <- seq(dist_reg_params$n_bags + 1, length = sum(pew_data$matched))

pew_data[, bag := bags$bags_newdata]
pew_data[, weight := 1]

kernel_params = getKernParams(X = landmarks$X, kernel_type = dist_reg_params$kernel_type, sigma = dist_reg_params$sigma)

features = getFeatures(data = landmarks$X
                       , bag = as.numeric(pew_data[, bag])
                       , train_ind = as.numeric(pew_data[, get(dist_reg_params$train_ind)])
                       , landmarks = landmarks$landmarks
                       , kernel_params = kernel_params
                       , weight = as.numeric(pew_data[, weight])
                       )



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


# calculate mse
calcMSE(Y = as.numeric(unlist(pew_data[get(dist_reg_params$test_ind) == 1, which(names(pew_data) %in% dist_reg_params$outcome), with = F]))
                   , Y_pred = as.numeric(unlist(lasso_fit$Y_hat[which(pew_data[,get(dist_reg_params$test_ind) == 1])])))



cat("\tFitting LOGIT\t")
X_ecol = merge(data.table('bag' = features$phi_x$bag), features$mu_hat, by = 'bag', sort = F, all.x = T)
X_ecol_mat = as.matrix(X_ecol[, 2:ncol(X_ecol), with = F])

X_lasso_and_ecol = cbind(X_lasso, X_ecol_mat)
lasso_ecol_fit = tryN(fitLasso(mu_hat = X_lasso_and_ecol[which(pew_data$matched == 1), ]
                          , Y_bag = as.matrix(pew_data[matched == 1, .SD, .SDcols = dist_reg_params$outcome])
                          , phi_x = X_lasso_and_ecol
                          , family = 'multinomial'
))

setnames(lasso_ecol_fit$Y, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
#data[, paste0(outcome, '_hat') := as.list(data.frame(fit$Y))]


coef(lasso_fit$fit, s = 'lambda.min')
coef(lasso_ecol_fit$fit, s = 'lambda.min')

# calculate mse
calcMSE(Y = as.numeric(unlist(pew_data[get(dist_reg_params$test_ind) == 1, which(names(pew_data) %in% dist_reg_params$outcome), with = F]))
                   , Y_pred = as.numeric(unlist(lasso_ecol_fit$Y_hat[which(pew_data[,get(dist_reg_params$test_ind) == 1])])))


