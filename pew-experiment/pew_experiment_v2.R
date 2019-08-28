rm(list = ls())
library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)
library(kernlab)
library(plotROC)
library(R.utils)

setwd('~/github/bdr')
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

plot_dir = '~/github/bdr/pew-experiment/plots/'

# read in *recoded* data
pew_data = fread('data/data_recoded.csv')


#categorize variables
vars = list()
vars$all = names(pew_data)[grepl('demo|^age_scaled', names(pew_data))]
vars$survey = c('demo_mode', 'demo_education', 'demo_phonetype', 'demo_ideology')#, 'demo_party')
vars$file_and_survey = c('demo_sex', 'demo_age_bucket', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')
vars$all = vars$all[-which(grepl('demo_state|month_called', vars$all))]
vars$file_only = vars$all[!vars$all %in% c(vars$survey, vars$file_and_survey)]


## get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000
                         , n_surveyed = 2000
                         , n_matched = 1000
                         , p_surveyed = pew_data$p_surveyed
                         , p_matched = pew_data$p_matched
)
pew_data[, unmatched := as.numeric(surveyed == 1 & matched == 0)]
pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]



#------------------------------------------------------------------------
################################
######### WEIGHT DATA ##########
################################

#B = 5.0  # upper bound; B = 1 is the unweighted solution
#sigma = 0.25

pew_data[, kmm_weight := 1]

# weight JUST for age
w_age = getWeights(data = pew_data
                   , vars = 'age_scaled'
                   , train_ind = 'matched'
                   , target_ind = 'voterfile'
                   , kernel_type = 'rbf'  # marginals won't match with RBF kernel, but weights are too extreme with linear
                   , sigma = 0.2  #setting based on median distance
                   , B = 3)

pew_data[matched == 1, kmm_weight := w_age]

ggplot(pew_data[matched == 1, .(mean_wt = mean(kmm_weight)), age_scaled], aes(x = age_scaled, y = mean_wt)) + geom_point()

# weight non-age file vars
w_matched = getWeights(data = pew_data
                       , vars = c(vars$file_and_survey, vars$file_only)
                       , train_ind = 'matched'
                       , target_ind = 'voterfile'
                       , weight_col = 'kmm_weight'
                       , kernel_type = 'linear'  # marginals won't match with RBF kernel, but weights are too extreme with linear
                       , sigma = 0.2
                       , B = 3)
hist(w_matched)
summary(w_matched)

pew_data[matched == 1, kmm_weight := w_matched]

w_unmatched = getWeights(data = pew_data
                         , vars = c(file_and_survey_vars, survey_vars)
                         , train_ind = 'unmatched'
                         , target_ind = 'matched'
                         , weight_col = 'kmm_weight'
                         , kernel_type = 'linear' #important for weighting so that marginals match
                         , B = 5)
hist(w_unmatched)
summary(w_unmatched)

pew_data[unmatched == 1, kmm_weight := w_unmatched]

hist(pew_data$kmm_weight)
summary(pew_data$kmm_weight)

# check that distributions match ok
rbindlist(lapply(c(file_and_survey_vars), function(v){
  cbind(v, pew_data[, .(dist_vf = sum(voterfile * kmm_weight)/sum(pew_data$voterfile)
                        , dist_matched = sum(matched * kmm_weight)/sum(pew_data$matched)
                        , dist_unmatched = sum(unmatched * kmm_weight)/sum(pew_data$unmatched)
  ), get(v)])
}))[order(v, get)]


#------------------------------------------------------------------------
################################
######### WEIGHT DATA - LINEAR ##########
################################

#B = 5.0  # upper bound; B = 1 is the unweighted solution
#sigma = 0.25

pew_data[, lin_weight := 1]

w_matched = getWeights(data = pew_data
                       , vars = c(file_and_survey_vars, file_only_vars)
                       , train_ind = 'matched'
                       , target_ind = 'voterfile'
                       , kernel_type = 'rbf'  # marginals won't match with RBF kernel, but weights are too extreme with linear
                       , sigma = 0.3
                       , B = 3)
hist(w_matched)
summary(w_matched)
pew_data[matched == 1, lin_weight := w_matched]

ggplot(pew_data[matched == 1, .(mean_wt = mean(lin_weight)), age][order(age)], aes(x = age, y = mean_wt)) + geom_point()



w_unmatched = getWeights(data = pew_data
                         , vars = c(file_and_survey_vars, survey_vars)
                         , train_ind = 'unmatched'
                         , target_ind = 'matched'
                         , weight_col = 'lin_weight'
                         , kernel_type = 'linear' #important for weighting so that marginals match
                         , B = 5)
hist(w_unmatched)
summary(w_unmatched)

pew_data[unmatched == 1, lin_weight := w_unmatched]

hist(pew_data$lin_weight)
summary(pew_data$lin_weight)

# check that distributions match ok
rbindlist(lapply(c(file_and_survey_vars), function(v){
  cbind(v, pew_data[, .(dist_vf = sum(voterfile * lin_weight)/sum(pew_data$voterfile)
                        , dist_matched = sum(matched * lin_weight)/sum(pew_data$matched)
                        , dist_unmatched = sum(unmatched * lin_weight)/sum(pew_data$unmatched)
  ), get(v)])
}))[order(v, get)]


#------------------------------------------------------------------------
###################################
############ FIT MODELS ###########
###################################

regression_vars = c(file_and_survey_vars, file_only_vars)
n_landmarks = 200
n_bags = 75
results = list()

## fix landmarks
landmarks = getLandmarks(data = pew_data
                         , vars = regression_vars
                         , n_landmarks = n_landmarks
                         , subset_ind = (pew_data[, voterfile] == 1))

## fix bags
bags = getBags(data = pew_data[surveyed == 1,]
               , vars = file_and_survey_vars
               , n_bags = n_bags
               , newdata = pew_data[, file_and_survey_vars, with = F])

# give each matched data point its own bag
# bags_unm = getBags(data = pew_data[unmatched == 1,]
#                , vars = file_and_survey_vars
#                , n_bags = n_bags
#                , newdata = pew_data[, file_and_survey_vars, with = F])
bags_unm = bags
bags_unm$bags_newdata[pew_data$matched == 1] <- seq(n_bags + 1, length = sum(pew_data$matched))


### fit DR -- WEIGHTED
outcome = c('y_dem', 'y_rep', 'y_oth')
fit_wdr = doBasicDR(data = pew_data
                    , make_bags_vars = file_and_survey_vars
                    , score_bags_vars = file_and_survey_vars
                    , regression_vars = regression_vars
                    , outcome = outcome
                    , n_bags = n_bags
                    #, n_landmarks = 300
                    , sigma = 0.01
                    , family = 'multinomial'
                    , bagging_ind = 'surveyed'
                    , train_ind = 'voterfile'
                    , test_ind = 'holdout'
                    , weight_col = 'kmm_weight'
                    , landmarks = landmarks
                    , bags = bags
)
fit_wdr$mse_test
results[['wdr']] = fit_wdr$y_hat

### fit DR -- Linear
fit_dr_linear = doBasicDR(data = pew_data
                          , make_bags_vars = file_and_survey_vars
                          , score_bags_vars = file_and_survey_vars
                          , regression_vars = regression_vars
                          , outcome = outcome
                          , n_bags = n_bags
                          #, n_landmarks = 300
                          #, sigma = 0.01
                          , family = 'multinomial'
                          , bagging_ind = 'surveyed'
                          , train_ind = 'voterfile'
                          , test_ind = 'holdout'
                          , kernel_type = 'linear'
                          , landmarks = landmarks)
fit_dr_linear$mse_test
results[['dr_linear']] = fit_dr_linear$y_hat

### fit DR -- weighted & Linear
fit_wdr_linear = doBasicDR(data = pew_data
                           , make_bags_vars = file_and_survey_vars
                           , score_bags_vars = file_and_survey_vars
                           , regression_vars = regression_vars
                           , outcome = outcome
                           #, n_bags = n_bags
                           #, n_landmarks = 300
                           #, sigma = 0.01
                           , family = 'multinomial'
                           , bagging_ind = 'surveyed'
                           , train_ind = 'voterfile'
                           , test_ind = 'holdout'
                           , kernel_type = 'linear'
                           , weight_col = 'kmm_weight'
                           , landmarks = landmarks
                           , bags = bags)
fit_wdr_linear$mse_test
results[['wdr_linear']] = fit_wdr_linear$y_hat


### fit DR - NO WEIGHTING
fit_dr = doBasicDR(data = pew_data
                   , make_bags_vars = file_and_survey_vars
                   , score_bags_vars = file_and_survey_vars
                   , regression_vars = regression_vars
                   , outcome = outcome
                   , n_bags = n_bags
                   #, n_landmarks = 300
                   , sigma = 0.01
                   , family = 'multinomial'
                   , bagging_ind = 'surveyed'
                   , train_ind = 'voterfile'
                   , test_ind = 'holdout'
                   , landmarks = landmarks
                   , bags = bags
)
fit_dr$mse_test
results[['dr']] = fit_dr$y_hat


### fit DR - matched own bags
fit_dr_sepbags = doBasicDR(data = pew_data
                           , make_bags_vars = file_and_survey_vars
                           , score_bags_vars = file_and_survey_vars
                           , regression_vars = regression_vars
                           , outcome = outcome
                           , n_bags = n_bags
                           #, n_landmarks = 300
                           , sigma = 0.01
                           , family = 'multinomial'
                           , bagging_ind = 'surveyed'
                           , train_ind = 'voterfile'
                           , test_ind = 'holdout'
                           , landmarks = landmarks
                           , bags = bags_unm
)
fit_dr_sepbags$mse_test
results[['dr_sepbags']] = fit_dr_sepbags$y_hat

#------------------------------------------------------------------------
###### calculate group means - re-run after running DR because bags will be re-fit

Y_grp_means = fit_wdr$data[surveyed == 1, lapply(.SD, mean), .SDcols = outcome, by = bag]
setnames(Y_grp_means, c('bag',paste0(outcome, '_grpmean')))

Y_grp_means = merge(pew_data[, .(bag)], Y_grp_means, by = 'bag', all.x = T)

Y_grp_means[is.na(y_dem_grpmean), y_dem_grpmean := 0]
Y_grp_means[is.na(y_rep_grpmean), y_rep_grpmean := 0]
Y_grp_means[is.na(y_oth_grpmean), y_oth_grpmean := 0]

# add to results
results[['grpmean']] = Y_grp_means[, -1]


#------------------------------------------------------------------------
## Basic LASSO
lasso_frmla = as.formula(paste0("~", paste(c(file_and_survey_vars, file_only_vars), collapse = '+')))
X_lasso = modmat_all_levs(pew_data, formula = lasso_frmla)

lasso_fit = fitLasso(mu_hat = X_lasso[which(pew_data$matched == 1), ]
                     , Y_bag = as.matrix(pew_data[matched == 1, .(y_dem, y_rep, y_oth)])
                     , phi_x = X_lasso
                     , family = 'multinomial'
)

pew_data[, c('y_dem_logit', 'y_rep_logit', 'y_oth_logit') := 
           as.list(data.table(matrix(lasso_fit$Y_hat, ncol= 3)))]

results[['logit']] = pew_data[, c('y_dem_logit', 'y_rep_logit', 'y_oth_logit'), with = F]


