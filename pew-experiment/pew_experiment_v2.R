library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)
library(kernlab)

setwd('~/github/bdr')

plot_dir = '~/github/bdr/pew-experiment/plots/'

source('functions.R') # contains the key functions for basic dist reg

# read in *recoded* data
pew_data = fread('data/pew_data.csv')
pew_data[y_dem == 1, support := '1-Dem']
pew_data[y_rep == 1, support := '2-Rep']
pew_data[y_oth == 1, support := '3-Other']


# mean impute age
pew_data[, age_num_imp := as.numeric(age_num)]
pew_data[is.na(age_num_imp), age_num_imp := mean(pew_data$age_num, na.rm = T)]
summary(pew_data[, age_num_imp])

#scale age
pew_data[, age_scaled := scale(age_num_imp)/5]

#categorize variables
survey_vars = c('demo_mode', 'demo_education', 'demo_phonetype', 'demo_ideology')
file_and_survey_vars = c('demo_sex', 'demo_age_bucket', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')
all_vars = names(pew_data)[grepl('demo|^age_scaled', names(pew_data))]
all_vars = all_vars[-which(grepl('demo_state|month_called', all_vars))]
file_only_vars = all_vars[!all_vars %in% c(survey_vars, file_and_survey_vars)]


## get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000, n_surveyed = 2000, n_matched = 1000
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

w_matched = getWeights(data = pew_data
           , vars = c(file_and_survey_vars, file_only_vars)
           , train_ind = 'matched'
           , target_ind = 'voterfile'
           , kernel_type = 'rbf'  #important for weighting so that marginals match
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
bags_umn = bags
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
                     )
setnames(pew_data, paste0(outcome, '_hat'), new = paste0(outcome, "_wdr"))
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
setnames(pew_data, paste0(outcome, '_hat'), new = paste0(outcome, "_dr_ln"))
fit_dr_linear$mse_test
results[['dr_linear']] = fit_dr_linear$y_hat

### fit DR -- weighted & Linear
fit_wdr_linear = doBasicDR(data = pew_data
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
                          , weight_col = 'kmm_weight'
                          , landmarks = landmarks)
setnames(pew_data, paste0(outcome, '_hat'), new = paste0(outcome, "_wdr_ln"))
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
                            )
setnames(pew_data, paste0(outcome, '_hat'), new = paste0(outcome, "_dr"))
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
setnames(pew_data, paste0(outcome, '_hat'), new = paste0(outcome, "_dr_sepbags"))
fit_dr_sepbags$mse_test
results[['dr_sepbags']] = fit_dr_sepbags$y_hat

#------------------------------------------------------------------------
###### calculate group means - re-run after running DR because bags will be re-fit
pew_data[, y_dem_grpmean := NULL]
pew_data[, y_rep_grpmean := NULL]
pew_data[, y_oth_grpmean := NULL]

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



#------------------------------------------------------------------------
#####################
# RESULT COMPARISON #
#####################

### MSE
lapply(results, function(r){
  calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, which(names(pew_data) %in% outcome), with = F]))
          , Y_pred = as.numeric(unlist(r[which(pew_data[,holdout == 1])]))
          )
})

### classification rates
lapply(results, function(r){
  mean(pew_data$support == c('1-Dem', '2-Rep', '3-Other')[apply(r, 1, which.max)])
})


###### PREP plot data
plot_data = rbindlist(lapply(names(results), function(r_lab){
  cbind(model = r_lab, results[[r_lab]], pew_data[, .(support, holdout)])
}))
setnames(plot_data, c('model', 'P(Dem)', 'P(Rep)', 'P(Other)', 'actual', 'holdout'))
plot_data
plot_data_melted = melt(plot_data, id.vars = c('model', 'actual', 'holdout'))
#cbind(plot_data_melted, actual = rep(pew_data[, support], length(unique(plot_data_melted$model)) * 3))

# add jitter for unique breaks
plot_data_melted[, value := value + rnorm(nrow(plot_data_melted), 0, 0.000000001)]
plot_data_melted = plot_data_melted[, .(actual, holdout, value, score_decile = cut(value, breaks = quantile(value, probs = seq(0,1,0.1)), labels = 1:10, include.lowest = T))
                                    , by = .(model, variable)]


###### PLOT: check distributions of predicted probabilities

plot_pred_probs = ggplot(plot_data_melted, aes(x = value, color = actual)) + geom_density() +
  facet_grid(variable ~ model) +
  scale_color_manual(values=c("dodgerblue3", "red3", "forestgreen"), 
                    name="Actual Support") +
  ggtitle("Model predicted probs by actual respondent support")+
  xlab("Predicted probability")
plot_pred_probs

ggsave(filename = paste0(plot_dir, '/plot_pred_probs.png')
       , plot = plot_pred_probs, device = 'png', width = 10, height = 6)

##### PLOT: check rank-order

plot_data_melted[, actual_ind := as.numeric(gsub('.-', '', actual) == gsub('P[(]|[)]', '', variable))]

plot_deciles = ggplot(plot_data_melted[holdout == 1, .(avg_class = mean(actual_ind)),.(score_decile, model, variable)], aes(x = score_decile, y = avg_class)) + geom_bar(stat = 'identity') +
  facet_grid(variable ~ model) +
  ggtitle("Actual class rate by decile of score - HOLDOUT group")+
  xlab("Score decile")
plot_deciles
ggsave(filename = paste0(plot_dir, '/plot_deciles.png')
       , plot = plot_deciles, device = 'png', width = 10, height = 6)

#------------------------------------------------------------------------
##### XTABS

getXtab = function(var, data){
  tab = data[, .(.N
                     , prop_surveyed = mean(surveyed)
                     , prop_matched = mean(matched)
                     , n_holdout = sum(holdout)
                     , dist_raw = .N/nrow(data)
                     , dist_matched = sum(matched)/sum(data$matched)
                 
                 , surveyed_y_dem = sum(y_dem * surveyed)/sum(surveyed)
                 , matched_y_dem = sum(y_dem * matched)/sum(matched)
                 , voterfile_y_dem = sum(y_dem * voterfile * (matched == 0))/sum(voterfile * (matched == 0))
                     
                     , test_y_dem = sum(y_dem * holdout)/sum(holdout)
                     , test_y_dem_dr = sum(y_dem_hat * holdout)/sum(holdout)
                     , test_y_dem_logit = sum(y_dem_logit * holdout)/sum(holdout)
                     , test_y_dem_grpmean = sum(y_dem_grpmean * holdout)/sum(holdout)
                     
                     , test_y_rep = sum(y_rep * holdout)/sum(holdout)
                     , test_y_rep_dr = sum(y_rep_hat * holdout)/sum(holdout)
                     , test_y_rep_logit = sum(y_rep_logit * holdout)/sum(holdout)
                     , test_y_rep_grpmean = sum(y_rep_grpmean * holdout)/sum(holdout)
                     
                     , test_y_oth = sum(y_oth * holdout)/sum(holdout)
                     , test_y_oth_dr = sum(y_oth_hat * holdout)/sum(holdout)
                     , test_y_oth_logit = sum(y_oth_logit * holdout)/sum(holdout)
                     , test_y_oth_grpmean = sum(y_oth_grpmean * holdout)/sum(holdout)
                 
                 , test_y_dem_2way = sum(y_dem * holdout)/sum(holdout * as.numeric(y_dem + y_rep > 0))
                 , test_y_dem_2way_dr = sum(y_dem_hat * holdout)/sum(holdout * as.numeric(y_dem + y_rep > 0))
                 , test_y_dem_2way_logit = sum(y_dem_logit * holdout)/sum(holdout * as.numeric(y_dem + y_rep > 0))
                 , test_y_dem_2way_grpmean = sum(y_dem_grpmean * holdout)/sum(holdout * as.numeric(y_dem + y_rep > 0))
                 
                 , test_class_dr = sum((support == class_dr) * holdout)/sum(holdout)
                 , test_class_logit = sum((support == class_logit) * holdout)/sum(holdout)
                 , test_class_grpmean = sum((support == class_grpmean) * holdout)/sum(holdout)
                     
  ), by = get(var)]
  
  tab[, error_dem_dr := test_y_dem_dr - test_y_dem]
  tab[, error_dem_logit := test_y_dem_logit - test_y_dem]
  tab[, error_dem_grpmean := test_y_dem_grpmean - test_y_dem]
  
  tab[, error_rep_dr := test_y_rep_dr - test_y_rep]
  tab[, error_rep_logit := test_y_rep_logit - test_y_rep]
  tab[, error_rep_grpmean := test_y_rep_grpmean - test_y_rep]
  
  tab[, error_oth_dr := test_y_oth_dr - test_y_oth]
  tab[, error_oth_logit := test_y_oth_logit - test_y_oth]
  tab[, error_oth_grpmean := test_y_oth_grpmean - test_y_oth]
  
  tab[, error_dem_2way_dr := test_y_dem_2way_dr - test_y_dem_2way]
  tab[, error_dem_2way_logit := test_y_dem_2way_logit - test_y_dem_2way]
  tab[, error_dem_2way_grpmean := test_y_dem_2way_grpmean - test_y_dem_2way]
  
  tab = cbind(var, tab)
  return(tab)
}

all_tabs = rbindlist(lapply(c(' topline', file_and_survey_vars[-which(file_and_survey_vars == 'demo_state')]), getXtab, data = cbind(' topline' = 1, pew_data)))

all_tabs[, label_ordered := factor(get, levels = all_tabs[order(test_y_dem)]$get, ordered = T)]
all_tabs[order(label_ordered)]

ggplot(all_tabs) +
  #geom_point(aes(x = error_dem, y = label_ordered, color = 'actual')) + 
  geom_point(aes(x = error_dem_dr_wt, y = label_ordered, color = 'DRW')) + 
  geom_point(aes(x = error_dem_dr, y = label_ordered, color = 'DR')) + 
  geom_point(aes(x = error_dem_logit, y = label_ordered, color = 'logit')) + 
  geom_point(aes(x = error_dem_grpmean, y = label_ordered, color = 'group mean'))


