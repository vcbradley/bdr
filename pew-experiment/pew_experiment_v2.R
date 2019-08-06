library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)
library(tictoc) # for benchmarking
library(kernlab)

setwd('~/github/bdr')

source('functions.R') # contains the key functions for basic dist reg

# read in *recoded* data
pew_data = fread('data/pew_data.csv')
pew_data[y_dem == 1, support := '1-Dem']
pew_data[y_rep == 1, support := '2-Rep']
pew_data[y_oth == 1, support := '3-Other']

#categorize variables
survey_vars = c('demo_mode', 'demo_education', 'demo_phonetype', 'month_called', 'demo_ideology')
file_and_survey_vars = c('demo_sex', 'demo_age_bucket', 'demo_state', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')
all_vars = names(pew_data)[grepl('demo', names(pew_data))]
file_only_vars = all_vars[!all_vars %in% c(survey_vars, file_and_survey_vars)]

# set n_bags
n_bags = 30

## get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000, n_surveyed = 2000, n_matched = 1000
                         , p_surveyed = pew_data$p_surveyed
                         , p_matched = pew_data$p_matched
)
pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]



    
### fit DR
dr_fit = doBasicDR(data = pew_data
                     , bagging_vars = c(file_and_survey_vars, survey_vars)
                     , regression_vars = c(file_and_survey_vars, file_only_vars)
                     , outcome = c('y_dem', 'y_rep', 'y_oth')
                     , n_bags = 75
                     , n_landmarks = 500
                     , sigma = 0.0005
                     , family = 'multinomial'
                     , bagging_ind = 'surveyed'
                     , train_ind = 'voterfile'
                     , test_ind = 'holdout')
dr_fit$mse_test

### calculate group means - re-run after running DR because bags will be re-fit
pew_data[, y_dem_grpmean := NULL]
pew_data[, y_rep_grpmean := NULL]
pew_data[, y_oth_grpmean := NULL]
Y_grp_means = dr_fit$data[surveyed == 1, lapply(.SD, mean), .SDcols = outcome, by = bag]
setnames(Y_grp_means, c('bag',paste0(outcome, '_grpmean')))
pew_data = merge(pew_data, Y_grp_means, by = 'bag', all.x = T)

calcMSE(Y = as.numeric(unlist(pew_data[holdout == 1, which(names(pew_data) %in% outcome), with = F]))
                   , Y_pred = as.numeric(unlist(pew_data[holdout == 1, which(names(pew_data) %in% paste0(outcome, '_grpmean')), with = F])))


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


##### COMPARE #####

pew_data$class_dr = c('1-Dem', '2-Rep', '3-Other')[apply(pew_data[, paste0(outcome, '_hat'), with = F], 1, which.max)]
pew_data$class_grpmean = c('1-Dem', '2-Rep', '3-Other')[apply(pew_data[, paste0(outcome, '_grpmean'), with = F], 1, which.max)]
pew_data$class_logit = c('1-Dem', '2-Rep', '3-Other')[apply(pew_data[, paste0(outcome, '_logit'), with = F], 1, which.max)]


# check distributions
plotlist = lapply(c(paste0(outcome, '_hat')
                    , paste0(outcome, '_grpmean')
                    , paste0(outcome, '_logit')
                    ), function(o){
  ggplot(pew_data, aes(x = get(o), color = support)) + geom_density()
})
marrangeGrob(plotlist, nrow = 3, ncol = 3)


# check rank-order
grid.arrange(
doDecilePlot(data = pew_data, score_name = 'y_dem_hat', title = 'Dist Reg')
, doDecilePlot(data = pew_data, score_name = 'y_dem_grpmean', title = 'Group Means')
, doDecilePlot(data = pew_data, score_name = 'y_dem_logit', title = 'Logit')
, ncol = 3)


# check sub-groups
v = 'demo_sex'


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
  geom_point(aes(x = error_dem_dr, y = label_ordered, color = 'DR')) + 
  geom_point(aes(x = error_dem_logit, y = label_ordered, color = 'logit')) + 
  geom_point(aes(x = error_dem_grpmean, y = label_ordered, color = 'group mean'))


