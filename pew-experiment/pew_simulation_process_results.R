
library(ggplot2)
library(data.table)

setwd('~/github/bdr')
pew_data = fread('data/data_recoded.csv')


mse_files = list.files('~/github/bdr/pew-experiment/results/sim_randparams/', pattern = 'mse_', full.names = T)
mses = rbindlist(lapply(mse_files, function(f) fread(f)))

mses[, length(unique(results_id))]

mses[, match_rate_bkt := floor(match_rate * 5)]

ggplot(mses, aes(x = mse, color = model)) + geom_density() + facet_grid(party~.)

mses[mse_rellogit < 1 & model != 'logit_alldata']

ggplot(mses[match_rate < 0.2 ], aes(x = mse, color = model)) + geom_density()

ggplot(mses, aes(x = match_rate, y = mse, color = model)) + 
  geom_point() +
  #geom_smooth() + 
  facet_grid(party~model)

ggplot(mses[refit_bags == F], aes(x = n_landmarks, y = mse, color = n_bags)) + 
  geom_point() +
  #geom_smooth() + 
  facet_grid(party~model)


ggplot(mses[model == 'dr_cust' & party == 'onfile']) + 
  geom_point(aes(x = n_bags, y = n_landmarks, color = mse)) +
  facet_grid(~refit_bags)

ggplot(mses[model == 'dr_cust']) + geom_contour(aes(x = n_landmarks, y = n_bags, z = mse), bins = 2)

ggplot(mses[model == 'dr_cust' & party == 'insurvey' & refit_bags == F]) + 
  geom_point(aes(x = n_landmarks, y = mse_relall, color = n_bags)) + facet_wrap(~match_rate_bkt)



pred_files = list.files('~/github/bdr/pew-experiment/results/sim_randparams', pattern = '^party', full.names = T)

holdout_error = rbindlist(lapply(pred_files, function(f){
  temp = fread(f)
  holdout_ind = which(temp[model == 'logit',]$holdout == 1)
  
  temp$act_class = rep(pew_data$support, length(unique(temp$model)))
  temp[, pred_class := c('1-Dem', '2-Rep', '3-Oth')[apply(temp[, .(y_hat_dem, y_hat_rep, y_hat_oth)], 1, which.max)]]
  temp[, correct_class := as.numeric(act_class == pred_class)]
  
  holdout_error = cbind(temp[holdout == 1, .(y_hat_dem = mean(y_hat_dem)
                                      , y_hat_rep = mean(y_hat_rep)
                                      , y_hat_oth = mean(y_hat_oth)
                                      , class_rate = mean(correct_class)
                                      ), by = .(model, results_id, match_rate, n_bags, n_landmarks, refit_bags, party)]
  , pew_data[holdout_ind, .(y_dem = mean(y_dem)
                                , y_rep = mean(y_rep)
                                , y_oth = mean(y_oth)
                                )]
        )
  holdout_error[, y_hat_dem_2way := y_hat_dem/(1 - y_hat_oth)]
  
  holdout_error[, error_dem := y_hat_dem - y_dem]
  holdout_error[, error_rep := y_hat_rep - y_rep]
  holdout_error[, error_oth := y_hat_oth - y_oth]
  holdout_error[, error_dem_2way := y_hat_dem_2way - (y_dem/(1-y_oth))]
  
  holdout_error
}))

holdout_error


ggplot(holdout_error[model %in% c('logit', 'logit_alldata', 'dr', 'dr_sepbags', 'wdr')]) + 
  geom_density(aes(x = error_dem, color= model)) + 
  
  facet_grid(~party)

ggplot(holdout_error) + 
  geom_density(aes(x = error_rep, color= model)) + 
  facet_grid(~party)

ggplot(holdout_error) + 
  geom_density(aes(x = error_oth, color= model)) + 
  facet_grid(~party)


ggplot(holdout_error) + 
  geom_density(aes(x = error_dem_2way, color= model)) + 
  facet_grid(~party)

# classification rate
ggplot(holdout_error[model %in% c('logit', 'logit_alldata', 'dr', 'dr_sepbags', 'wdr')]) +
  geom_density(aes(x = class_rate, color = model)) +
  facet_grid(~party)


results_dir = '~/github/bdr/pew-experiment/results/sim_randparams_v2/'
rmarkdown::render(input = '~/github/bdr/pew-experiment/pew_experiment.Rmd'
                    , output_dir = results_dir)
