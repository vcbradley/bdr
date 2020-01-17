library(ggplot2)
library(data.table)
library(knitr)
library(R.utils)

#results_dir = '~/github/bdr/pew-experiment/results/sim_randparams_v2/'
results_dir = '~/github/bdr/ecological features/sim_results/v1/'

# set the working directory
setwd(results_dir)

if(!dir.exists('plots')){
  dir.create(paste0(results_dir, 'plots'))
}

# source functions
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

# load data
pew_data = fread('~/github/bdr/data/data_recoded_v2.csv')

# load MSE results
mse_files = list.files(results_dir, pattern = 'mse_', full.names = T)
mses = rbindlist(lapply(mse_files, function(f) fread(f)))
mses[, match_rate_bkt := floor(match_rate * 5)]

# DROP OUTLIER
mses = mses[mse < 0.3]

#rename logit
mses[, model := gsub('logit', 'lasso', model)]


#------------------------------------
# table of model descriptions
model_desc_tab = data.frame(model_name = unique(mses$model))
model_desc_tab$kernel = ifelse(grepl('cust',model_desc_tab$model_name), "custom"
                               , ifelse(grepl('linear',model_desc_tab$model_name), "linear"
                                        , ifelse(grepl('dr',model_desc_tab$model_name), "rbf", "")))
model_desc_tab$weighted = ifelse(grepl('wdr',model_desc_tab$model_name), "X", "")
model_desc_tab$separate_bags = ifelse(grepl('sepbags',model_desc_tab$model_name), "X", "")

kable(model_desc_tab)


#------------------------------------
# CALCULATE BIAS
if(file.exists(paste0(results_dir, 'bias_summary.csv'))){
  holdout_error = fread('bias_summary.csv')
}else{
  pred_files = list.files(results_dir, pattern = '^party', full.names = T)
  
  holdout_error = rbindlist(lapply(pred_files, function(f){
    temp = fread(f)
    #cat(paste(f, '\n'))
    holdout_ind = which(temp[model == 'logit',]$holdout == 1)
    
    temp$act_class = rep(pew_data$support, length(unique(temp$model)))
    temp[is.na(temp$y_hat_oth), ] <- 0
    temp[, pred_class := c('1-Dem', '2-Rep', '3-Oth')[apply(temp[, .(y_hat_dem, y_hat_rep, y_hat_oth)], 1, which.max)]]
    temp[, correct_class := as.numeric(act_class == pred_class)]
    
    holdout_error = cbind(temp[holdout == 1, .(y_hat_dem = mean(y_hat_dem)
                                               , y_hat_rep = mean(y_hat_rep)
                                               , y_hat_oth = mean(y_hat_oth)
                                               , class_rate = mean(correct_class)
    ), by = .(model, results_id, match_rate, n_bags, n_landmarks, party)]
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
  
  write.csv(holdout_error, file = 'bias_summary.csv', row.names = F)
}

#rename logit
holdout_error[, model := gsub('logit', 'lasso', model)]


#_------------------------------
##PLOT FUNCTIONS

plot_mse = function(data, x, x_lab = NULL){
  if(is.null(x_lab)){
    x_lab = gsub("_", " ", x)
  }
  
  ggplot(data = data, aes(x = get(x), y = mse, color = model)) + 
    geom_point(alpha = 0.2) +
    geom_smooth() + 
    xlab(x_lab) +
    ylab("MSE") +
    facet_grid(~party, labeller = labeller(party = function(p) paste0('party: ', p))) + 
    theme_light() +
    ggtitle(paste("Holdout MSE by", x_lab))
}

plot_bias = function(data, x, x_lab = NULL){
  if(is.null(x_lab)){
    x_lab = gsub("_", " ", x)
  }
  
  ggplot(data, aes(x = get(x),y = error_dem, color = model)) + 
    geom_hline(yintercept = 0, color = 'red') +
    geom_point(alpha = 0.2) +
    xlab(x_lab) +
    ylab("Bias of estimated % Dem") +
    geom_smooth() +
    facet_grid(party~model, labeller = labeller(party = function(p) paste0('party: ', p))) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position="bottom") +
    ggtitle(paste("Holdout Dem bias by", x_lab))
}


#----------------------------------------
# PLOTS


# TOPLINE
mses[, model_order := factor(model, levels = c('lasso_alldata', 'lasso','grpmean'
                                              , 'dr_sepbags', 'dr_sepbags_lin', 'dr_sepbags_cust', 'wdr_sepbags'
                                              , 'dr', 'dr_linear','dr_cust', 'wdr','wdr_linear'
                                              , 'lasso_bagfeat', 'lasso_regfeat'))]
holdout_error[, model_order := factor(model, levels = c('lasso_alldata', 'lasso','grpmean'
                                               , 'dr_sepbags', 'dr_sepbags_lin', 'dr_sepbags_cust', 'wdr_sepbags'
                                               , 'dr', 'dr_linear','dr_cust', 'wdr','wdr_linear'
                                               , 'lasso_bagfeat', 'lasso_regfeat'))]
plot_mse_dist = ggplot(mses[mse < 0.3], aes(x = model_order, y = mse )) + geom_boxplot() + 
  facet_grid(~party, scales = 'free', labeller = labeller(party = function(p) paste0('party: ', p))) + 
  coord_flip() +
  xlab('') +
  theme_light() +
  scale_x_discrete(limits = rev(levels(mses$model_order))) + 
  ggtitle("Distribution of MSE by model and party")
plot_mse_dist
ggsave(plot_mse_dist, filename = 'plots/plot_mse_dist.png', width = 8, height = 4)

plot_bias_dist = ggplot(holdout_error, aes(x = model_order, y = error_dem)) + 
  geom_hline(yintercept = 0, color = 'red') + facet_grid(~party) + 
  geom_boxplot() + 
  coord_flip() +
  scale_x_discrete(limits = rev(levels(mses$model_order))) + 
  xlab('') +
  theme_light() +
  ggtitle("Bias in Estimated % Dem")
plot_bias_dist
ggsave(plot_bias_dist, filename = 'plots/plot_bias_dist.png', width = 8, height = 4)


model_subset = c('lasso_alldata', 'lasso', 'grpmean', 'dr','wdr','dr_sepbags', 'lasso_bagfeat', 'lasso_regfeat')

# MATCH RATE
plot_matchrate_mse = plot_mse(mses[model %in% model_subset], x = 'match_rate')
plot_matchrate_mse
ggsave(plot_matchrate_mse, filename = 'plots/plot_matchrate_mse.png', width = 8, height = 4)

plot_matchrate_bias = plot_bias(holdout_error[model %in% model_subset], x = 'match_rate')
plot_matchrate_bias
ggsave(plot_matchrate_bias, filename = 'plots/plot_matchrate_bias.png', width = 10, height = 5)


# N BAGS
plot_nbags_mse = plot_mse(mses[model %in% model_subset], x = 'n_bags', x_lab = 'number of bags')
plot_nbags_mse
ggsave(plot_nbags_mse, filename = 'plots/plot_nbags_mse.png', width = 8, height = 4)

plot_nbags_bias = plot_bias(holdout_error[model %in% model_subset], x = 'n_bags', x_lab = 'number of bags')
plot_nbags_bias
ggsave(plot_nbags_bias, filename = 'plots/plot_nbags_bias.png', width = 10, height = 5)



# OBSERVATIONS PER BAG
mses[, avg_obs_per_bag := (match_rate * 2000)/n_bags]
holdout_error[, avg_obs_per_bag := (match_rate * 2000)/n_bags]

plot_obsperbag_mse = plot_mse(mses[model %in% model_subset], x = 'avg_obs_per_bag', x_lab = 'avg observations per bag')
plot_obsperbag_mse
ggsave(plot_obsperbag_mse, filename = 'plots/plot_obsperbag_mse.png', width = 8, height = 4)

plot_obsperbag_bias = plot_bias(holdout_error[model %in% model_subset], x = 'avg_obs_per_bag', x_lab = 'avg observations per bag')
plot_obsperbag_bias
ggsave(plot_obsperbag_bias, filename = 'plots/plot_obsperbag_bias.png', width = 10, height = 5)


plot_obsperbag_mse_refit = ggplot(mses[model %in% c('dr_sepbags', 'wdr_sepbags')], 
       aes(x = avg_obs_per_bag,y = mse, color = refit_bags)) + 
  geom_point(alpha = 0.2) +
  geom_smooth() + 
  facet_grid(party~model, scales = 'free') +
  ggtitle("Holdout MSE by avg number of observations per bag") +
  theme_light()+
  theme(legend.title=element_text(size=10), legend.position="bottom") +
  xlab("Avg number of observations per bag") +
  ylab("MSE") +
  scale_color_discrete(name = "Separate bags for matched data")
plot_obsperbag_mse_refit
ggsave(plot_obsperbag_mse_refit, filename = 'plots/plot_obsperbag_mse_refit.png', width = 6, height = 4)

plot_obsperbag_bias_refit = ggplot(holdout_error[model %in% c('dr_sepbags', 'wdr_sepbags')], 
       aes(x = avg_obs_per_bag,y = error_dem, color = refit_bags)) + 
  geom_hline(yintercept = 0, color = 'red', lty = 2) +
  geom_point(alpha = 0.2) +
  geom_smooth() + 
  facet_grid(party~model, scales = 'free') +
  ggtitle("Holdout Dem bias by avg number of observations per bag") +
  xlab("Avg number of observations per bag") +
  ylab("Bias of estimated % Dem") +
  theme_light()+
  theme(legend.position='bottom')
plot_obsperbag_bias_refit
ggsave(plot_obsperbag_bias_refit, filename = 'plots/plot_obsperbag_bias_refit.png', width = 6, height = 4)


# LANDMARKS

plot_landmarks_mse = plot_mse(mses[model %in% model_subset], x = 'n_landmarks', 'number of landmark points')
plot_landmarks_mse
ggsave(plot_landmarks_mse, filename = 'plots/plot_landmarks_mse.png', width = 8, height = 4)

plot_landmarks_bias = plot_bias(holdout_error[model %in% model_subset], x = 'n_landmarks', 'number of landmark points')
plot_landmarks_bias
ggsave(plot_nbags_bias, filename = 'plots/plot_landmarks_bias.png', width = 10, height = 5)


# SEPARATE BAGS FOR MATCHED DATA
plot_sepbags_mse = ggplot(mses[model %in% c('dr_sepbags', 'wdr_sepbags')], 
       aes(x = n_bags,y = mse, color = refit_bags)) + 
  geom_point(alpha = 0.2) +
  geom_smooth() + 
  facet_grid(party~model, scales = 'free_y') +
  ggtitle("Holdout MSE by number of bags") +
  theme_light()+
  ylab("MSE") +
  theme(legend.title=element_text(size=10)) +
  xlab("Number of bags") +
  scale_color_discrete(name = "Bagging uses\nonly unmatched data")
plot_sepbags_mse
ggsave(plot_sepbags_mse, filename = 'plots/plot_sepbags_mse.png', width = 8, height = 4)


plot_sepbags_bias = ggplot(holdout_error[model %in% c('dr_sepbags', 'wdr_sepbags')], 
                          aes(x = n_bags,y = error_dem, color = refit_bags)) + 
  geom_hline(yintercept = 0, color = 'red')+
  geom_point(alpha = 0.2) +
  geom_smooth() + 
  facet_grid(party~model, scales = 'free_y') +
  ggtitle("Holdout MSE by number of bags") +
  theme_light()+
  ylab("Bias of estimated % Dem") +
  theme(legend.title=element_text(size=10)) +
  xlab("Number of bags") +
  scale_color_discrete(name = "Bagging uses\nonly unmatched data")
plot_sepbags_bias
ggsave(plot_sepbags_bias, filename = 'plots/plot_sepbags_bias.png', width = 8, height = 4)

#-----------------------------


plot_prob_dist = ggplot(pew_data) + 
  geom_density(aes(x = p_surveyed, color = 'P(surveyed)')) +
  geom_density(aes(x = p_matched, color = 'P(matched)')) +
  ggtitle("Distribution of response and match probabilities") +
  theme_light() +
  xlab("probability")
plot_prob_dist
ggsave(plot_prob_dist, filename = 'plots/plot_prob_dist.png', width = 6, height = 3)



mses[, .(median(mse)), .(party, model)]


rand_pred = data.frame(y_hat = pew_data[, sample(c('1-Dem','2-Rep','3-Other'), size = nrow(pew_data)
                                                 , replace = T
                                                 #, prob = c(0.48, 0.42, 0.1)
)])
rand_pred = modmat_all_levs(~-1 + y_hat, data = rand_pred)

calcMSE(Y = unlist(pew_data[, .(y_dem, y_rep, y_oth)]), Y_pred = unlist(rand_pred))

party_id = pew_data[, .(y_dem = as.numeric(grepl('Dem', demo_party))
                        , y_rep = as.numeric(grepl('Rep', demo_party))
                        , y_oth = as.numeric(!grepl('Rep|Dem', demo_party))
)]
calcMSE(Y = unlist(pew_data[, .(y_dem, y_rep, y_oth)]), Y_pred = unlist(party_id))
