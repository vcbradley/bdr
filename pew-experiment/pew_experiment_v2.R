library(data.table)
library(entropy)
library(gridExtra)
library(ggplot2)
library(Rfast)

setwd('~/github/bdr')

source('functions.R') # contains the key functions for basic dist reg

survey_vars = c('demo_mode', 'demo_education', 'demo_phonetype', 'month_called', 'demo_ideology', 'demo_party')
file_and_survey_vars = c('demo_sex', 'demo_age_bucket', 'demo_state', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')

# read in *recoded* data
pew_data = fread('data/pew_data.csv')

# set n_bags
n_bags = 30
n_landmarks = 50


# get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000, n_surveyed = 2000, n_matched = 1000
                         , p_surveyed = pew_data$p_surveyed
                         , p_matched = pew_data$p_matched
)
pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]


####### bags with whole file v. just survey ########
bag_iter = 50
bag_svy_mat = matrix(rep(0, bag_iter * nrow(pew_data)), nrow = nrow(pew_data))
bag_all_mat = matrix(rep(0, bag_iter * nrow(pew_data)), nrow = nrow(pew_data))

for(i in 1:bag_iter){
  cat(paste(i, '\n'))
  bag_svy_mat[,i] = getBags(data = pew_data[surveyed == 1, ]
                         , vars = file_and_survey_vars
                         , n_bags = n_bags
                         , newdata = pew_data)$bags_newdata
  bag_all_mat[,i] = getBags(data = pew_data[holdout == 0, ]
                     , vars = file_and_survey_vars
                     , n_bags = n_bags
                     , newdata = pew_data)$bags_newdata
  
}

bag_svy_mat = melt(bag_svy_mat, varnames = c('id', 'iter'), value.name = 'bag_svy')
bag_all_mat = melt(bag_all_mat, varnames = c('id', 'iter'), value.name = 'bag_all')
bag_mat = data.table(merge(bag_svy_mat, bag_all_mat, by = c('id','iter')))



#### Bag size
plot_bagsize_all = ggplot() + 
  geom_density(aes(bag_mat[, .(.N), .(bag_svy, iter)]$N, color = 'bagged with survey')) +
  geom_density(aes(bag_mat[, .(.N), .(bag_all, iter)]$N, color = 'bagged with all data')) +
  xlab("Bag Size") +
  ggtitle("Dist of bag size - ")

plot_bagsize_svy = ggplot() + 
  geom_density(aes(bag_mat[id %in% pew_data[, which(surveyed == 1)], .(.N), .(bag_svy, iter)]$N, color = 'bagged with survey')) +
  geom_density(aes(bag_mat[id %in% pew_data[, which(surveyed == 1)], .(.N), .(bag_all, iter)]$N, color = 'bagged with all data')) +
  xlab("Bag Size") +
  ggtitle("Dist of bag size within survey data")

grid.arrange(plot_bagsize_all, plot_bagsize_svy)

# do we always have 30 bags in survey data? 
bag_mat[id %in% pew_data[, which(surveyed == 1)], .(length(unique(bag_svy)), length(unique(bag_all))), iter]

#### Bag similarity - LOTS of overlap
bag_mat[, iter_bag_svy := interaction(iter, bag_svy)]
bag_mat[, iter_bag_all := interaction(iter, bag_all)]

bag_svy_overlap = merge(bag_mat, bag_mat, by = 'iter_bag_svy', allow.cartesian=TRUE)[, .N, by = .(id.x,id.y)]
bag_all_overlap = merge(bag_mat, bag_mat, by = 'iter_bag_all', allow.cartesian=TRUE)[, .N, by = .(id.x,id.y)]

ggplot() + 
  geom_density(data = bag_svy_overlap[id.x != id.y,], aes(x = N, color = 'svy')) +
  geom_density(data = bag_all_overlap[id.x != id.y,], aes(x = N, color = 'all')) +
  ggtitle("Bag overlap") +
  xlab("Number of overlaps")





