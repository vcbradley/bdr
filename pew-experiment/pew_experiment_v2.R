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

## get test train sets
testtrain = getTestTrain(data = pew_data
                         , n_holdout = 1000, n_surveyed = 2000, n_matched = 1000
                         , p_surveyed = pew_data$p_surveyed
                         , p_matched = pew_data$p_matched
)
pew_data[, .(.N, mean(y_dem)), .(holdout, surveyed, matched, voterfile)]


# get bags
bags = getBags(data = pew_data[surveyed == 1, ]  # bag with survey data
        , vars = file_and_survey_vars  # and overlapping variables
        , n_bags = n_bags
        , newdata = pew_data)




