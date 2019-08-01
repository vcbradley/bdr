
library(rstan)
library(ggplot2)
library(data.table)

# set cores for parallelization
options(mc.cores = parallel::detectCores())

setwd('~/github/bdr')
source('functions.R')

# read in data
pew_data = fread('data/pew_recoded.csv')

# specify which model
stan_model = stan_model(file = 'stan/bdr-mean-shrinkage.stan')

# set params
n_bags = 50
n_landmarks = 100
sigma = 0.003
n_holdout = 1000
n_surveyed = 2000
n_matched = 1000

# specify where each var is observed (survey, file or both)
survey_vars = c('demo_mode', 'demo_education', 'demo_phonetype', 'month_called', 'demo_ideology', 'demo_party')
file_and_survey_vars = c('demo_sex', 'demo_age_bucket', 'demo_state', 'demo_income', 'demo_region', 'demo_race', 'demo_hispanic')

# get test and train
testtrain = getTestTrain(data = pew_data
                         , n_holdout = n_holdout
                         , n_surveyed = n_surveyed
                         , n_matched = n_matched)



# prep data
stan_data = list(n_bags
                 , p
                 , d
                 , b
                 , bags
                 , mu
                 , Y)

samples = sampling(object = stan_model, data = stan_data)

# optimize model
map = optimizing(object = stan_model, data = stan_data)



