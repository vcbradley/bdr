## Clean Pew Data

rm(list = ls())
library(data.table)
library(ggplot2)
library(memisc)
library(foreign)


setwd('~/github/bdr')
source('functions.R')

# import data
data_sept18 = data.table(read.spss('data/Sept18/Sept18 public.sav', to.data.frame = T), stringsAsFactors = F)
data_june18 = data.table(read.spss('data/June18/June18 public.sav', to.data.frame = T), stringsAsFactors = F)
data_may18 = data.table(read.spss('data/May18/May18 public.sav', to.data.frame = T), stringsAsFactors = F)

# RECODE all data sets
data_recoded = rbindlist(lapply(list(data_sept18, data_may18, data_june18), doPewRecode))
# write.csv(data_recoded, file = 'data/pew_recoded.csv', row.names = F)

data_recoded


##### Set prob of being surveyed
data_recoded[, age_scaled := scale(age_num)]
data_recoded[is.na(age_scaled), age_scaled := 0]

data_recoded[, p_surveyed := 
               (-2)
             + 2 * age_scaled 
             - 0.5 * is.na(age_num) 
             + 1.5 * as.numeric(demo_mode == 'cell') 
             - 1.5 * as.numeric(demo_party == '05-Ind')
             - 3 * as.numeric(demo_party == "99-DK/refused") 
             + 1.5 * as.numeric(demo_education %in% c('01-postgrad', '02-bach'))
             + 3 * as.numeric(demo_ideology == 'Very conservative' | demo_ideology == 'Very liberal')
             ]
data_recoded[, p_surveyed := exp(p_surveyed)/(1 + exp(p_surveyed))]
hist(data_recoded[, p_surveyed])

data_recoded[, .(.N, mean(p_surveyed)), .(demo_age_bucket)][order(demo_age_bucket)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_mode)][order(demo_mode)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_party)][order(demo_party)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_ideology)][order(demo_ideology)]



###### Set prob of matching
data_recoded[, p_matched := NULL]
data_recoded[, p_matched :=
               -2 +
               -2 * as.numeric(demo_mode == 'landline') 
             + 3 * as.numeric(demo_race == 'W')
             + -2 * as.numeric(demo_reg == '03-No')
             + -1 * as.numeric(demo_hhsize == 2)
             + -2 * as.numeric(demo_hhsize == 3)
             + 2 *age_scaled
             + as.numeric(demo_income)/3
             - 4* as.numeric(demo_income == '99-DK/refused')
             ]
data_recoded[, p_matched := exp(p_matched)/(1 + exp(p_matched))]
hist(data_recoded$p_matched)

data_recoded[, .(.N, mean(p_matched)), demo_mode]
data_recoded[, .(.N, mean(p_matched)), demo_hispanic]
data_recoded[, .(.N, mean(p_matched)), demo_age_bucket][order(demo_age_bucket)]

ggplot(data_recoded, aes(x = p_surveyed, y = p_matched)) + geom_point() + ggtitle("P(matched) v. P(surveyed)")


# few more recodes
pew_data[y_dem == 1, support := '1-Dem']
pew_data[y_rep == 1, support := '2-Rep']
pew_data[y_oth == 1, support := '3-Other']


# mean impute age
pew_data[, age_num_imp := as.numeric(age_num)]
pew_data[is.na(age_num_imp), age_num_imp := mean(pew_data$age_num, na.rm = T)]
summary(pew_data[, age_num_imp])

#scale age
pew_data[, age_scaled := scale(age_num_imp)/5]

# write to CSV
write.csv(data_recoded, file = 'data/pew_data.csv', row.names = F)
