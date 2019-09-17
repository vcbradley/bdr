## Clean Pew Data

rm(list = ls())
library(data.table)
library(ggplot2)
library(memisc)
library(foreign)


setwd('~/github/bdr')
x = sourceDirectory('~/github/bdr/utils', modifiedOnly=FALSE)

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
             + -0.1 * month_called
             ]
data_recoded[, p_surveyed := exp(p_surveyed)/(1 + exp(p_surveyed))]
hist(data_recoded[, p_surveyed])

data_recoded[, .(.N, mean(p_surveyed)), .(demo_age_bucket)][order(demo_age_bucket)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_mode)][order(demo_mode)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_party)][order(demo_party)]
data_recoded[, .(.N, mean(p_surveyed)), .(demo_ideology)][order(demo_ideology)]
data_recoded[, .(.N, mean(p_surveyed)), .(month_called)][order(month_called)]



###### Set prob of matching
data_recoded[, p_matched := NULL]
data_recoded[, p_matched :=
               -2 +
               -2 * as.numeric(demo_mode == 'landline') 
             +  as.numeric(demo_phonetype == '01-Both') 
             - 3 * as.numeric(demo_race == 'W')
             -2 * as.numeric(demo_reg == '03-No')
             + -1 * as.numeric(demo_hhsize == 2)
             + -2 * as.numeric(demo_hhsize == 3)
             + 2 *age_scaled^2
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
data_recoded[y_dem == 1, support := '1-Dem']
data_recoded[y_rep == 1, support := '2-Rep']
data_recoded[y_oth == 1, support := '3-Other']


# mean impute age
data_recoded[, age_num_imp := as.numeric(age_num)]
data_recoded[is.na(age_num_imp), age_num_imp := mean(data_recoded$age_num, na.rm = T)]
summary(data_recoded[, age_num_imp])

#scale age
data_recoded[, age_scaled := scale(age_num_imp)/5]

# write to CSV
write.csv(data_recoded, file = 'data/data_recoded_v2.csv', row.names = F)
