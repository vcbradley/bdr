import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
import datetime as dt
from sklearn import preprocessing as prep

# set directory
import os
os.chdir('/Users/valeriebradley/github/libdems/projection/')


pandas2ri.activate()  # to translate R obj into pandas df

data_path = '~/Documents/LibDems/data/'

r['load'](data_path+"model_sample_data_cleaned.RData")
dr_covars = r.dr_covars
dr_covars

dr_covars['VANID'] = dr_covars['VANID'].astype(int)
dr_covars.set_index('VANID', inplace = True)

# calculate reg timing
eday = '2017-06-08'
dr_covars['year_reg'] = [dt.datetime.strptime(date, '%Y-%m-%d').date().year for date in dr_covars['DATE_OF_UPDATE'].values]
dr_covars['wks_from_reg_to_eday'] = [(dt.datetime.strptime(eday, '%Y-%m-%d') - dt.datetime.strptime(date, '%Y-%m-%d'))/ dt.timedelta (days=1) for date in dr_covars['DATE_OF_UPDATE'].values]

reg_timing_cuts = [-100000,0,6,52,260,100000]
dr_covars['demo_reg_timing'] = pd.cut(dr_covars['wks_from_reg_to_eday'], bins = reg_timing_cuts
       , labels = ['1-After Eday', '2-During campaign', '3-Same year as election', '4-Past 10 years', '5-1999'])



# fill missing values
dr_covars['eth_white'] = dr_covars['eth_white'].fillna(0)
dr_covars['eth_easteur'] = dr_covars['eth_easteur'].fillna(0)
dr_covars['eth_eu'] = dr_covars['eth_eu'].fillna(0)
dr_covars['eth_migrant'] = dr_covars['eth_migrant'].fillna(0)
dr_covars['eth_southasian'] = dr_covars['eth_southasian'].fillna(0)
dr_covars['eth_multi'] = dr_covars['eth_multi'].fillna(0)
dr_covars['eth_asian'] = dr_covars['eth_asian'].fillna(0)


# scale lat/lon data
X_latlong = pd.DataFrame(prep.scale(dr_covars[['latitude_imp', 'longitude_imp']]), index = dr_covars.index)
X_latlong.columns = ['latitude_imp', 'longitude_imp']
X_latlong = pd.concat([dr_covars[['code_num', 'constituency']], X_latlong], axis = 1)


# categorize columns
demo_cols = ['demo_region', 'demo_sex', 'demo_age_bucket', 'demo_counciltax','demo_area_class'
             , 'demo_pct_with_deg', 'demo_mult_dep_ind', 'demo_reg_timing', 'mr_contact'
                ]
demo_ind_cols = ['eth_white'
                , 'eth_easteur'
                , 'eth_eu'
                , 'eth_migrant'
                , 'eth_southasian'
                , 'eth_multi'
                , 'eth_asian'
                , 'demo_ht_elderly'
                , 'demo_ht_student'
                , 'member_flag']
#  contact_hist_cols = names(data)[grepl('ever|switch', names(data))]
score_cols = ['ld_2019_imp','lab_2019_imp', 'con_2019_imp', 'snp_2019_imp', 'bp_2019_imp'
                , 'remain_2019_imp','unscored_2019', 'euref_remain_imp']
    
vh_cols = ["vote_eup2019", "vote_l2019", "vote_l2017", "vote_w2017", "vote_ref2016"]



# one-hot encode categorical cols
X_demo = pd.get_dummies(dr_covars[demo_cols]
                             , prefix_sep="__" 
                             , columns=demo_cols
                            )
X_demo = pd.concat([X_demo, dr_covars[demo_ind_cols], dr_covars[vh_cols]], axis=1)
X_demo_scaled = pd.DataFrame(prep.scale(X_demo), index = dr_covars.index)
X_demo_scaled.columns = X_demo.columns

# concatenate all demo columns
X_demo = pd.concat([dr_covars[['code_num', 'constituency']], X_demo_scaled], axis=1)



# scale score columns
X_scores = pd.DataFrame(prep.scale(dr_covars[score_cols]), index = dr_covars.index)
X_scores.columns = score_cols
X_scores = pd.concat([dr_covars[['code_num', 'constituency']], X_scores], axis=1)


########################
# SAVE DATA AS PICKLES #
########################
X_latlong.to_pickle(data_path + '/projection_data/X_latlong.pkl')
X_scores.to_pickle(data_path + '/projection_data/X_scores.pkl')
X_demo.to_pickle(data_path + '/projection_data/X_demo.pkl')




