
# cd ~/github/libdems/projection/
# source env/bin/activate

import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri

# set directory
import os
os.chdir('/Users/valeriebradley/github/libdems/projection/')

pandas2ri.activate()


data_path = '~/Documents/LibDems/data/'


#############
#### 2017 GE RESULTS
results_ge2017 = pd.read_csv('~/Documents/LibDems/Background/HoC-GE2017-constituency-results.csv')
results_ge2017.rename(columns = {'constituency_name':'constituency', 'ons_id':'code', 'electorate':'nreg_ge2017', 'valid_votes': 'votes_ge2017'}, inplace = True)

# make subset
results_ge2017_subset = results_ge2017[['code', 'constituency', 'region_name', 'nreg_ge2017', 'votes_ge2017']]

# calculate percentages
results_ge2017_pct = results_ge2017[['con','lab', 'ld','ukip','green','snp', 'pc']].apply(lambda x: x / results_ge2017['votes_ge2017'], axis = 0)
results_ge2017_pct.columns = ['pct_ge17_' + col for col in results_ge2017_pct.columns]

# combine
results_ge2017_subset = pd.concat([results_ge2017_subset, results_ge2017_pct], axis = 1)

# calculate turnout pct
results_ge2017_subset['pct_turnout_ge2017'] = results_ge2017_subset['votes_ge2017']/results_ge2017_subset['nreg_ge2017']

# final
results_ge2017_subset


#############
#### BES DATA
bes_data = pd.read_csv('/Users/valeriebradley/Documents/LibDems/Background/BES/bes_1719_supportturnout.csv')

# make dummies
bes_data = pd.get_dummies(bes_data
                             , prefix="bes19pre_supp_" 
                             , columns=['support19pre_recoded']
                            )

bes_data = pd.get_dummies(bes_data
                             , prefix="bes19pre_turn_" 
                             , columns=['turnout19pre']
                            )
# drop the observations without 2019 support responses
bes_support19 = bes_data.dropna(subset=['code', 'support19pre'], how='any')

# aggregate WEIGHTED support at the constit level
support_cols = [col for col in bes_support19.columns if '_supp__' in col]
bes_support19_constit = bes_support19.groupby('code')[support_cols].agg(lambda x: np.sum(x * bes_support19.loc[x.index, "weight19pre"]))

#drop first row where we're missing code
bes_support19_constit = bes_support19_constit.drop(bes_support19_constit.index[0])  
bes_support19_constit



#### MRP SEPT DATA
ld_polls = pd.read_csv('/Users/valeriebradley/Documents/LibDems/polling/polling_allpolls.csv')

# get MRP data subset
mrpdata = ld_polls[ld_polls['pollid']== 'mrpsept']

mrpdata = pd.get_dummies(mrpdata
                         , prefix="mrpsept_supp_" 
                         , columns=['q_support_ge2019']
                        )
support_cols = [col for col in mrpdata.columns if 'mrpsept_supp_' in col]
support_cols = support_cols[0:10]   # drop refused

# group by constit
mrpdata['constituency_lower'] = mrpdata['constituency'].str.replace('&', 'and').str.lower()
mrpdata_constit = mrpdata.groupby('constituency_lower')[support_cols].agg('sum')
mrpdata_constit = mrpdata_constit.div(mrpdata_constit.sum(axis = 1), axis = 0)
mrpdata_constit



#### MRP DEC DATA
mrpdec = pd.read_csv('/Users/valeriebradley/Documents/LibDems/yougov_mrp_dec.csv')
mrpdec_pct = mrpdec[['Con', 'Lab','LD','Brexit','Green','SNP','PC','Other']].div(100,axis = 1)
mrpdec_pct.columns = ['mrpdec_supp_' + col.lower() for col in mrpdec_pct.columns]
mrpdec = pd.concat([mrpdec[['code', 'constituency']], mrpdec_pct], axis = 1)
mrpdec



#### STANDING FLAGS
standing_flags = pd.read_csv('/Users/valeriebradley/Documents/LibDems/standing_flags.csv')
standing_flags.columns = [col if (col == 'constituency' or col == 'code') else 'standing_'+ col.lower() for col in standing_flags.columns]
standing_flags


##### MERGE EVERYTHING TOGETHER
outcome_data = pd.merge(results_ge2017_subset, standing_flags
         , right_on = ['constituency','code']
         , left_on = ['constituency', 'code'], how = 'outer')

outcome_data = pd.merge(outcome_data, bes_support19_constit
         , right_on = ['code']
         , left_on = ['code'], how = 'outer')

outcome_data['constituency_lower'] = outcome_data['constituency'].str.lower()
outcome_data = pd.merge(outcome_data, mrpdata_constit
         , left_on = ['constituency_lower']
         #, right_on = ['constituency_lower']
         , right_index = True
         , how = 'outer')

outcome_data = pd.merge(outcome_data, mrpdec
         , left_on = ['constituency', 'code']
         , right_on = ['constituency', 'code']
         , how = 'outer')

outcome_data.columns


## SAVE DATA
outcome_data.to_pickle(data_path + 'projection_data/outcome_data.pkl')



