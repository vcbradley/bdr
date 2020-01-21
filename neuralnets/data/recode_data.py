import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri
import datetime as dt
from sklearn import preprocessing as prep

# set directory
import os
os.chdir('/Users/valeriebradley/github/libdems/projection/')
pandas2ri.activate()  # to translate R obj into pandas df

def recode_covar_data():

    data_path = '~/Documents/LibDems/data/'

    r['load'](data_path + "model_sample_data_cleaned.RData")
    dr_covars = r.dr_covars

    dr_covars['VANID'] = dr_covars['VANID'].astype(int)
    dr_covars.set_index('VANID', inplace=True)

    # calculate reg timing
    eday = '2017-06-08'
    dr_covars['year_reg'] = [dt.datetime.strptime(date, '%Y-%m-%d').date().year for date in
                             dr_covars['DATE_OF_UPDATE'].values]
    dr_covars['wks_from_reg_to_eday'] = [
        (dt.datetime.strptime(eday, '%Y-%m-%d') - dt.datetime.strptime(date, '%Y-%m-%d')) / dt.timedelta(days=1) for date in
        dr_covars['DATE_OF_UPDATE'].values]

    reg_timing_cuts = [-100000, 0, 6, 52, 260, 100000]
    dr_covars['demo_reg_timing'] = pd.cut(dr_covars['wks_from_reg_to_eday'], bins=reg_timing_cuts
                                          , labels=['1-After Eday', '2-During campaign', '3-Same year as election',
                                                    '4-Past 10 years', '5-1999'])

    # fill missing values
    dr_covars['eth_white'] = dr_covars['eth_white'].fillna(0)
    dr_covars['eth_easteur'] = dr_covars['eth_easteur'].fillna(0)
    dr_covars['eth_eu'] = dr_covars['eth_eu'].fillna(0)
    dr_covars['eth_migrant'] = dr_covars['eth_migrant'].fillna(0)
    dr_covars['eth_southasian'] = dr_covars['eth_southasian'].fillna(0)
    dr_covars['eth_multi'] = dr_covars['eth_multi'].fillna(0)
    dr_covars['eth_asian'] = dr_covars['eth_asian'].fillna(0)

    # scale lat/lon data
    X_latlong = pd.DataFrame(prep.scale(dr_covars[['latitude_imp', 'longitude_imp']]), index=dr_covars.index)
    X_latlong.columns = ['latitude_imp', 'longitude_imp']
    X_latlong = pd.concat([dr_covars[['code_num', 'constituency']], X_latlong], axis=1)

    # categorize columns
    demo_cols = ['demo_region', 'demo_sex', 'demo_age_bucket', 'demo_counciltax', 'demo_area_class'
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
    score_cols = ['ld_2019_imp', 'lab_2019_imp', 'con_2019_imp', 'snp_2019_imp', 'bp_2019_imp'
        , 'remain_2019_imp', 'unscored_2019', 'euref_remain_imp']

    vh_cols = ["vote_eup2019", "vote_l2019", "vote_l2017", "vote_w2017", "vote_ref2016"]

    # one-hot encode categorical cols
    X_demo = pd.get_dummies(dr_covars[demo_cols]
                            , prefix_sep="__"
                            , columns=demo_cols
                            )
    X_demo = pd.concat([X_demo, dr_covars[demo_ind_cols], dr_covars[vh_cols]], axis=1)
    X_demo_scaled = pd.DataFrame(prep.scale(X_demo), index=dr_covars.index)
    X_demo_scaled.columns = X_demo.columns

    # concatenate all demo columns
    X_demo = pd.concat([dr_covars[['code_num', 'constituency']], X_demo_scaled], axis=1)

    # scale score columns
    X_scores = pd.DataFrame(prep.scale(dr_covars[score_cols]), index=dr_covars.index)
    X_scores.columns = score_cols
    X_scores = pd.concat([dr_covars[['code_num', 'constituency']], X_scores], axis=1)

    ########################
    # SAVE DATA AS PICKLES #
    ########################
    X_latlong.to_pickle(data_path + '/projection_data/X_latlong.pkl')
    X_scores.to_pickle(data_path + '/projection_data/X_scores.pkl')
    X_demo.to_pickle(data_path + '/projection_data/X_demo.pkl')

    X_all = np.concatenate((X_latlong.iloc[:, 2:], X_scores.iloc[:, 2:], X_demo.iloc[:, 2:]), axis=1)
    var_cat = np.concatenate((np.repeat('latlong', X_latlong.shape[1] - 2),
                                  np.repeat('scores', X_scores.shape[1] - 2),
                                  np.repeat('demo', X_demo.shape[1] - 2)))

    np.save(data_path + '/projection_data/X_all.npy', X_all)
    np.save(data_path + '/projection_data/var_cat.npy', var_cat)

    return X_demo[['code_num', 'constituency']]


def recode_outcome_data(constit_code_tbl):
    data_path = '~/Documents/LibDems/data/'

    #############
    #### 2017 GE RESULTS
    results_ge2017 = pd.read_csv('~/Documents/LibDems/Background/HoC-GE2017-constituency-results.csv')
    results_ge2017.rename(columns={'constituency_name': 'constituency', 'ons_id': 'code', 'electorate': 'nreg_ge2017',
                                   'valid_votes': 'votes_ge2017'}, inplace=True)

    # make subset
    results_ge2017_subset = results_ge2017[['code', 'constituency', 'region_name', 'nreg_ge2017', 'votes_ge2017']]

    # calculate percentages
    results_ge2017_pct = results_ge2017[['con', 'lab', 'ld', 'ukip', 'green', 'snp', 'pc']].apply(
        lambda x: x / results_ge2017['votes_ge2017'], axis=0)
    results_ge2017_pct.columns = ['pct_ge17_' + col for col in results_ge2017_pct.columns]

    # combine
    results_ge2017_subset = pd.concat([results_ge2017_subset, results_ge2017_pct], axis=1)

    # calculate turnout pct
    results_ge2017_subset['pct_turnout_ge2017'] = results_ge2017_subset['votes_ge2017'] / results_ge2017_subset[
        'nreg_ge2017']


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
    bes_support19_constit = bes_support19.groupby('code')[support_cols].agg(
        lambda x: np.sum(x * bes_support19.loc[x.index, "weight19pre"]))

    # drop first row where we're missing code
    bes_support19_constit = bes_support19_constit.drop(bes_support19_constit.index[0])


    ##############
    #### MRP SEPT DATA
    ld_polls = pd.read_csv('/Users/valeriebradley/Documents/LibDems/polling/polling_allpolls.csv')

    # get MRP data subset
    mrpdata = ld_polls[ld_polls['pollid'] == 'mrpsept']

    mrpdata = pd.get_dummies(mrpdata
                             , prefix="mrpsept_supp_"
                             , columns=['q_support_ge2019']
                             )
    support_cols = [col for col in mrpdata.columns if 'mrpsept_supp_' in col]
    support_cols = support_cols[0:10]  # drop refused

    # group by constit
    mrpdata['constituency_lower'] = mrpdata['constituency'].str.replace('&', 'and').str.lower()
    mrpdata_constit = mrpdata.groupby('constituency_lower')[support_cols].agg('sum')
    mrpdata_constit = mrpdata_constit.div(mrpdata_constit.sum(axis=1), axis=0)

    #### MRP DEC DATA
    mrpdec = pd.read_csv('/Users/valeriebradley/Documents/LibDems/yougov_mrp_dec.csv')
    mrpdec_pct = mrpdec[['Con', 'Lab', 'LD', 'Brexit', 'Green', 'SNP', 'PC', 'Other']].div(100, axis=1)
    mrpdec_pct.columns = ['mrpdec_supp_' + col.lower() for col in mrpdec_pct.columns]
    mrpdec = pd.concat([mrpdec[['code', 'constituency']], mrpdec_pct], axis=1)

    #### STANDING FLAGS
    standing_flags = pd.read_csv('/Users/valeriebradley/Documents/LibDems/standing_flags.csv')
    standing_flags.columns = [col if (col == 'constituency' or col == 'code') else 'standing_' + col.lower() for col in
                              standing_flags.columns]

    ##################
    ##### MERGE EVERYTHING TOGETHER
    outcome_data = pd.merge(results_ge2017_subset, standing_flags
                            , right_on=['constituency', 'code']
                            , left_on=['constituency', 'code'], how='outer')

    outcome_data = pd.merge(outcome_data, bes_support19_constit
                            , right_on=['code']
                            , left_on=['code'], how='outer')

    outcome_data['constituency_lower'] = outcome_data['constituency'].str.lower()
    outcome_data = pd.merge(outcome_data, mrpdata_constit
                            , left_on=['constituency_lower']
                            # , right_on = ['constituency_lower']
                            , right_index=True
                            , how='outer')

    outcome_data = pd.merge(outcome_data, mrpdec
                            , left_on=['constituency', 'code']
                            , right_on=['constituency', 'code']
                            , how='outer')



    #########
    # Set constituency code numbers

    outcome_data['constituency_lower'] = outcome_data['constituency'].str.lower()
    outcome_data.set_index(['code', 'constituency'], inplace=True)

    # add code num to outcome data
    temp = constit_code_tbl
    temp.insert(2, "constituency_lower", temp['constituency'].str.lower())
    temp = temp.groupby(['code_num', 'constituency_lower']).size().reset_index()

    outcome_data = pd.merge(temp, outcome_data, left_on='constituency_lower', right_on='constituency_lower',
                            how='inner')

    ## SAVE DATA
    outcome_data.to_pickle(data_path + 'projection_data/outcome_data.pkl')





if __name__ == '__main__':
    print('Recoding covariate data...')
    constit_code_tbl = recode_covar_data()

    print('Recoding outcome data...')
    recode_outcome_data(constit_code_tbl = constit_code_tbl)