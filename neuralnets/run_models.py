from __future__ import division, print_function
from functools import partial
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
import tensorflow as tf

from neuralnets.features import Features
import neuralnets.radial
from neuralnets.radial import build_radial_net
from neuralnets.train import eval_network, train_network
from neuralnets.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


import matplotlib.pyplot as plt
import seaborn as sns





############## START CODE ################

#### set params
data_path = '~/Documents/LibDems/data/projection_data'


#######################################
#### Import data

## import covariate data
X_latlong = pd.read_pickle(data_path + '/X_latlong.pkl')
X_scores = pd.read_pickle(data_path + '/X_scores.pkl')
X_demo = pd.read_pickle(data_path + '/X_demo.pkl')

X_constit = pd.get_dummies(X_demo['code_num']
                             , prefix_sep="__"
                             , columns=['code_num']
                            )
mean_mat = X_constit.div(X_constit.sum(axis=0), axis=1)


## import outcome data
outcome_data = pd.read_pickle(data_path + '/outcome_data.pkl')
outcome_data['constituency_lower'] = outcome_data['constituency'].str.lower()
outcome_data.set_index(['code', 'constituency'], inplace = True)

# add code num to outcome data
temp = X_demo[['constituency', 'code_num']]
temp.insert(2,"constituency_lower", temp['constituency'].str.lower())
temp = temp.groupby(['code_num','constituency_lower']).size().reset_index()

outcome_data = pd.merge(temp, outcome_data, left_on = 'constituency_lower', right_on = 'constituency_lower', how = 'inner')

## import landmark data
r['load']('~/github/libdems/turnout/turnout_ldmks_300_data.RData')
turnout_ldmks = r.turnout_ldmks
landmark_ind = turnout_ldmks[0].astype(int)



# X_latlong_grouped = X_latlong.groupby('code_num')
# X_latlong_np = []
# for name, group in X_latlong_grouped:
#     X_latlong_np.append(group.values[:,2:])
# X_latlong_np


#### MAKE FEATURES
bags = X_latlong.to_numpy()[:,2:]
n_pts = X_latlong.groupby('code_num')['code_num'].count().to_numpy()

feats = Features(bags = bags, n_pts=n_pts, stack=True, copy=False, bare=False, y = outcome_data['pct_turnout_ge2017'])


args = {'reg_out':0
        , 'reg_out_bias':0
        , 'scale_reg_by_n':False
        , 'dtype_double':False
        , 'type': 'radial'
        , 'init_from_ridge':False
        , 'landmarks':None
        }


def make_network(args, train):
    kw = {'in_dim': train.dim
        , 'reg_out': args['reg_out']
        , 'reg_out_bias': args['reg_out_bias']
        , 'scale_reg_by_n': args['scale_reg_by_n']
        , 'dtype': tf.float64 if args['dtype_double'] else tf.float32
          }

    kw['bw'] = bw = np.sqrt(get_median_sqdist(train) / 2) #* args['bw_scale']
    need_means = args['init_from_ridge']
    need_all_feats = False

    # get landmarks
    #kw['landmarks'] = landmarks = pick_landmarks(args, train)
    kw['landmarks'] = landmarks = args['landmarks']

    get_f = partial(rbf_kernel, Y=landmarks, gamma=1 / (2 * bw**2))  #creating a partial function call of rbf_kernel with Y and gamma fixed
    if need_all_feats:
        train.make_stacked()
        train_feats = Features(get_f(train.stacked_features), train.n_pts)
        if need_means:
            train_means = train_feats.means().stacked_features
    elif need_means:
        train_means = np.empty((len(train), landmarks.shape[0]))
        for i, bag in enumerate(train):
            train_means[i] = get_f(bag).mean(axis=0)

    # initialize weights with ridge regression
    if args['init_from_ridge']:
        print("Fitting ridge...")
        ridge = Ridge(alpha=2 * args['reg_out'] * len(train),
                      solver='saga', tol=0.1, max_iter=500)
        # Ridge mins  ||y - X w - b||^2 + alpha ||w||^2
        # we min  1/n ||y - X w - b||^2 + reg_out/2 ||w||^2
        # (at least in radial...shrinkage a bit different, but w/e)
        ridge.fit(train_means, train.y)
        init_mse = mean_squared_error(train.y, ridge.predict(train_means))
        print("Ridge MSE: {:.4f}".format(init_mse))
        kw['init_out'] = ridge.coef_
        kw['init_out_bias'] = ridge.intercept_

    return build_radial_net(**kw)


make_network(args, feats)

