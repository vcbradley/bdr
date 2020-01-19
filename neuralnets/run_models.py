from __future__ import division, print_function
from functools import partial
import os
import sys

import numpy as np
import pandas as pd
import importlib as imp
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state
import tensorflow as tf

from neuralnets.features import Features
from neuralnets.base import Network
from neuralnets.radial import build_radial_net
from neuralnets.train import eval_network, train_network
from neuralnets.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()


import matplotlib.pyplot as plt
import seaborn as sns


# imp.reload(Features)
# imp.reload(radial)
# imp.reload(neuralnets.train)
#


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




#### MAKE FEATURES
bags = X_latlong.to_numpy()[:,2:]
n_pts = X_latlong.groupby('code_num')['code_num'].count().to_numpy()
# feats = Features(bags = bags, n_pts=n_pts, stack=True, copy=False, bare=False, y = outcome_data['pct_turnout_ge2017'])
#
# X_latlong_grouped = X_latlong.groupby('code_num').apply(pd.Series.tolist).tolist()
#
# np.concatenate(X_latlong_grouped)
#
# X_latlong_np = []
# for i,v in enumerate(X_latlong_grouped): print(i):
#     X_latlong_np.append(v[:,2:])
# X_latlong_np
# feats = Features(bags = X_latlong_np, stack=False, copy=False, bare=False, y = outcome_data['pct_turnout_ge2017'])

i = X_latlong['code_num'].values
ukeys, slices = np.unique(i, True)
X_latlong_array = np.split(X_latlong.iloc[:, 2:].values, slices[1:])

feats = Features(bags = X_latlong_array, copy=False, bare=False, y = outcome_data['pct_turnout_ge2017'])
feats.make_stacked()
feats.stacked


args = {'reg_out':0
        , 'reg_out_bias':0
        , 'scale_reg_by_n':False
        , 'dtype_double':False
        , 'type': 'radial'
        , 'init_from_ridge':False
        , 'landmarks':feats.stacked_features[landmark_ind]
        , 'opt_landmarks': True

        , 'optimizer':'adam'
        , 'out_dir':'/users/valeriebradley/github/bdr/neuralnets/results/'
        , 'batch_pts':np.inf
        , 'batch_bags':30
        , 'eval_batch_pts':np.inf
        , 'eval_batch_bags':100
        , 'max_epochs':1000
        , 'first_early_stop_epoch':1000/3
        , 'learning_rate':0.01

        #, 'n_estop':50
        , 'test_size':0.2
        , 'trainval_size':None
        , 'val_size':0.1875
        , 'train_estop_size':None
        , 'estop_size':0.23
        ,'train_size':None
        , 'split_seed':np.random.randint(2**32)
        }


def _split_feats(args, feats, labels=None, groups=None):
    if groups is None:
        ss = ShuffleSplit
    else:
        ss = GroupShuffleSplit

    rs = check_random_state(args['split_seed'])
    test_splitter = ss(
        1, train_size=args['trainval_size'], test_size=args['test_size'],
        random_state=rs)
    (trainval, test), = test_splitter.split(feats, None, None)

    val_splitter = ss(
        1, train_size=args['train_estop_size'], test_size=args['val_size'],
        random_state=rs)
    X_v = feats[trainval]
    y_v = None if labels is None else labels[trainval]
    g_v = None if groups is None else groups[trainval]
    (train_estop, val), = val_splitter.split(X_v, y_v, g_v)

    estop_splitter = ss(
        1, train_size=args['train_size'], test_size=args['estop_size'],
        random_state=rs)
    X = X_v[train_estop]
    y = None if labels is None else y_v[train_estop]
    g = None if groups is None else g_v[train_estop]
    (train, estop), = estop_splitter.split(X, y, g)
    return X[train], X[estop], X_v[val], feats[test]


train, estop, val, test = _split_feats(args, feats)



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
    kw['opt_landmarks'] = args['opt_landmarks']

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

net = make_network(args, train)



def train_net(sess, args, net, train, val):
    optimizer = {
        'adam': tf.compat.v1.train.AdamOptimizer,
        'sgd': tf.compat.v1.train.GradientDescentOptimizer,
    }[args['optimizer']]
    train_network(sess, net, train, val,
                  os.path.join(args['out_dir'], 'checkpoints/model'),
                  batch_pts=args['batch_pts'], batch_bags=args['batch_bags'],
                  eval_batch_pts=args['eval_batch_pts'],
                  eval_batch_bags=args['eval_batch_bags'],
                  max_epochs=args['max_epochs'],
                  first_early_stop_epoch=args['first_early_stop_epoch'],
                  optimizer=optimizer,
                  lr=args['learning_rate'])



with tf_session(n_cpus=4) as sess:
    train_net(sess, args, net, train, estop)
