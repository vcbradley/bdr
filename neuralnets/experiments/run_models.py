#!/usr/bin/env python3

from __future__ import division, print_function
from functools import partial
import os
os.chdir('./neuralnets')
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib as imp
from sklearn.linear_model import Ridge
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state
from sklearn import preprocessing as prep

from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

from neuralnets.bdr.features import Features
from neuralnets.bdr.networks.radial import build_simple_rbf, build_spatsep_rbf
from neuralnets.bdr.train import eval_network, train_network
from neuralnets.bdr.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()




def _split_feats(args, feats, labels=None, groups=None):
    if groups is None:
        ss = ShuffleSplit
    else:
        ss = GroupShuffleSplit

    rs = check_random_state(args['split_seed'])

    test_splitter = ss(1,
                       train_size=args['test_size'],
                       random_state=rs)
    (test_ind, train_estop_val_ind), = test_splitter.split(feats, None, None)

    X_v = feats[train_estop_val_ind]
    y_v = None if labels is None else labels[train_estop_val_ind]
    g_v = None if groups is None else groups[train_estop_val_ind]

    if args['val_size'] is not None:
        val_splitter = ss(1,
                          train_size=args['val_size'],
                          random_state=rs)
        (val_ind, train_estop_ind), = val_splitter.split(X_v, y_v, g_v)
        val = X_v[val_ind]
    else:
        val = None
        train_estop_ind = list(range(0,len(train_estop_val_ind)))


    estop_splitter = ss(1,
                        train_size=args['estop_size'],
                        random_state=rs)
    X = X_v[train_estop_ind]
    y = None if labels is None else y_v[train_estop_ind]
    g = None if groups is None else g_v[train_estop_ind]
    (estop_ind, train_ind), = estop_splitter.split(X, y, g)

    return X[train_ind], X[estop_ind], val, feats[test_ind]

def pick_landmarks(args, train):
    train.make_stacked()
    rs = check_random_state(args.landmark_seed)
    if args.kmeans_landmarks:
        kmeans = KMeans(n_clusters=args.n_landmarks, random_state=rs, n_jobs=1)
        kmeans.fit(train.stacked_features)
        return kmeans.cluster_centers_
    else:
        w = rs.choice(train.total_points, args.n_landmarks, replace=False)
        return train.stacked_features[w]


network_types = {
    'simple': build_simple_rbf,
    'spatial_sep':build_spatsep_rbf
}


def make_network(args, train):

    kw = {'in_dim':train.dim
        , 'reg_out': args['reg_out']
        , 'reg_out_bias': args['reg_out_bias']
        , 'scale_reg_by_n': args['scale_reg_by_n']
        , 'dtype': tf.float64 if args['dtype_double'] else tf.float32
        , 'outcome_type':args['outcome_type']
          }

    kw['bw'] = np.sqrt(get_median_sqdist(train) / 2)

    # get landmarks
    #kw['landmarks'] = landmarks = pick_landmarks(args, train)
    kw['landmarks'] = args['landmarks']
    kw['opt_landmarks'] = args['opt_landmarks']

    # if args['type'] != 'simple':
    #     raise Exception(args['type'] + ' not recognized type of network')

    if args['type'] != 'simple':
        kw['feat_types'] = args['feat_types']

    return network_types[args['type']](**kw)



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


###############################

if __name__ == '__main__':

    data_path = '~/Documents/LibDems/data/projection_data'

    ## covariate data
    X_all = pd.read_pickle(data_path + '/X_all.pkl')

    ## outcome data
    outcome_data = pd.read_pickle(data_path + '/outcome_data.pkl')

    #### MAKE FEATURES
    i = X_all['code_num'].values
    ukeys, slices = np.unique(i, True)
    X_all_array = np.split(X_all, slices[1:])

    feats = Features(bags=X_all_array, copy=False, bare=False, y=outcome_data['pct_turnout_ge2017'])
    feats.make_stacked()


    # split data
    train, estop, val, test = _split_feats(args, feats)

    # build network
    net = make_network(args, train)

    # initialize outcome vars
    d = {'args': args}

    # train and evaluate
    with tf_session(n_cpus=4) as sess:
        train_net(sess, args, net, train, estop)

        # save weights
        for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES):
            d[v.name] = v.eval()

        # whether we want to evaluate the variance as well (which we ignore)
        do_var = False

        for name, ds in [('train', train), ('estop', estop), ('test', test)]:
            print()
            preds = eval_network(sess, net, ds
                                 , batch_pts=args['eval_batch_pts']
                                 , batch_bags=args['eval_batch_bags']
                                 , do_var=do_var)
            if do_var:
                preds, preds_var = preds
                d[name + '_preds_var'] = preds_var

            d[name + '_y'] = y = ds.y
            d[name + '_preds'] = preds

            d[name + '_mse'] = mse = mean_squared_error(y, preds)
            print('{} MSE: {}'.format(name, mse))

            d[name + '_r2'] = r2 = r2_score(y, preds)
            print('{} R2: {}'.format(name, r2))

            if do_var:
                liks = stats.norm.pdf(y, preds, np.sqrt(preds_var))
                d[name + '_nll'] = nll = -np.log(liks).mean()
                print('{} NLL: {}'.format(name, nll))

                cdfs = stats.norm.cdf(y, preds, np.sqrt(preds_var))
                coverage = np.mean((cdfs > .025) & (cdfs < .975))
                d[name + '_coverage'] = coverage
                print('{} coverage at 95%: {:.1%}'.format(name, coverage))



def plot_results(d, save_file = None):
    x = np.linspace(min(d['test_preds']), max(d['test_preds']), 100)
    plt.plot(x, x, linestyle='--', color='black')

    plt.plot(d['train_y'], d['train_preds'], 'o', label='training set')
    plt.plot(d['test_y'], d['test_preds'], 'o', label = 'test set')
    plt.plot(d['estop_y'], d['estop_preds'], 'o', label = 'estop set')

    plt.title("Test set performance")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.legend()

    plt.show()

    if save_file is not None:
        save_file



plot_results(d)

#tf.saved_model.load(args['out_dir']+'/checkpoints')


d