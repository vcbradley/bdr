#!/usr/bin/env python3

from __future__ import division, print_function
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from sklearn.utils import check_random_state
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf

from neuralnets.bdr.features import Features
from neuralnets.bdr.networks.radial import build_simple_rbf, build_spatsep_rbf
from neuralnets.bdr.train import eval_network, train_network
from neuralnets.bdr.utils import get_median_sqdist, tf_session

from rpy2.robjects import r, pandas2ri
pandas2ri.activate()



def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault('help', "Default %(default)s.")
        return g.add_argument(*args, **kwargs)
    return f



def _add_args(subparser):
    network = subparser.add_argument_group("Network parameters")
    n = get_adder(network)
    n('--type', '-n', required=True, choices=network_types)
    n('--opt-landmarks', action='store_true', default=False)
    n('--no-opt-landmarks', action='store_false', dest='opt_landmarks')
    n('--kmeans-landmarks', action='store_true', default=False)
    n('--landmark-seed', type=int, default=1)
    n('--bw-scale', type=float, default=1)
    n('--n-freqs', type=int, default=64)
    n('--reg-out', type=float, default=0)
    n('--reg-out-bias', type=float, default=0)
    n('--reg-obs-var', type=float, default=0)
    n('--reg-freqs', type=float, default=0)
    n('--scale-reg-by-n', action='store_true', default=False)
    n('--init-from-ridge', action='store_true', default=False)
    n('--init-prior-feat-var', type=float, default=1)
    n('--opt-prior-feat-var', action='store_true', default=True)
    n('--fix-prior-feat-var', action='store_false', dest='opt_prior_feat_var')
    n('--shrink-towards-mean', action='store_true', default=False)
    n('--use-empirical-cov', action='store_true', default=False)
    n('--empirical-cov-add-diag', type=float, default=0)
    n('--use-alpha-reg', action='store_true', default=False,
      help='Regularize the regression with an RKHS norm, instead of an L2 norm '
           'on the weights. Default %(default)s.')

    g = get_adder(network.add_mutually_exclusive_group())
    g('--use-cholesky', action='store_true', default=True)
    g('--no-cholesky', action='store_false', dest='use_cholesky')

    G = network.add_mutually_exclusive_group()
    g = get_adder(G)
    g('--init-obs-var', type=float, default=None)
    g('--init-obs-var-mult', type=float, default=1)

    G = network.add_mutually_exclusive_group()
    g = get_adder(G)
    g('--use-real-R', action='store_true', default=False)
    g('--use-rbf-R', action='store_false', dest='use_real_R')
    n('--init-prior-measure-var', type=float, default=1)
    n('--init-R-scale', type=float, default=1)
    n('--opt-R-scale', action='store_true', default=False)
    n('--rbf-R-bw-scale', type=float, default=np.sqrt(2),
      help="Scale the bandwidth of the rbf R (default sqrt(2)).")

    train = subparser.add_argument_group("Training parameters")
    t = get_adder(train)
    t('--max-epochs', type=int, default=1000)
    t('--first-early-stop-epoch', type=int, help="Default: MAX_EPOCHS / 3.")
    int_inf = lambda x: np.inf if x.lower() in {'inf', 'none'} else int(x)
    t('--batch-pts', type=int_inf, default='inf')
    t('--batch-bags', type=int_inf, default=30)
    t('--eval-batch-pts', type=int_inf, default='inf')
    t('--eval-batch-bags', type=int_inf, default=100)
    t('--learning-rate', '--lr', type=float, default=.01)
    t('--dtype-double', action='store_true', default=False)
    t('--dtype-single', action='store_false', dest='dtype_double')
    t('--optimizer', choices=['adam', 'sgd'], default='adam')

    io = subparser.add_argument_group("I/O parameters")
    i = get_adder(io)
    io.add_argument('out_dir')
    i('--n-cpus', type=int, default=min(8, multiprocessing.cpu_count()))


def make_parser(rest_of_args=_add_args):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="The dataset to run on")
    # Subparser chosen by the first argument of your parser
    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group('Data parameters')
        rest_of_args(subparser)
        return data, get_adder(data)

    chi2, d = add_subparser('chi2')
    d('--n-train', type=int, default=1000)
    d('--n-estop', type=int, default=500)
    d('--n-val',   type=int, default=500)
    d('--n-test',  type=int, default=1000)
    d('--dim', '-d', type=int, default=5)
    d('--data-seed', type=int, default=np.random.randint(2**32))
    d('--size-type', choices=['uniform', 'neg-binom', 'manual', 'special'],
      default='uniform', help='special for fixed bag size of 1000')
    d('--min-size', type=int, default=20)
    d('--max-size', type=int, default=100)
    d('--size-mean', type=int, default=20)
    d('--size-std', type=int, default=10)
    d('--min-df', type=float, default=4)
    d('--max-df', type=float, default=8)
    d('--noise-std', type=float, default=0)

    def add_split_args(g):
        a = g.add_argument
        a('--test-size', type=float, default=.2,
          help="Number or portion of overall data to use for testing "
               "(default %(default)s).")
        a('--trainval-size', type=float, default=None,
          help="Number or portion of overall data to use for training, "
               "early-stopping, and validation together "
               "(default: complement of --test-size).")
        a('--val-size', type=float, default=.1875,
          help="Number or portion of non-test data to use for validation "
               "(default %(default)s).")
        a('--train-estop-size', type=float, default=None,
          help="Number or portion of non-test data to use for training and "
               "early stopping together (default: complement of "
               "--val-size).")
        a('--estop-size', type=float, default=0.2308,
          help="Number or portion of train-estop data to use for "
               "early stopping (default %(default)s).")
        a('--train-size', type=float, default=None,
          help="Number or portion of train-estop data to use for training "
               "(default: complement of --estop-size).")
        a('--split-seed', type=int, default=np.random.randint(2**32),
          help="Seed for the split process (default: random).")

    faces, d = add_subparser('imdb_faces')
    add_split_args(faces)

    return parser


def parse_args(rest_of_args=_add_args):
    parser = make_parser(rest_of_args)
    args = parser.parse_args()
    check_output_dir(args.out_dir, parser, make_checkpoints=True)
    return args


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


###############################

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    parser.add_argument('--sum', dest='accumulate', action='store_const',
                        const=sum, default=max,
                        help='sum the integers (default: find the max)')

    args = parser.parse_args()

    data_path = '~/Documents/LibDems/data/projection_data'

    ## covariate data
    X_all = pd.read_pickle(data_path + '/X_all.pkl')

    ## outcome data
    outcome_data = pd.read_pickle(data_path + '/outcome_data.pkl')

    #### MAKE FEATURES
    i = X_all['code_num'].values
    ukeys, slices = np.unique(i, True)
    X_all_array = np.split(X_all, slices[1:])

    feats = Features(bags=X_all_array, copy=False, bare=False, y=outcome_data[args['outcome_name']])
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



plot_results(d)

#tf.saved_model.load(args['out_dir']+'/checkpoints')


d