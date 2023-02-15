import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.ioff()

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
import seaborn as sns
from sklearn import preprocessing
from datetime import datetime

# Internal modules
from src.gisuelogit.visualizations import plot_predictive_performance, plot_heatmap_demands, plot_convergence_estimates, \
    plot_top_od_flows_periods, plot_utility_parameters_periods
from src.gisuelogit.models import UtilityParameters, GISUELOGIT, BPRParameters, ODParameters, compute_rr, \
    compute_insample_outofsample_error
from src.gisuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from src.gisuelogit.etl import get_design_tensor, get_y_tensor, add_period_id
from src.gisuelogit.descriptive_statistics import mse, btcg_mse, mnrmse

# Seed for reproducibility
_SEED = 2022
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

## Build network
network_name = 'SiouxFalls'
tntp_network = build_tntp_network(network_name=network_name)

## Read OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q=Q)

Q_historic = isl.factory.random_disturbance_Q(tntp_network.Q.copy(), sd=np.mean(tntp_network.Q) * 0.1)
#
# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

# REad synthethic data which was generated under the assumption of path sets of size 2.
df = pd.read_csv(
    main_dir + '/output/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

features_Z = ['tt_sd', 's']
# features_Z = []

n_sparse_features = 0
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# features_sparse = []
features_Z.extend(features_sparse)

# Prepare the training and validation dataset.

n_timepoints = len(df.period.unique())
n_links = len(tntp_network.links)

# Add free flow travel times
df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_timepoints)

# TODO: exclude test data from transform to avoid data leakage
# df[features_Z + ['traveltime'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['traveltime'] + ['tt_ff']])

# X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)
traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_timepoints = n_timepoints)
flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_timepoints = n_timepoints)

period_feature = 'hour'

# For splitting in training and testing set
df['hour'] = 0
df.loc[df['period']<= 50,'hour'] = 1
df = add_period_id(df, period_feature=period_feature)
# df = add_period_id(df, period_feature='period')

period_keys = df[[period_feature,'period_id']].drop_duplicates().reset_index().drop('index',axis =1).sort_values('hour')
print(period_keys)

Y = tf.concat([traveltime_data, flow_data], axis=2)
X = get_design_tensor(Z=df[features_Z + ['period_id']], n_links=n_links, n_timepoints = n_timepoints)

# X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), Y.numpy(), test_size=0.2, random_state=_SEED)
# X_train, X_test, Y_train, Y_test = [tf.constant(i) for i in [X_train, X_test, Y_train, Y_test]]

X_train, X_test, Y_train, Y_test = X, None, Y, None

# _EPOCHS = {'learning': 400, 'equilibrium': 0}
# _EPOCHS = {'learning': 100, 'equilibrium': 0}
_EPOCHS = {'learning': 30, 'equilibrium': 5}
_LR = 1e-1
_RELATIVE_GAP = 1e-8
_XTICKS_SPACING = 50
# To reduce variability in estimates of experiments, it is better to not use batches
# _BATCH_SIZE = None
_BATCH_SIZE = 16
_MOMENTUM_EQUILIBRIUM = 1

# loss_metric = mnrmse
_LOSS_METRIC = mse

_EPOCHS_PRINT_INTERVAL = 5

_LOSS_WEIGHTS ={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1,
                'ntrips': 0, 'prop_od': 0}

# Models
list_models = ['equilibrium', 'lue', 'ode', 'ode-nosuelogit', 'lpe',
               'odlue', 'odlulpe', 'odlulpe-no-equilibrium', 'tvodlulpe']

# run_model = dict.fromkeys(list_models,True)
run_model = dict.fromkeys(list_models, False)

# run_model['equilibrium'] = True
# run_model['lue'] = True
# run_model['ode'] = True
# run_model['ode-nosuelogit'] = True
# run_model['lpe'] = True
# run_model['odlue'] = True
# run_model['odlulpe'] = True
# run_model['odlulpe-no-equilibrium'] = True
run_model['tvodlulpe'] = True

train_results_dfs = {}
val_results_dfs = {}

if run_model['equilibrium']:

    # _RELATIVE_GAP = 1e-8

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    print('ISUELOGIT: Equilibrium')

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 1},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 trainable=False)

    len(tntp_network.q.flatten())

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    suelogit = GISUELOGIT(
        key='suelogit',
        # endogenous_flows=True,
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        # column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_results_dfs['suelogit'], val_results_dfs['suelogit'] = suelogit.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        # loss_metric = mse,
        loss_metric=_LOSS_METRIC,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval = _EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    train_results_estimates, train_results_losses = suelogit.split_results(results=train_results_dfs['suelogit'])
    val_results_estimates, val_results_losses = suelogit.split_results(results=val_results_dfs['suelogit'])

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=val_results_dfs['suelogit'],
                                xticks_spacing= 250)

    fig, ax = plot_convergence_estimates(
        estimates=train_results_losses.drop([0]).assign(
            relative_gap = np.abs(train_results_losses['relative_gap']))[['epoch','relative_gap']])

    ax.set_yscale('log')
    ax.set_ylabel('relative gap (log scale)')

    plt.show()

if run_model['ode-nosuelogit']:
    print('\n ODE: OD estimation without equilibrium term')

    # To show the problems of no considering equilibrium term, we will reduce the coverage

    # _RELATIVE_GAP = 1e-6

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values= isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    ode_nosuelogit = GISUELOGIT(
        key='ode-nosuelogit',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        # n_periods=len(np.unique(X_train[:, :, -1].numpy().flatten()))
    )

    train_results_dfs['ode-nosuelogit'], val_results_dfs['ode-nosuelogit'] = ode_nosuelogit.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=_LOSS_METRIC,
        # generalization_error={'train': False, 'validation': True},
        #loss_weights= dict(_LOSS_WEIGHTS, od=1),
        loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 0},
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    # plot_histograms_flow(true = df.true_counts.values.flatten(),
    #                      predicted = ode_nosuelogit.flows().numpy().flatten(),
    #                      # observed = Y_train[:,:,1].numpy().flatten()
    #                      )

    print(mse(ode_nosuelogit.flows(),df.true_counts[0:tntp_network.get_n_links()].values).numpy())
    print(mse(ode_nosuelogit.flows(), Y_train[:, :, 1]).numpy())

    plt.show()
    plot_predictive_performance(train_losses=train_results_dfs['ode-nosuelogit'],
                                val_losses=val_results_dfs['ode-nosuelogit'],
                                xticks_spacing= 250)

    plt.show()

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic, 'estimated': tf.sparse.to_dense(ode_nosuelogit.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(ode_nosuelogit.theta.numpy())))}")
    print(f"alpha = {ode_nosuelogit.alpha: 0.2f}, beta  = {ode_nosuelogit.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(ode_nosuelogit.q - tntp_network.q.flatten())): 0.2f}")

if run_model['ode']:
    print('\n ODE: OD estimation with historic OD')

    # _RELATIVE_GAP = 1e-6

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values= isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )

    ode = GISUELOGIT(
        key='ode',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        # n_periods=len(np.unique(X_train[:, :, -1].numpy().flatten()))
    )

    train_results_dfs['ode'], val_results_dfs['ode'] = ode.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=_LOSS_METRIC,
        # generalization_error={'train': False, 'validation': True},
        loss_weights=dict(_LOSS_WEIGHTS, od=1),
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    # plot_histograms_flow(true = df.true_counts.values.flatten(),
    #                      predicted = ode.flows().numpy().flatten(),
    #                      # observed = Y_train[:,:,1].numpy().flatten()
    #                      )

    # plot_histograms_flow(true = df.true_counts[0:tntp_network.get_n_links()].values-ode.flows().numpy(),
    #                      predicted = (Y_train[:,:,1]-ode.flows().numpy().flatten()).numpy().flatten(),
    #                      # observed = Y_train[:,:,1].numpy().flatten()
    #                      )

    plot_predictive_performance(train_losses=train_results_dfs['ode'], val_losses=val_results_dfs['ode'],
                                xticks_spacing= 250)

    plt.show()

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic, 'estimated': tf.sparse.to_dense(ode.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(ode.theta.numpy())))}")
    print(f"alpha = {ode.alpha: 0.2f}, beta  = {ode.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(ode.q - tntp_network.q.flatten())): 0.2f}")

if run_model['lue']:
    print('\n model 1: Benchmark of gisuelogit and isuelogit (utility only with link count and traveltime data)')

    # optimizer = NGD(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=_LR)

    # Initialize again the optimizer as there are some decay parameters that are stored in the object and that will
    # affect the next model estimation
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # _RELATIVE_GAP = 1e-7

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': True, 'tt_sd': True, 's': True},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])
    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 1},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False))

    od_parameters = ODParameters(key='od',
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 # initial_values=tntp_network.q.flatten(),
                                 # true_values=tntp_network.q.flatten(),
                                 initial_values=q_historic.flatten(),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 # historic_values={0: tntp_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    lue = GISUELOGIT(
        key='lue',
        # endogenous_flows=True,
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        # column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        # n_periods=len(np.unique(X_train[:, :, -1].numpy().flatten()))
    )

    train_results_dfs['lue'], val_results_dfs['lue'] = lue.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=_LOSS_METRIC,
        loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        momentum_equilibrium = _MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    train_results_estimates, train_results_losses = lue.split_results(results=train_results_dfs['lue'])
    val_results_estimates, val_results_losses = lue.split_results(results=val_results_dfs['lue'])

    plot_predictive_performance(train_losses=train_results_losses, val_losses=val_results_losses,
                                xticks_spacing=250)

    plt.show()

    plot_convergence_estimates(estimates=train_results_estimates.\
                               assign(rr = train_results_estimates['tt_sd']/train_results_estimates['tt'])[['epoch','rr']],
                               true_values={'rr':lue.utility.true_values['tt']/lue.utility.true_values['tt_sd']})
    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(lue.theta.numpy())))}")
    print(f"alpha = {lue.alpha: 0.2f}, beta  = {lue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(lue.q - tntp_network.q.flatten())): 0.2f}")

if run_model['lpe']:
    print('\nLPE: link performance estimation')

    # _RELATIVE_GAP = 1e-9

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': -1, 'tt_sd': -1.3, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'tt_sd': False, 's': False},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], True),
                                   # trainables={'alpha': False, 'beta': True}
                                   )

    bpr_parameters.random_initializer((0, 0), ['alpha', 'beta'])

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values= isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 # historic_values={0: tntp_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    lpe = GISUELOGIT(
        key='lpe',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['lpe'], val_results_dfs['lpe'] = lpe.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=_LOSS_METRIC,
        # loss_metric=mnrmse,
        # generalization_error={'train': False, 'validation': True},
        loss_weights= dict(_LOSS_WEIGHTS),
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    train_results_estimates, train_results_losses = lpe.split_results(results=train_results_dfs['lpe'])
    val_results_estimates, val_results_losses = lpe.split_results(results=val_results_dfs['lpe'])

    plot_predictive_performance(train_losses=train_results_losses, val_losses=val_results_dfs['lpe'],
                                xticks_spacing=250)
    plt.show()

    plot_convergence_estimates(estimates=train_results_estimates[['epoch', 'alpha', 'beta']],
                               true_values=lpe.bpr.true_values)
    plt.show()

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic, 'estimated': tf.sparse.to_dense(lpe.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(lpe.theta.numpy())))}, "
          f"rr = {train_results_estimates.eval('tt_sd/tt').values[-1]:0.2f}")
    print(f"alpha = {lpe.alpha: 0.2f}, beta  = {lpe.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(lpe.q - tntp_network.q.flatten())): 0.2f}")

if run_model['odlue']:
    print('\nODLUE: OD + utility estimation with historic OD')

    # _RELATIVE_GAP = 1e-9

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           # signs={'tt': '-', 'tt_sd': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': True, 'tt_sd': True, 's': True},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])
    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values= isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 # historic_values={0: tntp_network.q.flatten()},
                                 total_trips={0: np.sum(tntp_network.Q)},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    odlue = GISUELOGIT(
        key='odlue',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlue'], val_results_dfs['odlue'] = odlue.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=_LOSS_METRIC,
        # loss_metric=mse,
        # generalization_error={'train': False, 'validation': True},
        # loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights=dict(_LOSS_WEIGHTS),
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    
    train_results_estimates, train_results_losses = odlue.split_results(results=train_results_dfs['odlue'])
    val_results_estimates, val_results_losses = odlue.split_results(results=val_results_dfs['odlue'])

    plot_predictive_performance(train_losses= train_results_losses, val_losses=val_results_dfs['odlue'],
                                xticks_spacing = 250)

    plot_convergence_estimates(estimates=train_results_estimates.\
                               assign(rr = train_results_estimates['tt_sd']/train_results_estimates['tt'])[['epoch','rr']],
                               true_values={'rr':odlue.utility.true_values['tt']/odlue.utility.true_values['tt_sd']})
    plt.show()

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic, 'estimated': tf.sparse.to_dense(odlue.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlue.theta.numpy())))}, "
          f"rr = {train_results_estimates.eval('tt_sd/tt').values[-1]:0.2f}")
    print(f"alpha = {odlue.alpha: 0.2f}, beta  = {odlue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlue.q - tntp_network.q.flatten())): 0.2f}")
    
if run_model['odlulpe']:
    # _RELATIVE_GAP = 1e-10

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 2},
                                   # initial_values={'alpha': np.ones_like(tntp_network.links, dtype=np.float32),
                                   #                 'beta': 1 * np.ones_like(tntp_network.links, dtype=np.float32)},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':True},
                                   )

    bpr_parameters.random_initializer((0, 0), ['alpha', 'beta'])

    # bpr_parameters.random_initializer((-1,1),['beta'])
    # bpr_parameters.random_initializer((-0.15, 0.15), ['alpha'])

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values=isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 # historic_values={0: tntp_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           # signs={'tt': '-', 'tt_sd': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True
                                               , 'tt': True, 'tt_sd': True, 's': True},
                                           )

    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])
    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    odlulpe = GISUELOGIT(
        key='odlulpe',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlulpe'], val_results_dfs['odlulpe'] = odlulpe.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)


    train_results_estimates, train_results_losses = odlulpe.split_results(results=train_results_dfs['odlulpe'])
    val_results_estimates, val_results_losses = odlulpe.split_results(results=val_results_dfs['odlulpe'])

    print(compute_insample_outofsample_error(Y = Y_train,
                                       true_counts=df.true_counts.values[0:tntp_network.get_n_links()],
                                       true_traveltimes=df.true_traveltime.values[0:tntp_network.get_n_links()],
                                       model = odlulpe))

    plot_predictive_performance(train_losses= train_results_losses, val_losses=val_results_dfs['odlulpe'],
                                xticks_spacing = 250)

    plot_convergence_estimates(estimates=train_results_estimates.\
                               assign(rr = train_results_estimates['tt_sd']/train_results_estimates['tt'])[['epoch','rr']],
                               true_values={'rr':odlulpe.utility.true_values['tt_sd']/odlulpe.utility.true_values['tt']})

    plot_convergence_estimates(estimates=train_results_estimates[['epoch','alpha','beta']],
                               true_values=odlulpe.bpr.true_values)

    # sns.displot(pd.melt(pd.DataFrame({'alpha':odlulpe.alpha, 'beta': odlulpe.beta}), var_name = 'parameters'),
    #             x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)
    #
    # sns.displot(pd.DataFrame({'fixed_effect':np.array(odlulpe.fixed_effect)}),
    #             x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic, 'estimated': tf.sparse.to_dense(odlulpe.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe.theta.numpy())))}, "
          f"rr = {train_results_estimates.eval('tt_sd/tt').values[-1]:0.2f}")
    print(f"alpha = {np.mean(odlulpe.alpha): 0.2f}, beta  = {np.mean(odlulpe.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlulpe.q - tntp_network.q.flatten())): 0.2f}")

if run_model['odlulpe-no-equilibrium']:
    # _RELATIVE_GAP = 1e-10

    print('\nODLULPE with no equilibrium component')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 2},
                                   # initial_values={'alpha': np.ones_like(tntp_network.links, dtype=np.float32),
                                   #                 'beta': 1 * np.ones_like(tntp_network.links, dtype=np.float32)},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':True},
                                   )

    bpr_parameters.random_initializer((0, 0), ['alpha', 'beta'])

    # bpr_parameters.random_initializer((-1,1),['beta'])
    # bpr_parameters.random_initializer((-0.15, 0.15), ['alpha'])

    od_parameters = ODParameters(key='od',
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values=isl.networks.denseQ(Q_historic).flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={0: isl.networks.denseQ(Q_historic).flatten()},
                                 # historic_values={0: tntp_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           # signs={'tt': '-', 'tt_sd': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True
                                               , 'tt': True, 'tt_sd': True, 's': True},
                                           )

    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])
    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])

    equilibrator = Equilibrator(
        network=tntp_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    odlulpe_no_equilibrium = GISUELOGIT(
        key='odlulpe-no-equilibrium',
        network=tntp_network,
        dtype=tf.float64,
        endogenous_flows=False,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlulpe-no-equilibrium'], val_results_dfs['odlulpe-no-equilibrium'] \
        = odlulpe_no_equilibrium.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights= dict(_LOSS_WEIGHTS, eq_flow = 0),
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS
    )

    train_results_estimates, train_results_losses \
        = odlulpe_no_equilibrium.split_results(results=train_results_dfs['odlulpe-no-equilibrium'])
    val_results_estimates, val_results_losses \
        = odlulpe_no_equilibrium.split_results(results=val_results_dfs['odlulpe-no-equilibrium'])

    print(compute_insample_outofsample_error(Y=Y_train,
                                             true_counts=df.true_counts.values[0:tntp_network.get_n_links()],
                                             true_traveltimes=df.true_traveltime.values[0:tntp_network.get_n_links()],
                                             model=odlulpe_no_equilibrium))

    plot_predictive_performance(train_losses= train_results_losses,
                                val_losses =val_results_dfs['odlulpe-no-equilibrium'],
                                xticks_spacing = 250)

    plot_convergence_estimates(estimates=
                               train_results_estimates.assign(
                                   rr = train_results_estimates['tt_sd']/train_results_estimates['tt'])[['epoch','rr']],
                               true_values={'rr':odlulpe_no_equilibrium.utility.true_values['tt_sd']
                                                 /odlulpe_no_equilibrium.utility.true_values['tt']})

    plot_convergence_estimates(estimates=train_results_estimates[['epoch','alpha','beta']],
                               true_values=odlulpe_no_equilibrium.bpr.true_values)

    # sns.displot(pd.melt(pd.DataFrame({'alpha':odlulpe_no_equilibrium.alpha, 'beta': odlulpe_no_equilibrium.beta}), var_name = 'parameters'),
    #             x="value", hue="parameters", multiple="stack", kind="kde", alpha = 0.8)
    #
    # sns.displot(pd.DataFrame({'fixed_effect':np.array(odlulpe_no_equilibrium.fixed_effect)}),
    #             x="fixed_effect", multiple="stack", kind="kde", alpha = 0.8)

    Qs = {'true': tntp_network.OD.Q_true, 'historic': Q_historic,
          'estimated': tf.sparse.to_dense(odlulpe_no_equilibrium.Q).numpy()}

    plot_heatmap_demands(Qs=Qs, vmin=np.min(Qs['true']), vmax=np.max(Qs['true']), subplots_dims=(1, 3), figsize=(12, 4))

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe_no_equilibrium.theta.numpy())))}, "
          f"rr = {train_results_estimates.eval('tt_sd/tt').values[-1]:0.2f}")
    print(f"alpha = {np.mean(odlulpe_no_equilibrium.alpha): 0.2f}, "
          f"beta  = {np.mean(odlulpe_no_equilibrium.beta): 0.2f}")
    print(f"Avg abs diff between observed and estimated OD: "
          f"{np.mean(np.abs(odlulpe_no_equilibrium.q - tntp_network.q.flatten())): 0.2f}")

if run_model['tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           initial_values={'tt': 0, 'tt_sd': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'tt_sd': -1.3, 's': -3},
                                           trainables={'psc_factor': False, 'fixed_effect': True
                                               , 'tt': True, 'tt_sd': True, 's': True},
                                           time_varying=True,
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','tt_sd','s'])
    utility_parameters.random_initializer((0, 0), ['tt', 'tt_sd', 's'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15 * np.ones_like(tntp_network.links, dtype=np.float32),
                                                   'beta': 4 * np.ones_like(tntp_network.links, dtype=np.float32)},
                                   trainables={'alpha': True, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={0: tntp_network.q.flatten()},
                                 total_trips = {0: np.sum(tntp_network.Q)},
                                 time_varying=True,
                                 trainable=True)

    tvodlulpe = GISUELOGIT(
        key='tvodlulpe',
        network=tntp_network,
        dtype=tf.float64,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
        n_periods=len(np.unique(X_train[:, :, -1].numpy().flatten()))
    )

    train_results_dfs['tvodlulpe'], val_results_dfs['tvodlulpe'] = tvodlulpe.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights=_LOSS_WEIGHTS,
        loss_metric=_LOSS_METRIC,
        momentum_equilibrium=_MOMENTUM_EQUILIBRIUM,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # Plot heatmap with flows of top od pairs
    plot_top_od_flows_periods(tvodlulpe, period_keys = period_keys, period_feature='hour', top_k=20,
                              historic_od=tntp_network.q.flatten())

    plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=val_results_dfs['tvodlulpe'],
                                xticks_spacing=_XTICKS_SPACING)

    plot_convergence_estimates(estimates=train_results_dfs['tvodlulpe'][['epoch', 'alpha', 'beta']],
                               xticks_spacing=_XTICKS_SPACING)

    sns.displot(pd.melt(pd.DataFrame({'alpha': tvodlulpe.alpha, 'beta': tvodlulpe.beta}), var_name='parameters'),
                x="value", hue="parameters", multiple="stack", kind="kde", alpha=0.8)

    plt.show()

    # Compute utility parameters over time (heatmap) and value of travel time reliability (lineplot)
    theta_df = plot_utility_parameters_periods(tvodlulpe, period_keys = period_keys, period_feature='hour')

    plt.show()

    rr_df = theta_df.apply(compute_rr, axis=1).reset_index().rename(columns={'index': 'hour', 0: 'rr'})

    sns.lineplot(data=rr_df, x='hour', y="rr")

    plt.show()

    sns.displot(pd.DataFrame({'fixed_effect': np.array(tvodlulpe.fixed_effect)}),
                x="fixed_effect", multiple="stack", kind="kde", alpha=0.8)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(tvodlulpe.theta.numpy())))}")
    print(f"alpha = {np.mean(tvodlulpe.alpha): 0.2f}, beta  = {np.mean(tvodlulpe.beta): 0.2f}")
    print(
        f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(tvodlulpe.q - tntp_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(tntp_network.q.flatten())): 0.2f}")

# Write predictions
predictions = pd.DataFrame({'link_key': list(tntp_network.links_keys) * Y_train.shape[0],
                            'observed_traveltime': Y_train[:, :, 0].numpy().flatten(),
                            'observed_flow': Y_train[:, :, 1].numpy().flatten()})

predictions['period'] = df.period

for model in [lue,odlue,odlulpe,odlulpe_no_equilibrium]:

    model = odlulpe_no_equilibrium

    predicted_flows = model.flows()
    predicted_traveltimes = model.traveltimes()

    predictions['predicted_traveltime_' + model.key] = np.tile(predicted_traveltimes, (Y_train.shape[0], 1)).flatten()
    predictions['predicted_flow_' + model.key] = np.tile(predicted_flows, (Y_train.shape[0], 1)).flatten()

predictions.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_predictions_{network_name}.csv")

# Write csv file with estimation results
train_results_df, val_results_df \
    = map(lambda x: pd.concat([results.assign(model = model)[['model'] + list(results.columns)]
                               for model, results in x.items()],axis = 0), [train_results_dfs, val_results_dfs])

train_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_train_results_{network_name}.csv")
val_results_df.to_csv(f"./output/tables/{datetime.now().strftime('%y%m%d%H%M%S')}_validation_results_{network_name}.csv")

## Summary of model parameters

models = [lue,odlue,odlulpe]
results = pd.DataFrame({'parameter': [], 'model': []})

for model in models:
    results = results.append(pd.DataFrame(
        {'parameter': ['tt'] + features_Z +
                      ['rr'] +
                      ['fixed_effect_mean', 'fixed_effect_std',
                       'alpha_mean', 'alpha_std',
                       'beta_mean', 'beta_std',
                       'od_mean', 'od_std',],
         'values': list(np.mean(model.theta.numpy(),axis =0))  +
                   [float(model.get_parameters_estimates().eval('tt_sd/tt'))] +
                   [np.mean(model.fixed_effect),np.std(model.fixed_effect),
                    np.mean(model.alpha),np.std(model.alpha),
                    np.mean(model.beta),np.std(model.beta),
                    np.mean(model.q),np.std(model.q)]}).\
                             assign(model = model.key)
                             )

print(results.pivot_table(index = ['parameter'], columns = 'model', values = 'values', sort=False).round(4))

## Summary of models goodness of fit

results_losses = pd.DataFrame({})
loss_columns = ['loss_flow', 'loss_tt', 'loss_eq_flow', 'loss_total']

for i, model in enumerate(models):

    results_losses_model = model.split_results(train_results_dfs[model.key])[1].assign(model = model.key)
    results_losses_model = results_losses_model[results_losses_model.epoch == _EPOCHS['learning']].iloc[[0]]
    results_losses = results_losses.append(results_losses_model)

results_losses[loss_columns] = (results_losses[loss_columns]-1)*100

print(results_losses[['model'] + loss_columns].round(1))

## Plot of convergence toward true reliabiloty ratio (rr) across models

models = [lue,odlue,odlulpe]

train_estimates = {}
train_losses = {}

for model in models:
    train_estimates[model.key], train_losses[model.key] = model.split_results(results=train_results_dfs[model.key])

    train_estimates[model.key]['model'] = model.key

train_estimates_df = pd.concat(train_estimates.values())

train_estimates_df['rr'] = train_estimates_df['tt_sd']/train_estimates_df['tt']

estimates = train_estimates_df[['epoch','model','rr']].reset_index().drop('index',axis = 1)
estimates = estimates[estimates.epoch != 0]

fig, ax = plt.subplots(nrows=1, ncols=1)

g = sns.lineplot(data=estimates, x='epoch', hue='model', y='rr')

ax.hlines(y=compute_rr(utility_parameters.true_values), xmin=estimates['epoch'].min(), xmax=estimates['epoch'].max(), linestyle='--', label = 'truth')

ax.set_ylabel('reliability ratio')

plt.ylim(ymin=0)

plt.show()
