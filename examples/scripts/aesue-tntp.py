'''
Install isuelogit using pip install -q git+https://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@github.com/pabloguarda/isuelogit.git
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import random
import numpy as np
import pandas as pd
import tensorflow as tf
import isuelogit as isl
from sklearn import preprocessing

from src.aesuelogit.visualizations import plot_predictive_performance
from src.aesuelogit.models import UtilityParameters, GISUELOGIT, AETSUELOGIT, NGD, BPRParameters, ODParameters
from src.aesuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data
from src.aesuelogit.descriptive_statistics import mse, btcg_mse, mnrmse

# Seed for reproducibility
_SEED = 2022

np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# Experiments
list_models = ['equilibrium','isuelogit', 'ode', 'odlue', 'odlulce']

# run_model = dict.fromkeys(list_models,True)
run_model = dict.fromkeys(list_models, False)

# run_model['equilibrium'] = True
# run_model['isuelogit'] = True
# run_model['ode'] = True
# run_model['odlue'] = True

# It will be not included for IATBR abstract because I need to fix issue with Fresno
run_model['odlulce'] = True

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

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

# REad synthethic data which was generated under the assumption of path sets of size 2.
df = pd.read_csv(
    main_dir + '/output/network-data/' + tntp_network.key + '/links/' + tntp_network.key + '-link-data.csv')

n_days = len(df.period.unique())
n_links = len(tntp_network.links)
n_hours = 1

features_Z = ['c', 's']
# features_Z = []

n_sparse_features = 0
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# features_sparse = []
features_Z.extend(features_sparse)

# Prepare the training and validation dataset.

# Add free flow travel times
df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_days)

# TODO: exclude test data from transform to avoid data leakage
# df[features_Z + ['traveltime'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['traveltime'] + ['tt_ff']])


# X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)

traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_days=n_days, n_hours=n_hours)
flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

Y = tf.concat([traveltime_data, flow_data], axis=3)
X = get_design_tensor(Z=df[features_Z], n_links=n_links, n_days=n_days, n_hours=n_hours)

X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), Y.numpy(), test_size=0.2, random_state=42)

X_train, X_test, Y_train, Y_test = [tf.constant(i) for i in [X_train, X_test, Y_train, Y_test]]


train_losses_dfs = {}
val_losses_dfs = {}

if run_model['equilibrium']:

    print('Suelogit equilibria')

    _EPOCHS = 200
    _BATCH_SIZE = 16  # None
    _LR = 1e-1
    _RELATIVE_GAP = 1e-5

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': -1, 'c': -6, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           # signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                           #        'bus_stops': '-', 'intersections': '-'},
                                           # trainables={'psc_factor': False, 'fixed_effect': False},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'c': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 1},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={1: tntp_network.q.flatten()},
                                 trainable=False)

    # q_historic = isl.networks.denseQ(isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.05).copy())

    q_historic = tntp_network.q

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


    model_0 = GISUELOGIT(
        key='model_0',
        # endogenous_flows=True,
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        # column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )
    # X_train.shape
    # Y_train.shape
    # Y_test.shape
    train_losses_dfs['model_0'], val_losses_dfs['model_0'] = model_0.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        loss_metric = mse,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'bpr': 0, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_0'], val_losses=val_losses_dfs['model_0'])

if run_model['isuelogit']:
    print('\n model 1: Benchmark of aesuelogit and isuelogit (utility only with link count and traveltime data)')

    _EPOCHS = 500
    _BATCH_SIZE = 16  # None
    _LR = 5e-1
    _RELATIVE_GAP = 1e-5

    # optimizer = NGD(learning_rate=_LR)

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'c': -6, 's': -3},
                                           # signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                           #        'bus_stops': '-', 'intersections': '-'},
                                           # trainables={'psc_factor': False, 'fixed_effect': False},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': True, 'c': True, 's': True},
                                           # trainables={'psc_factor': False, 'fixed_effect': False
                                           #     , 'tt': False, 'c': False, 's': False},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','c','s'])
    # utility_parameters.random_initializer((0, 0), ['tt', 'c', 's'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 1, 'beta': 1},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False))

    # q_historic = isl.networks.denseQ(isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.05).copy())

    q_historic = tntp_network.q

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 # initial_values=tntp_network.q.flatten(),
                                 # true_values=tntp_network.q.flatten(),
                                 initial_values=q_historic.flatten(),
                                 historic_values={1: q_historic.flatten()},
                                 # historic_values={1: tntp_network.q.flatten()},
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

    column_generator = ColumnGenerator(equilibrator=equilibrator,
                                       utility=utility_parameters,
                                       n_paths=0,
                                       ods_coverage=0.1,
                                       ods_sampling='sequential',
                                       # ods_sampling='demand',
                                       )


    model_1 = GISUELOGIT(
        key='model_1',
        # endogenous_flows=True,
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        # column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_losses_dfs['model_1'], val_losses_dfs['model_1'] = model_1.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_metric=mse,
        # loss_metric = mnrmse,
        loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_1'], val_losses=val_losses_dfs['model_1'],
                                xticks_spacing= 50)

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_1.theta.numpy())))}")
    print(f"alpha = {model_1.alpha: 0.2f}, beta  = {model_1.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_1.q - tntp_network.q.flatten())): 0.2f}")

if run_model['ode']:
    print('\nmodel 2: ODE (OD estimation with historic OD)')

    _EPOCHS = 200
    _BATCH_SIZE = 16  # None
    _LR = 1e-1
    _RELATIVE_GAP = 1e-5

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': -1, 'c': -6, 's': -3, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': False, 'c': False, 's': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    q_historic = isl.networks.denseQ(isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.05).copy())

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values=0.8 * q_historic.flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={1: q_historic.flatten()},
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




    model_2 = GISUELOGIT(
        key='model_2',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_2'], val_losses_dfs['model_2'] = model_2.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # loss_metric=mnrmse,
        loss_metric=mse,
        # generalization_error={'train': False, 'validation': True},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_2'], val_losses=val_losses_dfs['model_2'],
                                xticks_spacing= 20)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_2.theta.numpy())))}")
    print(f"alpha = {model_2.alpha: 0.2f}, beta  = {model_2.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_2.q - tntp_network.q.flatten())): 0.2f}")

if run_model['odlue']:
    print('\nmodel 3: ODLUE (OD + utility estimation with historic OD)')

    _EPOCHS = 400
    _BATCH_SIZE = 16  # None
    _LR = 1e-1
    _RELATIVE_GAP = 1e-5

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'c': -6, 's': -3},
                                           # signs={'tt': '-', 'c': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': True, 'c': True, 's': True},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','c','s'])

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    q_historic = isl.networks.denseQ(isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.1).copy())

    # q_historic = tntp_network.q

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=tntp_network.q.flatten(),
                                 initial_values=q_historic.flatten(),
                                 # initial_values=np.ones_like(tntp_network.q.flatten()),
                                 historic_values={1: q_historic.flatten()},
                                 # historic_values={1: tntp_network.q.flatten()},
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


    model_3 = GISUELOGIT(
        key='model_3',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_3'], val_losses_dfs['model_3'] = model_3.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # loss_metric=mnrmse,
        loss_metric=mse,
        # generalization_error={'train': False, 'validation': True},
        # loss_weights={'od': 10, 'theta': 0, 'tt': 1, 'flow': 10, 'eq_flow': 10},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_3'], val_losses=val_losses_dfs['model_3'],
                                xticks_spacing= 50)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_3.theta.numpy())))}")
    print(f"alpha = {model_3.alpha: 0.2f}, beta  = {model_3.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_3.q - tntp_network.q.flatten())): 0.2f}")
    
if run_model['odlulce']:

    # TODO: This model will be analyzed further. Estimating link cost parameters is unstable. A larger network may be used
    # to ensure identifiability

    print('\nmodel 4: ODLUE + link performance parameters with historic OD matrix')

    _EPOCHS = 500
    _BATCH_SIZE = 16  # None
    _LR = 5e-1
    _RELATIVE_GAP = 1e-5

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 1, 'beta': 1},
                                   # initial_values={'alpha': np.ones_like(tntp_network.links, dtype=np.float32),
                                   #                 'beta': 4 * np.ones_like(tntp_network.links, dtype=np.float32)},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':True},
                                   )

    # bpr_parameters.random_initializer((-1,1),['beta'])
    # bpr_parameters.random_initializer((-0.15, 0.15), ['alpha'])

    # bpr_parameters.initial_values

    q_historic = isl.networks.denseQ(
        isl.factory.random_disturbance_Q(tntp_network.Q, sd=np.mean(tntp_network.Q) * 0.1).copy())

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values= q_historic.flatten(),
                                 historic_values={1: q_historic.flatten()},
                                 # historic_values={1: tntp_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(tntp_network.links)},
                                           true_values={'tt': -1, 'c': -6, 's': -3},
                                           # signs={'tt': '-', 'c': '-', 's': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False
                                               , 'tt': True, 'c': True, 's': True},
                                           )

    # utility_parameters.random_initializer((-1,1),['tt','c','s'])

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


    model_4 = GISUELOGIT(
        key='model_4',
        network=tntp_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_4'], val_losses_dfs['model_4'] = model_4.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_metric=mse,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_4'], val_losses=val_losses_dfs['model_4'],
                                xticks_spacing = 20)

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_4.theta.numpy())))}")
    print(f"alpha = {np.mean(model_4.alpha): 0.2f}, beta  = {np.mean(model_4.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_4.q - tntp_network.q.flatten())): 0.2f}")