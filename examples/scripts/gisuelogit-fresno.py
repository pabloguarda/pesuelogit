'''
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.ion()

import numpy as np
import pandas as pd
import random
# import cudf as pd
import tensorflow as tf
import isuelogit as isl
import glob

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)
isl.config.dirs['read_network_data'] = "input/network-data/fresno/"

# Internal modules
from src.aesuelogit.models import UtilityParameters, BPRParameters, ODParameters, GISUELOGIT, NGD
from src.aesuelogit.visualizations import plot_predictive_performance
from src.aesuelogit.networks import load_k_shortest_paths, read_paths, build_fresno_network, \
    Equilibrator, sparsify_OD, ColumnGenerator, read_OD
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, data_curation, temporal_split
from src.aesuelogit.descriptive_statistics import mse, btcg_mse, mnrmse

# Seed for reproducibility
_SEED = 2022
np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# To write estimation report and set seed for all algorithms involving some randomness
# estimation_reporter = isl.writer.Reporter(
#     folderpath=isl.config.dirs['output_folder'] + 'estimations/' + 'Fresno', seed=_SEED)

## Build Fresno network
fresno_network = build_fresno_network()

# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type != 'LWRLK']

## Read OD matrix
# TODO: option to specify path to read OD matrix
read_OD(network=fresno_network, sparse=True)

# Read paths
# read_paths(network=fresno_network, update_incidence_matrices=True, filename='paths-fresno.csv')
# read_paths(network=fresno_network, update_incidence_matrices=True, filename = 'paths-full-model-fresno.csv')

# For quick testing (do not need to read_paths before)
Q = fresno_network.load_OD(sparsify_OD(fresno_network.Q, prop_od_pairs=0.99))
load_k_shortest_paths(network=fresno_network, k=2, update_incidence_matrices=True)

## Read spatiotemporal data
folderpath = isl.config.dirs['read_network_data'] + 'links/spatiotemporal-data/'
df = pd.concat([pd.read_csv(file) for file in glob.glob(folderpath + "*fresno-link-data*")], axis=0)

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df = df[df['date'].dt.dayofweek.between(0, 4)]
# df = df[df['date'].dt.year == 2019]

df['period'] = df['date'].astype(str) + '-' + df['hour'].astype(str)
df['period'] = df.period.map(hash)

# df1 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2019-10-01-fresno-link-data.csv')
# df1['date'] = "2019-10-01"
# df1['period'] = 0
#
# df2 = pd.read_csv(main_dir + '/input/network-data/' + fresno_network.key + '/links/2020-10-06-fresno-link-data.csv')
# df2['date'] = "2020-10-06"
# df2['period'] = 1
#
# df = pd.concat([df1, df2], axis=0)

## Data curation

# df['tt_ff'] = np.tile([link.bpr.tf for link in fresno_network.links],len(df.date.unique())*len(df.hour.unique()))
# df['tt_ff'] = df['tf_inrix']

df['tt_ff'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_ref_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_ref_avg == 0),'tt_ff'] = float('nan')

df['tt_avg'] = np.where(df['link_type'] != 'LWRLK', 0,df['length']/df['speed_hist_avg'])
df.loc[(df.link_type == "LWRLK") & (df.speed_hist_avg == 0),'tt_avg'] = float('nan')

df = data_curation(df)

## Utility function

features_Z = ['speed_sd', 'median_inc', 'incidents', 'bus_stops', 'intersections']
# features_Z = ['speed_sd']
# features_Z = []

# utility_parameters.constant_initializer(0)

## Data processing

n_links = len(fresno_network.links)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df.date.dt.year
X, Y = {}, {}

# Select only dates used for previous paper
# df = df.query('date == "2019-10-01"  | date == "2020-10-06"')
# df = df.query('date == "2019-10-01"')
df = df.query('hour == 16')
# df = df.query('hour == 17')

print(df.query('year == 2019')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

print(df.query('year == 2020')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

# Normalization of features to range [0,1]

# (TODO: may enable normalization in get_design_tensor method. See if tensorflow have it)
# TODO: exclude test data from transform to avoid data leakage
# df[features_Z + ['tt_avg'] + ['tt_ff']] \
#     = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['tt_avg'] + ['tt_ff']])

# Set free flow travel times
# tt_ff_links = df.query('link_type == "LWRLK"').groupby('link_key')['tt_ff'].min()
tt_ff_links = df.groupby('link_key')['tt_ff'].min()
# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type == "LWRLK"]
for link in fresno_network.links:
    fresno_network.links_dict[link.key].performance_function.tf = float(tt_ff_links[tt_ff_links.index==str(link.key)])

tt_ff_links.mean()
df[['tt_avg','tt_ff','tf_inrix']].mean()

df['tt_ff'] = df.groupby('link_key')['tt_ff'].transform(lambda x: x.min())

#Testing
# df['tt_avg'] = df['tt_ff']

df = df.sort_values(by = 'date').copy()

for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]

    n_days, n_hours = len(df_year.date.unique()), len(df_year.hour.unique())

    # TODO: Add an assert to check the dataframe is properly sorted before reshaping it into a tensor

    traveltime_data = get_y_tensor(y=df_year[['tt_avg']], n_links=n_links, n_days=n_days, n_hours=n_hours)
    flow_data = get_y_tensor(y=df_year[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

    Y[year] = tf.concat([traveltime_data, flow_data], axis=3)

    X[year] = get_design_tensor(Z=df_year[features_Z], n_links=n_links, n_days=n_days, n_hours=n_hours)

    tt_ff = get_design_tensor(Z=df_year[['tt_ff']], n_links=n_links, n_days=n_days, n_hours=n_hours)


# plt.hist(df_year['tt_avg'])
#
# bins = np.linspace(-10, 10, 100)

# plt.hist(df_year['tt_ff'], alpha=0.5, label='free flow travel times')
# plt.hist(df_year['tt_avg'], alpha=0.5, label='average travel times')
# plt.legend(loc='upper right')
# plt.show()
#
# plt.scatter(df_year['tt_ff'],df_year['tt_avg'])
# plt.legend(loc='upper right')
# plt.show()

# df.tt_ff.mean()
# df.tt_avg.mean()

## Training and validation sets

# We only pick data from one year
X = X[2019]
Y = Y[2019]

# Prepare the training and validation dataset
X, Y = tf.concat(X,axis = 0), tf.concat(Y,axis = 0)

# Split to comply with temporal ordering
X_train, X_test, Y_train, Y_test = temporal_split(X.numpy(), Y.numpy(), n_days = 20)

# X_train, X_test, Y_train, Y_test = train_test_split(X.numpy(), Y.numpy(), test_size=0.5, random_state=_SEED)
# X_train, X_test, Y_train, Y_test = X[2019], X[2020], Y[2019], Y[2020]

X_train, X_test, Y_train, Y_test = [tf.constant(i) for i in [X_train, X_test, Y_train, Y_test]]

## Models

# models = dict.fromkeys(['m0', 'm1', 'm2', 'm3', 'm4'], True)
run_model = dict.fromkeys(['equilibrium', 'lue', 'ode', 'odlue', 'odlulpe-1','odlulpe-2', 'tvodlulpe'], False)

# run_model.update(dict.fromkeys(['lue', 'odlue', 'odlulpe'], True))
# run_model = dict.fromkeys( for i in ['lue', 'odlue', 'odlulpe'], True)
# run_model['equilibrium'] = True
run_model['lue'] = True
run_model['odlue'] = True
run_model['odlulpe-1'] = True
run_model['odlulpe-2'] = True

train_results_dfs = {}
test_results_dfs = {}
#
# TODO: It will be not included for IATBR abstract and maybe not in paper 2
# run_model['tvodlulpe'] = True

_EPOCHS = 500
_BATCH_SIZE = 16
_LR = 5e-1
_RELATIVE_GAP = 1e-5
_XTICKS_SPACING = 20
_EPOCHS_PRINT_INTERVAL = 10

print(f"Relative gap threshold: {_RELATIVE_GAP}, "
      f"Learning rate: {_LR}, "
      f"Batch size: {_BATCH_SIZE}")

optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

if run_model['equilibrium']:

    print('Equilibrium computation')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'tt': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False,
                                                       'tt': False, 'speed_sd': False, 'median_inc': False,
                                                       'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=fresno_network,
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

    print("\nLink flow based autoencoder")

    suelogit = GISUELOGIT(
        key='suelogit',
        # endogenous_flows=True,
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )
    # X_train.shape
    # Y_train.shape
    train_results_dfs['suelogit'], test_results_dfs['suelogit'] = suelogit.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'bpr': 0, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    # print("\nTravel time based autoencoder")
    # model_0b = AETSUELOGIT(
    #     key='model_0b',
    #     # endogenous_traveltimes=True,
    #     network=fresno_network,
    #     dtype=tf.float64,
    #     equilibrator=equilibrator,
    #     column_generator=column_generator,
    #     utility=utility_parameters,
    #     bpr=bpr_parameters,
    #     od=od_parameters
    # )
    #
    # train_results_dfs['model_0b'], test_results_dfs['model_0b'] = model_0b.train(
    #     X_train, Y_train, X_test, Y_test,
    #     # generalization_error={'train': False, 'validation': True},
    #     optimizer=optimizer,
    #     batch_size=_BATCH_SIZE,
    #     loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 0, 'bpr': 0, 'eq_tt': 1e5},
    #     epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['suelogit'], val_losses=test_results_dfs['suelogit'])

if run_model['lue']:
    print('\nLUE: Benchmark of aesuelogit and isuelogit (utility only)')

    # _RELATIVE_GAP = 1e-4

    # optimizer = NGD(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'speed_sd': True, 'median_inc': True, 'incidents': True,
                                                              'bus_stops': True, 'intersections': True
                                                       },
                                           )

    utility_parameters.constant_initializer(0)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=fresno_network.q.flatten(),
                                 true_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=False)

    equilibrator = Equilibrator(
        network=fresno_network,
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

    lue = GISUELOGIT(
        key='lue',
        network=fresno_network,
        # endogenous_flows=False,
        # endogenous_traveltimes=False,
        dtype=tf.float64,
        equilibrator = equilibrator,
        column_generator = column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_results_dfs['lue'], test_results_dfs['lue'] = lue.train(
        X_train, Y_train, X_test, Y_test,
        # generalization_error={'train': False, 'validation': True},
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights={'od': 0, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1},
        # loss_metric=mnrmse,
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['lue'], val_losses=test_results_dfs['lue'],
                                xticks_spacing = _XTICKS_SPACING)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(lue.theta.numpy())))}")
    print(f"alpha = {lue.alpha: 0.2f}, beta  = {lue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(lue.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['ode']:

    print('\nODE: OD estimation with historic OD')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'psc_factor': 0, 'tt':-1,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False,
                                                       'tt': False, 'speed_sd': False, 'median_inc': False,
                                                       'incidents': False,
                                                       'bus_stops': False, 'intersections': False
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    ode = GISUELOGIT(
        key='ode',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['ode'], test_results_dfs['ode'] = ode.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # generalization_error={'train': False, 'validation': True},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 10},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['ode'], val_losses=test_results_dfs['ode'],
                                xticks_spacing = 50)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(ode.theta.numpy())))}")
    print(f"alpha = {ode.alpha: 0.2f}, beta  = {ode.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(ode.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlue']:

    print('\nODLUE: OD + utility estimation with historic OD')

    # _RELATIVE_GAP = 1e-4\
    # _XTICKS_SPACING = 50

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'speed_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    equilibrator = Equilibrator(
        network=fresno_network,
        # paths_generator=paths_generator,
        utility=utility_parameters,
        max_iters=100,
        method='fw',
        iters_fw=50,
        accuracy=1e-4,
    )

    odlue = GISUELOGIT(
        key='odlue',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlue'], test_results_dfs['odlue'] = odlue.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        # generalization_error={'train': False, 'validation': True},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlue'], val_losses=test_results_dfs['odlue'],
                                xticks_spacing = _XTICKS_SPACING)

    plt.show()

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlue.theta.numpy())))}")
    print(f"alpha = {odlue.alpha: 0.2f}, beta  = {odlue.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlue.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlulpe-1']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix (link specifics alpha)')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # _LR = 1e-2
    # _RELATIVE_GAP = 1e-5
    # _XTICKS_SPACING = 50

    # Some initializations of the bpr parameters, makes the optimization to fail (e.g. alpha =1, beta = 1). Using a common
    # alpha but different betas for every link make the estimation more stable but there is overfitting after a certain amount of iterations

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   # initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                   #                 'beta': 4*np.ones_like(fresno_network.links,dtype = np.float32)},
                                   initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                                   'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15,
                                   #                 'beta': 4 * np.ones_like(fresno_network.links, dtype=np.float32)},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':False},
                                   # trainables={'alpha': True, 'beta': True},
                                   # trainables={'alpha': False, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'psc_factor': 0, 'tt':0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'speed_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           )

    equilibrator = Equilibrator(
        network=fresno_network,
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

    odlulpe_1 = GISUELOGIT(
        key='odlulpe-1',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlulpe-1'], test_results_dfs['odlulpe-1'] = odlulpe_1.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        # loss_metric=mnrmse,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe-1'], val_losses=test_results_dfs['odlulpe-1'],
                                xticks_spacing = _XTICKS_SPACING)

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe_1.theta.numpy())))}")
    print(f"alpha = {np.mean(odlulpe_1.alpha): 0.2f}, beta  = {np.mean(odlulpe_1.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlulpe_1.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['odlulpe-2']:

    print('\nODLULPE: ODLUE + link performance parameters with historic OD matrix (link specifics alphas and betas)')

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    # _LR = 5e-1
    # _RELATIVE_GAP = 1e-5

    # Some initializations of the bpr parameters, makes the optimization to fail (e.g. alpha =1, beta = 1). Using a common
    # alpha but different betas for every link make the estimation more stable but there is overfitting after a certain amount of iterations

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                                   'beta': 4*np.ones_like(fresno_network.links,dtype = np.float32)},
                                   # initial_values={'alpha': 0.15*np.ones_like(fresno_network.links,dtype = np.float32),
                                   #                 'beta': 4},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.15,
                                   #                 'beta': 4 * np.ones_like(fresno_network.links, dtype=np.float32)},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   # trainables={'alpha': True, 'beta':False},
                                   trainables={'alpha': True, 'beta': True},
                                   # trainables={'alpha': False, 'beta': True},
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=1,
                                           initial_values={'psc_factor': 0, 'tt':0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': True,
                                                       'tt': True, 'speed_sd': True, 'median_inc': True,
                                                       'incidents': True,
                                                       'bus_stops': True, 'intersections': True
                                                       },
                                           )

    equilibrator = Equilibrator(
        network=fresno_network,
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

    odlulpe_2 = GISUELOGIT(
        key='odlulpe_2',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['odlulpe_2'], test_results_dfs['odlulpe_2'] = odlulpe_2.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        # loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'eq_flow': 1},
        loss_weights={'od': 1, 'theta': 0, 'tt': 1e10, 'flow': 1, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs_print_interval=_EPOCHS_PRINT_INTERVAL,
        # loss_metric=mnrmse,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['odlulpe_2'], val_losses=test_results_dfs['odlulpe_2'],
                                xticks_spacing = _XTICKS_SPACING)

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(odlulpe_2.theta.numpy())))}")
    print(f"alpha = {np.mean(odlulpe_2.alpha): 0.2f}, beta  = {np.mean(odlulpe_2.beta): 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(odlulpe_2.q - fresno_network.q.flatten())): 0.2f}")
    print(f"Avg observed OD: {np.mean(np.abs(fresno_network.q.flatten())): 0.2f}")

if run_model['tvodlulpe']:
    print('\ntvodlulpe: Time specific utility and OD, link performance parameters, no historic OD')

    utility_parameters = UtilityParameters(features_Y=['tt'],
                                           features_Z=features_Z,
                                           periods=3,
                                           initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                           'fixed_effect': np.zeros_like(fresno_network.links)},
                                           signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                                  'bus_stops': '-', 'intersections': '-'},
                                           trainables={'psc_factor': False, 'fixed_effect': False},
                                           )

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   # initial_values={'alpha': 0.1, 'beta': 1},
                                   trainables={'alpha': True, 'beta': False},
                                   # trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=3,
                                 initial_values=fresno_network.q.flatten(),
                                 trainable=True)

    tvodlulpe = GISUELOGIT(
        key='tvodlulpe',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_results_dfs['tvodlulpe'], test_results_dfs['tvodlulpe'] = tvodlulpe.train(
        X_train, Y_train, X_test, Y_test,
        optimizer=optimizer,
        # generalization_error={'train': False, 'validation': True},
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0, 'eq_flow': 1},
        threshold_relative_gap=_RELATIVE_GAP,
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_results_dfs['tvodlulpe'], val_losses=test_results_dfs['tvodlulpe'])

    plt.show()

    print(f"features = {utility_parameters.features}")
    print(f"theta = {'tvodlulpe'.theta.numpy()}")
    print(f"alpha = {'tvodlulpe'.alpha: 0.2f}, beta  = {'tvodlulpe'.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs('tvodlulpe'.q - fresno_network.q.flatten())): 0.2f}")
