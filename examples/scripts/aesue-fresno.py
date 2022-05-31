'''
'''

import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
from src.aesuelogit.models import UtilityParameters, BPRParameters, ODParameters, AESUELOGIT, NGD
from src.aesuelogit.visualizations import plot_predictive_performance
from src.aesuelogit.networks import load_k_shortest_paths, read_paths, build_fresno_network, \
    Equilibrator, sparsify_OD, ColumnGenerator, read_OD
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, data_curation

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

## Read paths
# read_paths(network=fresno_network, update_incidence_matrices=True, filename='paths-fresno.csv')
read_paths(network=fresno_network, update_incidence_matrices=True, filename = 'paths-full-model-fresno.csv')

# For quick testing
# Q = fresno_network.load_OD(sparsify_OD(fresno_network.Q, prop_od_pairs=0.99))
# load_k_shortest_paths(network=fresno_network, k=2, update_incidence_matrices=True)

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

# df.link_type.value_counts()

df = data_curation(df)

## Utility function

features_Z = ['speed_sd', 'median_inc', 'incidents', 'bus_stops', 'intersections']
# features_Z = ['speed_sd']
# features_Z = []

utility_parameters = UtilityParameters(features_Y=['tt'],
                                       features_Z=features_Z,
                                       periods = 1,
                                       initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                       'fixed_effect': np.zeros_like(fresno_network.links)},
                                       signs={'tt': '-', 'speed_sd': '-', 'median_inc': '+', 'incidents': '-',
                                              'bus_stops': '-', 'intersections': '-'},
                                       trainables={'psc_factor': False, 'fixed_effect': False},
                                       )

utility_parameters.constant_initializer(0)

## Data processing

n_links = len(fresno_network.links)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df['year'] = df.date.dt.year
X, Y = {}, {}

# Select only dates used for previous paper
# df = df.query('date == "2019-10-01"  | date == "2020-10-06"')
# df = df.query('date == "2019-10-01"')
# df = df.query('hour == 16')

print(df.query('year == 2019')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

print(df.query('year == 2020')[['counts', 'tt_ff', 'tt_avg', 'tf_inrix']].describe())

# Normalization of features to range [0,1]

# (TODO: may enable normalization in get_design_tensor method. See if tensorflow have it)
df[features_Z + ['tt_avg'] + ['tt_ff']] \
    = preprocessing.MaxAbsScaler().fit_transform(df[features_Z + ['tt_avg'] + ['tt_ff']])

# Set free flow travel times
# tt_ff_links = df.query('link_type == "LWRLK"').groupby('link_key')['tt_ff'].min()
tt_ff_links = df.groupby('link_key')['tt_ff'].min()
# [(link.bpr.tf,link.link_type) for link in fresno_network.links if link.link_type == "LWRLK"]
for link in fresno_network.links:
    fresno_network.links_dict[link.key].performance_function.tf = float(tt_ff_links[tt_ff_links.index==str(link.key)])

#Testing
# df['tt_avg'] = df['tt_ff']

for year in sorted(df['year'].unique()):
    df_year = df[df['year'] == year]

    n_days, n_hours = len(df_year.date.unique()), len(df_year.hour.unique())

    # TODO: Add an assert to check the dataframe is properly sorted before reshaping it into a tensor

    traveltime_data = get_y_tensor(y=df_year[['tt_avg']], n_links=n_links, n_days=n_days, n_hours=n_hours)
    flow_data = get_y_tensor(y=df_year[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

    Y[year] = tf.concat([traveltime_data, flow_data], axis=3)

    X[year] = get_design_tensor(Z=df_year[['tt_avg'] + features_Z], n_links=n_links, n_days=n_days, n_hours=n_hours)

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

## Training

# Prepare the training and validation dataset
# X, Y = tf.concat(list(X.values()),axis = 0), tf.concat(list(Y.values()),axis = 0)
# X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size=0.5, random_state=42)
X_train, X_val, Y_train, Y_val = X[2019], X[2020], Y[2019], Y[2020]

X_train, X_val, Y_train, Y_val = [tf.constant(i) for i in [X_train, X_val, Y_train, Y_val]]

_EPOCHS = 3
_BATCH_SIZE = 8
_LR = 5e-2  # Default is 1e-3. With 1e-1, training becomes unstable

# models = dict(zip(['m1', 'm2', 'm3', 'm4'], True))
models = dict.fromkeys(['m1', 'm2', 'm3', 'm4'], False)
# models['m1'] = True
# models['m2'] = True
# models['m3'] = True
models['m4'] = True

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

train_losses_dfs = {}
val_losses_dfs = {}

if models['m1']:
    print('\nmodel 1: Benchmark of aesuelogit and isuelogit (utility only)')

    _LR = 5e-1

    optimizer = NGD(learning_rate=_LR)

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

    model_1 = AESUELOGIT(
        key='model_1',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator = equilibrator,
        column_generator = column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters
    )

    train_losses_dfs['model_1'], val_losses_dfs['model_1'] = model_1.train(
        X_train, Y_train, X_val, Y_val,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 0, 'flow': 1, 'bpr': 0},
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_1'], val_losses=val_losses_dfs['model_1'])

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_1.theta.numpy())))}")
    print(f"alpha = {model_1.alpha: 0.2f}, beta  = {model_1.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_1.q - fresno_network.q.flatten())): 0.2f}")

if models['m2']:
    print('\nmodel 2: OD + utility estimation with historic OD')

    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    model_2 = AESUELOGIT(
        key='model_2',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_2'], val_losses_dfs['model_2'] = model_2.train(
        X_train, Y_train, X_val, Y_val,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 1, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_2'], val_losses=val_losses_dfs['model_2'])

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_2.theta.numpy())))}")
    print(f"alpha = {model_2.alpha: 0.2f}, beta  = {model_2.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_2.q - fresno_network.q.flatten())): 0.2f}")

if models['m3']:

    print('\nmodel 3: ODLUE + link performance parameters without historic OD matrix')

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 1},
                                   # initial_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta':False},
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 initial_values=fresno_network.q.flatten(),
                                 # historic_values={1: fresno_network.q.flatten()},
                                 trainable=True)

    model_3 = AESUELOGIT(
        key='model_3',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_3'], val_losses_dfs['model_3'] = model_3.train(
        X_train, Y_train, X_val, Y_val,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_3'], val_losses=val_losses_dfs['model_3'])

    print(f"theta = {dict(zip(utility_parameters.true_values.keys(), list(model_3.theta.numpy())))}")
    print(f"alpha = {model_3.alpha: 0.2f}, beta  = {model_3.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_3.q - fresno_network.q.flatten())): 0.2f}")

if models['m4']:
    print('\nmodel 4: Time specific utility and OD, link performance parameters, no historic OD')

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

    model_4 = AESUELOGIT(
        key='model_4',
        network=fresno_network,
        dtype=tf.float64,
        equilibrator=equilibrator,
        column_generator=column_generator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    train_losses_dfs['model_4'], val_losses_dfs['model_4'] = model_4.train(
        X_train, Y_train, X_val, Y_val,
        optimizer=optimizer,
        batch_size=_BATCH_SIZE,
        loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
        epochs=_EPOCHS)

    plot_predictive_performance(train_losses=train_losses_dfs['model_4'], val_losses=val_losses_dfs['model_4'])

    print(f"features = {utility_parameters.features}")
    print(f"theta = {model_4.theta.numpy()}")
    print(f"alpha = {model_4.alpha: 0.2f}, beta  = {model_4.beta: 0.2f}")
    print(f"Avg abs diff of observed and estimated OD: {np.mean(np.abs(model_4.q - fresno_network.q.flatten())): 0.2f}")
