# Setup
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import tensorflow as tf
import random

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

# Internal modules
from src.aesuelogit.visualizations import plot_predictive_performance, plot_convergence_estimates
from src.aesuelogit.models import UtilityParameters, BPRParameters, ODParameters, AESUELOGIT, NGD
from src.aesuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data

# Seed for reproducibility
_SEED = 2022

np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

network_name = 'SiouxFalls'

tntp_network = build_tntp_network(network_name=network_name)

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

# REad synthethic data which was generated under the assumption of path sets of size 2.
df = pd.read_csv(main_dir + '/output/network-data/' + network_name + '/links/' + network_name + '-link-data.csv')

n_days = len(df.period.unique())
n_links = len(tntp_network.links)
n_hours = 1

features_Z = ['c', 's']
# features_Z = []

n_sparse_features = 0
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]
# features_sparse = []
features_Z.extend(features_sparse)

utility_parameters = UtilityParameters(features_Y=['tt'],
                                       features_Z=features_Z,
                                       periods=1,
                                       true_values={'tt': -1, 'c': -6, 's': -3,
                                                    'psc_factor': 0,
                                                    'fixed_effect': np.zeros_like(tntp_network.links)},
                                       initial_values={'tt': 0, 'c': -6, 's': -3, 'psc_factor': 0,
                                                       'fixed_effect': np.zeros_like(tntp_network.links)},
                                       trainables={'psc_factor': False, 'fixed_effect': False}
                                       # trainables = None, #['features','psc_factor, fixed_effect']
                                       )

bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                               # initial_values={'alpha': np.ones(n_links) * 0.1, 'beta': 1},
                               initial_values={'alpha': 0.1, 'beta': 1},
                               true_values={'alpha': 0.15, 'beta': 4},
                               # true_values={'alpha': np.ones(n_links) * 0.15, 'beta': 4},
                               trainables=dict.fromkeys(['alpha', 'beta'], True),
                               # link_specifics = dict.fromkeys(['alpha','beta'],True)
                               # trainables = ['alpha'],
                               )

od_parameters = ODParameters(key='od',
                             periods=1,
                             # initial_values = np.ones_like(tntp_network.q),
                             initial_values=0.6 * tntp_network.q.flatten(),
                             true_values=tntp_network.q.flatten(),
                             historic_values={1: tntp_network.q.flatten()},
                             trainable=True)

# Prepare the training and validation dataset.

# Add free flow travel times
df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_days)

# X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)

traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_days=n_days, n_hours=n_hours)
flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

Y = tf.concat([traveltime_data, flow_data], axis=3)
X = get_design_tensor(Z=df[['traveltime'] + features_Z], y=df['tt_ff'], n_links=n_links, n_days=n_days, n_hours=n_hours)

X_train, X_val, Y_train, Y_val = train_test_split(X.numpy(), Y.numpy(), test_size=0.5, random_state=42)

X_train, X_val, Y_train, Y_val = [tf.constant(i) for i in [X_train, X_val, Y_train, Y_val]]

_EPOCHS = 5
_BATCH_SIZE = 4
_LR = 5e-1

optimizer = NGD(learning_rate=_LR)
# optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
# optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

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

model = AESUELOGIT(
    key='model',
    network=tntp_network,
    dtype=tf.float64,
    equilibrator=equilibrator,
    column_generator=column_generator,
    utility=utility_parameters,
    bpr=bpr_parameters,
    od=od_parameters,
)

train_results, val_results = model.train(
    X_train, Y_train, X_val, Y_val,
    optimizer=optimizer,
    batch_size=_BATCH_SIZE,
    loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
    epochs=_EPOCHS)

plot_predictive_performance(train_losses=train_results, val_losses=val_results)

estimates_columns = [col for col in train_results.columns if 'loss_' not in col]

true_values = pd.Series({k: v for k, v in {**bpr_parameters.true_values, **utility_parameters.true_values}.items()
                         if k in estimates_columns})

plot_convergence_estimates(estimates=train_results[estimates_columns], true_values=true_values)
