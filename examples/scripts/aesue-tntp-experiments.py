# Setup
import os
from pathlib import Path
import random

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

# isl.config.dirs['read_network_data'] = "input/network-data/SiouxFalls/"

# Internal modules
from src.aesuelogit.models import UtilityParameters, BPRParameters, ODParameters, AESUELOGIT, NGD
from src.aesuelogit.networks import load_k_shortest_paths, build_tntp_network, Equilibrator, ColumnGenerator
from src.aesuelogit.etl import get_design_tensor, get_y_tensor, simulate_suelogit_data
from src.aesuelogit.experiments import *

# Seed for reproducibility
_SEED = 2022

np.random.seed(_SEED)
random.seed(_SEED)
tf.random.set_seed(_SEED)

# Experiments
list_experiments = ['convergence', 'multiday', 'noisy_counts', 'noisy_od', 'ill_scaled_od']

# run_experiment = dict.fromkeys(list_experiments,True)
run_experiment = dict.fromkeys(list_experiments, False)

# run_experiment['convergence'] = True
run_experiment['multiday'] = True
# run_experiment['noisy_counts'] = True
# run_experiment['noisy_od'] = True
# run_experiment['ill_scaled_od'] = True

## Build network
network_name = 'SiouxFalls'

tntp_network = build_tntp_network(network_name=network_name)

## Read OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q=Q)

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)
# features_Z = []

# Read synthethic data which was generated under the assumption of path sets of size 2.
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

## Prepare the training and validation dataset

# Add free flow travel times
# df['tt_ff'] = np.tile([link.bpr.tf for link in tntp_network.links], n_days)

# X_train, X_val, y_train, y_val = train_test_split(input_data, traveltime_data, test_size=0.2, random_state=42)

traveltime_data = get_design_tensor(y=df['traveltime'], n_links=n_links, n_days=n_days, n_hours=n_hours)
flow_data = get_y_tensor(y=df[['counts']], n_links=n_links, n_days=n_days, n_hours=n_hours)

Y = tf.concat([traveltime_data, flow_data], axis=3)
X = get_design_tensor(Z=df[['traveltime'] + features_Z], n_links=n_links, n_days=n_days, n_hours=n_hours)

equilibrator = Equilibrator(
    network=tntp_network,
    # paths_generator=paths_generator,
    # utility=utility_parameters,
    max_iters=100,
    method='fw',
    iters_fw=50,
    accuracy=1e-4,
)
#
# column_generator = ColumnGenerator(equilibrator=equilibrator,
#                                    utility=utility_parameters,
#                                    n_paths=0,
#                                    ods_coverage=0.1,
#                                    ods_sampling='sequential',
#                                    # ods_sampling='demand',
#                                    )
## Learning parameters

_EPOCHS = 4
_BATCH_SIZE = None  # None
_LR = 5e-2

utility_parameters = UtilityParameters(features_Y=['tt'],
                                       features_Z=features_Z,
                                       true_values={'tt': -1, 'c': -6, 's': -3, 'psc_factor': 0, 'fixed_effect': 0},
                                       initial_values={'tt': 0, 'c': 0, 's': 0, 'psc_factor': 0,
                                                       'fixed_effect': np.zeros_like(tntp_network.links)},
                                       trainables={'psc_factor': False, 'fixed_effect': False},
                                       # trainables = None, #['features','psc_factor, fixed_effect']
                                       )


if run_experiment['convergence']:

    # optimizer = NGD(learning_rate=_LR)
    # optimizer = tf.keras.optimizers.Adagrad(learning_rate=_LR)
    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   initial_values={'alpha': 0.15, 'beta': 1},
                                   # initial_values={'alpha': 0.05, 'beta': 2},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   trainables={'alpha': True, 'beta': True},
                                   # link_specifics = dict.fromkeys(['alpha','beta'],True)
                                   # trainables = ['alpha'],
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values = np.ones_like(tntp_network.q),
                                 # initial_values=0.6 * tntp_network.q.flatten(),
                                 initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={1: tntp_network.q.flatten()},
                                 trainable=False)

    model = AESUELOGIT(
        key='model',
        network=tntp_network,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    convergence_experiment = ConvergenceExperiment(
        seed=_SEED,
        name='Convergence Experiment',
        folderpath=isl.config.dirs['output_folder'] + 'experiments/' + network_name,
        model=model,
        optimizer=optimizer,
        X=X,
        Y=Y)

    convergence_experiment.run(epochs=_EPOCHS,
                               batch_size=_BATCH_SIZE,
                               loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 1},)

if run_experiment['multiday']:

    optimizer = tf.keras.optimizers.Adam(learning_rate=_LR)

    bpr_parameters = BPRParameters(keys=['alpha', 'beta'],
                                   # initial_values={'alpha': 0.1, 'beta': 1},
                                   initial_values={'alpha': 0.15, 'beta': 4},
                                   true_values={'alpha': 0.15, 'beta': 4},
                                   trainables=dict.fromkeys(['alpha', 'beta'], False),
                                   )

    od_parameters = ODParameters(key='od',
                                 periods=1,
                                 # initial_values=100*np.ones_like(tntp_network.q).flatten(),
                                 initial_values=0.6*tntp_network.q.flatten(),
                                 # initial_values=tntp_network.q.flatten(),
                                 true_values=tntp_network.q.flatten(),
                                 historic_values={1: tntp_network.q.flatten()},
                                 trainable=True)

    model = AESUELOGIT(
        key='model',
        network=tntp_network,
        equilibrator=equilibrator,
        utility=utility_parameters,
        bpr=bpr_parameters,
        od=od_parameters,
    )

    multiday_experiment = MultidayExperiment(
        seed=_SEED,
        name='Multiday Experiment',
        folderpath=isl.config.dirs['output_folder'] + 'experiments/' + network_name,
        model=model,
        optimizer=optimizer,
        noise = {'tt': 0, 'flow': 0.05},
        X=X,
        Y=Y)

    multiday_experiment.run(epochs=_EPOCHS,
                            replicates = 10,
                            range_initial_values= (-1,1),
                            batch_size=_BATCH_SIZE,
                            levels = [10,50,100],
                            loss_weights={'od': 0, 'theta': 0, 'tt': 1, 'flow': 1, 'bpr': 0},
                            )