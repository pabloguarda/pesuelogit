import numpy as np
import os
import pandas as pd
from pathlib import Path
import isuelogit as isl
import random

from src.gisuelogit.models import UtilityParameters, NGD
from src.gisuelogit.networks import build_tntp_network, Equilibrator, load_k_shortest_paths
from src.gisuelogit.etl import simulate_features, simulate_suelogit_data, get_design_tensor, get_y_tensor

# Seed for reproducibility
_SEED = 2022

np.random.seed(_SEED)
random.seed(_SEED)


# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:',main_dir)

## Build network
network_name = 'SiouxFalls'
# network_name = 'Eastern-Massachusetts'
tntp_network = build_tntp_network(network_name=network_name)

## Read OD matrix
Q = isl.reader.read_tntp_od(network_name=network_name)
tntp_network.load_OD(Q=Q)

# links_df = isl.reader.read_tntp_linkdata(network_name='SiouxFalls')
# links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]
#
# # Link performance functions (assumed linear for consistency with link_cost function definion)
# tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
#                                                      'alpha': links_df.b,
#                                                      'beta': links_df.power,
#                                                      'tf': links_df.free_flow_time,
#                                                      'k': links_df.capacity
#                                                      }))

# Paths
load_k_shortest_paths(network=tntp_network, k=2, update_incidence_matrices=True)

n_days = 100
n_links = len(tntp_network.links)
features_Z = ['c', 's']

n_sparse_features = 0
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

utility_function = UtilityParameters(features_Y=['tt'],
                                     features_Z=features_Z,
                                     true_values={'tt': -1, 'c': -6, 's': -3}
                                     )

utility_function.add_sparse_features(Z=features_sparse)


equilibrator = Equilibrator(network=tntp_network,
                            utility_function=utility_function,
                            uncongested_mode=False,
                            max_iters=100,
                            method='fw',
                            accuracy = 1e-10,
                            iters_fw=100,
                            search_fw='grid')

exogenous_features = simulate_features(links=tntp_network.links,
                                       features_Z= features_Z + features_sparse,
                                       option='continuous',
                                       daytoday_variation=False,
                                       range=(0, 1),
                                       n_days = n_days)

# Generate data from multiple days. The value of the exogenous attributes varies between links but not between days (note: sd_x is the standard deviation relative to the true mean of traffic counts)

df = simulate_suelogit_data(
    days= list(exogenous_features.period.unique()),
    features_data = exogenous_features,
    equilibrator=equilibrator,
    sd_x = 0.1,
    sd_t = 0.1,
    network = tntp_network)

output_file = tntp_network.key + '-link-data.csv'
output_dir = Path('output/network-data/' + tntp_network.key + '/links')

output_dir.mkdir(parents=True, exist_ok=True)

# Write data to csv
df.to_csv(output_dir / output_file, index=False)