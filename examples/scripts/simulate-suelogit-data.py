import numpy as np
import os
from pathlib import Path

from src.spad.aesue import simulate_features, build_tntp_network, simulate_suelogit_data, UtilityFunction, \
    Equilibrator, get_design_tensor, get_counts_tensor, load_k_shortest_paths

# Path management
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:',main_dir)

tntp_network = build_tntp_network(network_name='SiouxFalls')

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

n_periods = 128
n_links = len(tntp_network.links)
features_Z = ['c', 's']

n_sparse_features = 3
features_sparse = ['k' + str(i) for i in np.arange(0, n_sparse_features)]

exogenous_features = simulate_features(links=tntp_network.links,
                                      features_Z= features_Z + features_sparse,
                                      option='continuous',
                                      range=(0, 1),
                                      n_periods = n_periods)

utility_function = UtilityFunction(features_Y=['tt'],
                                   features_Z=features_Z,
                                   true_values={'tt': -1, 'c': -6, 's': -3}
                                   )
utility_function.add_sparse_features(Z=features_sparse)


equilibrator = Equilibrator(network=tntp_network,
                            utility_function=utility_function,
                            uncongested_mode=False,
                            max_iters=100,
                            method='fw',
                            iters_fw=100,
                            search_fw='grid')

# Generate data from multiple days by varying the value of the exogenous attributes instead of adding random noise only
df = simulate_suelogit_data(
    periods = list(exogenous_features.period.unique()),
    features_data = exogenous_features,
    equilibrator=equilibrator,
    network = tntp_network)

output_file = tntp_network.key + '-link-data.csv'
output_dir = Path('output/network-data/' + tntp_network.key + '/links')

output_dir.mkdir(parents=True, exist_ok=True)

# Write data to csv
df.to_csv(output_dir / output_file, index=False)