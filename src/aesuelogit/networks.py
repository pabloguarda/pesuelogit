import isuelogit as isl
from isuelogit.networks import TNetwork
from isuelogit.printer import block_output, printIterationBar
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple
import ast
from sklearn import preprocessing


def read_paths(network, **kwargs) -> None:
    isl.factory.PathsGenerator().read_paths(network=network, **kwargs)



def build_tntp_network(network_name) -> TNetwork:
    '''
    Read data from tntp repository and build network object
    '''

    links_df = isl.reader.read_tntp_linkdata(network_name=network_name)
    links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

    network_generator = isl.factory.NetworkGenerator()
    A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))
    tntp_network = network_generator.build_network(A=A, network_name=network_name)

    # Link performance functions
    tntp_network.set_bpr_functions(bprdata=pd.DataFrame({'link_key': tntp_network.links_dict.keys(),
                                                         'alpha': links_df.b,
                                                         'beta': links_df.power,
                                                         'tf': links_df.free_flow_time,
                                                         'k': links_df.capacity
                                                         }))

    # Link features from TNTP repo
    # link_features_df = links_df[['link_key', 'length', 'speed', 'link_type', 'toll']]

    # OD matrix
    Q = isl.reader.read_tntp_od(network_name=network_name)
    tntp_network.load_OD(Q=Q)

    return tntp_network


def build_fresno_network():
    # Read nodes data
    nodes_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'nodes/' + 'fresno-nodes-data.csv')

    # Read link specific attributes
    links_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'links/' 'fresno-link-specific-data.csv',
                           converters={"link_key": ast.literal_eval, "pems_id": ast.literal_eval})

    links_df['free_flow_speed'] = links_df['length'] / links_df['tf_inrix']

    network_generator = isl.factory.NetworkGenerator()

    network = \
        network_generator.build_fresno_network(
            A=network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values)),
            links_df=links_df, nodes_df=nodes_df, network_name='Fresno')

    bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                      'alpha': links_df['alpha'],
                                      'beta': links_df['beta'],
                                      'tf': links_df['tf_inrix'],
                                      # 'tf': links_df['tf'],
                                      'k': pd.to_numeric(links_df['k'], errors='coerce', downcast='float')
                                      })

    # Normalize free flow travel time between 0 and 1
    bpr_parameters_df['tf'] = pd.DataFrame(
        preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf']).reshape(-1, 1)))

    network.set_bpr_functions(bprdata=bpr_parameters_df)

    network_generator.read_OD(network=network, sparse=True)

    return network


def load_k_shortest_paths(network, k, update_incidence_matrices=False, **kwargs):
    isl.factory.PathsGenerator().load_k_shortest_paths(network=network,
                                                       k=k,
                                                       update_incidence_matrices=False,
                                                       **kwargs)
    if update_incidence_matrices:
        paths_od = network.paths_od
        network.D = network.generate_D(paths_od=paths_od, links=network.links)
        network.M = network.generate_M(paths_od=paths_od)

        # TODO: remove dependency on C after translating operations to sparse representation
        network.C = network.generate_C(network.M)
