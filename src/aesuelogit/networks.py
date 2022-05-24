import isuelogit as isl
from isuelogit.networks import TNetwork
from isuelogit.printer import block_output, printIterationBar
import numpy as np
import pandas as pd
from isuelogit.nodes import Node
from isuelogit.links import Link
import tensorflow as tf
from typing import Dict, List, Tuple, Optional

from isuelogit.geographer import NodePosition
from isuelogit.links import BPR
from isuelogit.paths import Path
from isuelogit.printer import printProgressBar
from isuelogit.equilibrium import LUE_Equilibrator

import ast
import time
from sklearn import preprocessing


class TransportationNetwork(isl.networks.DiTNetwork):
    def __init__(self, key=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if key is not None:
            self._key = key

    def set_bpr_functions(self,
                          bprdata: pd.DataFrame,
                          parameters_keys=['alpha', 'beta', 'tf', 'k'],
                          link_key: Optional[str] = None) -> None:

        if link_key is None:
            link_key = 'link_key'

        links_keys = list(bprdata[link_key].values)

        counter = 0

        for key in links_keys:
            alpha, beta, tf, k \
                = tuple(bprdata[bprdata[link_key] == key][parameters_keys].values.flatten())

            self.links_dict[key].performance_function = BPR(alpha=alpha,
                                                            beta=beta,
                                                            tf=tf,
                                                            k=k)

            self.links_dict[key].bpr = self.links_dict[key].performance_function

            # Add parameters to Z_Dict
            self.links_dict[key].Z_dict['alpha'] = alpha
            self.links_dict[key].Z_dict['beta'] = beta
            self.links_dict[key].Z_dict['tf'] = tf
            self.links_dict[key].Z_dict['k'] = k

            # Initialize link travel time
            self.links_dict[key].set_traveltime_from_x(x=0)

            # Idx useful to speed up some operations with incidence matrices
            self.links_dict[key].order = counter

            counter += 1

    @staticmethod
    def generate_D(paths_od: Dict[Tuple[int, int], Path], links: List[Link], paths=None):
        """Matrix D: Path-link incidence matrix"""

        t0 = time.time()

        # print('Generating matrix D ')

        if paths is None:
            paths = []
            for pair in paths_od.keys():
                paths.extend(paths_od[pair])

        n_paths = len(paths)
        n_links = len(links)

        D = np.zeros([n_links, n_paths], dtype=np.int64)

        # print('\n')
        # links_paths_adjacencies = []
        for i, path in enumerate(paths):
            printProgressBar(i, n_paths - 1, prefix='Generating D:', suffix='', length=20)

            links_path_list = path.links
            for link in links_path_list:
                # links_paths_adjacencies.append([link.order, i])
                D[link.order, i] = 1

        # D = tf.sparse.reorder(tf.SparseTensor(indices = links_paths_adjacencies,
        #                                       values = tf.ones(len(links_paths_adjacencies)),
        #                                       dense_shape = (n_links,n_paths)))
        #
        # D = tf.sparse.to_dense(D)

        # assert D.shape[0] > 0, 'No matrix D generated'

        print('Matrix D ' + str(D.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]               \n')

        return D

    @staticmethod
    def generate_C(M):
        """Wide to long format
        Choice_set_matrix_from_M
        The new matrix has one rows per alternative

        # This is the availability matrix or choice set matrix. Note that it is very expensive to
        compute when using matrix operation Iq.T.dot(Iq) so a more programatic method was preferred
        """

        t0 = time.time()

        assert M.shape[0] > 0, 'Matrix C was not generated because M matrix is empty'

        # print('Generating choice set matrix')

        wide_matrix = M.astype(int)

        if wide_matrix.ndim == 1:
            wide_matrix = wide_matrix.reshape(1, wide_matrix.shape[0])

        C = np.repeat(wide_matrix, repeats=np.sum(wide_matrix, axis=1), axis=0)

        print('Matrix C ' + str(C.shape) + ' generated in ' + str(round(time.time() - t0, 1)) + '[s]               \n')

        return C


class Equilibrator(LUE_Equilibrator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def path_based_suelogit_equilibrium(self, *args, **kwargs):
        return LUE_Equilibrator.path_based_suelogit_equilibrium(self, *args, **kwargs)


class ColumnGenerator():
    def __init__(self,
                 utility_function,
                 equilibrator=None,
                 paths_generator=None,
                 **kwargs):

        # Create an arbitrary a equilibrator and provide it with an arbitrary path generator when no provided
        self.utility_function = utility_function
        self.paths_generator = paths_generator
        self.equilibrator = equilibrator

        if self.equilibrator is None:
            self.equilibrator = Equilibrator()

        elif self.equilibrator.paths_generator is None and self.paths_generator is None:
            self.paths_generator = isl.factory.PathsGenerator()
        elif self.equilibrator.paths_generator is not None and self.paths_generator is None:
            self.path_generator = self.equilibrator.paths_generator

        self.equilibrator.paths_generator = self.paths_generator

        column_generation_options = ['ods_coverage', 'ods_sampling', 'n_paths']

        self.options = {option: value for option, value in self.equilibrator.options['column_generation'].items()
                        if option in column_generation_options}

        for k, v in kwargs.items():
            if k in column_generation_options:
                self.options[k] = v

        # super().__init__(*args, **kwargs)

    def generate_paths(self,
                       *args,
                       **kwargs) -> None:

        # Pick a coordinate of theta by random if it has rank 2
        if tf.rank(kwargs['theta']) == 2:
            kwargs['theta'] = kwargs['theta'][np.random.randint(0, kwargs['theta'].shape[0]), :]

        kwargs['theta'] = {k: v for k, v in zip(self.utility_function.features, list(kwargs['theta']))}

        if self.options['n_paths'] > 0 and self.options['ods_coverage'] > 0:
            t0 = time.time()
            _ = self.equilibrator.sue_column_generation(*args, **kwargs, **self.options)
            print(f"Time: {time.time() - t0: 0.2g}\n")

    def select_paths(self, *args, **kwargs) -> None:
        self.equilibrator.path_set_selection(*args, **kwargs)


def build_tntp_network(network_name) -> TNetwork:
    '''
    Read data from tntp repository and build network object
    '''

    links_df = isl.reader.read_tntp_linkdata(network_name=network_name)
    links_df['link_key'] = [(i, j, '0') for i, j in zip(links_df['init_node'], links_df['term_node'])]

    # # Normalize free flow travel time between 0 and 1
    # links_df['free_flow_time'] = pd.DataFrame(
    #     preprocessing.MinMaxScaler().fit_transform(np.array(links_df['free_flow_time']).reshape(-1, 1)))

    network_generator = isl.factory.NetworkGenerator()
    A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))
    # tntp_network = network_generator.build_network(A=A, network_name=network_name)
    tntp_network = TransportationNetwork(A=A, key=network_name)

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


def build_small_network(network_name):
    network_generator = isl.factory.NetworkGenerator()

    As, Qs = network_generator.get_A_Q_custom_networks([network_name])

    network = TransportationNetwork(A=As[network_name], key=network_name)

    network.load_OD(Q=Qs[network.key])

    # Link performance functions
    linkdata_generator = isl.factory.LinkDataGenerator()

    if network.key == 'Yang':
        bpr_parameters_df = linkdata_generator.generate_Yang_bpr_parameters()
    elif network.key == 'Lo':
        bpr_parameters_df = linkdata_generator.generate_LoChan_bpr_parameters()
    elif network.key == 'Wang':
        bpr_parameters_df = linkdata_generator.generate_Wang_bpr_parameters()
    elif network.key == 'Toy':
        bpr_parameters_df = linkdata_generator.generate_toy_bpr_parameters()

    network.set_bpr_functions(bpr_parameters_df)

    return network


def build_fresno_network():
    # Read nodes data
    nodes_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'nodes/' + 'fresno-nodes-data.csv')

    # Read link specific attributes
    links_df = pd.read_csv(isl.config.dirs['read_network_data'] + 'links/' 'fresno-link-specific-data.csv',
                           converters={"link_key": ast.literal_eval, "pems_id": ast.literal_eval})

    links_df['free_flow_speed'] = links_df['length'] / links_df['tf_inrix']

    network_generator = isl.factory.NetworkGenerator()

    A = network_generator.generate_adjacency_matrix(links_keys=list(links_df['link_key'].values))

    network = TransportationNetwork(A=A, key='fresno')

    # Create link objects and set BPR functions and attributes values associated each link
    network.links_dict = {}
    network.nodes_dict = {}

    for index, row in links_df.iterrows():

        link_key = row['link_key']

        # Adding gis information via nodes object store in each link
        init_node_row = nodes_df[nodes_df['key'] == link_key[0]]
        term_node_row = nodes_df[nodes_df['key'] == link_key[1]]

        x_cord_origin, y_cord_origin = tuple(list(init_node_row[['x', 'y']].values[0]))
        x_cord_term, y_cord_term = tuple(list(term_node_row[['x', 'y']].values[0]))

        if link_key[0] not in network.nodes_dict.keys():
            network.nodes_dict[link_key[0]] = Node(key=link_key[0],
                                                   position=NodePosition(x_cord_origin, y_cord_origin, crs='xy'))

        if link_key[1] not in network.nodes_dict.keys():
            network.nodes_dict[link_key[1]] = Node(key=link_key[1],
                                                   position=NodePosition(x_cord_term, y_cord_term, crs='xy'))

        node_init = network.nodes_dict[link_key[0]]
        node_term = network.nodes_dict[link_key[1]]

        network.links_dict[link_key] = Link(key=link_key, init_node=node_init, term_node=node_term)

        # Store original ids from nodes and links
        network.links_dict[link_key].init_node.id = str(init_node_row['id'].values[0])
        network.links_dict[link_key].term_node.id = str(term_node_row['id'].values[0])
        # note that some ids include a large tab before the number comes up ('   1), I may remove those spaces
        network.links_dict[link_key].id = row['id']

    bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                      'alpha': links_df['alpha'],
                                      'beta': links_df['beta'],
                                      'tf': links_df['tf_inrix'],
                                      # 'tf': links_df['tf'],
                                      'k': pd.to_numeric(links_df['k'], errors='coerce', downcast='float')
                                      })

    # Normalize free flow travel time between 0 and 1
    # bpr_parameters_df['tf'] = pd.DataFrame(
    #     preprocessing.MinMaxScaler().fit_transform(np.array(bpr_parameters_df['tf']).reshape(-1, 1)))

    network.set_bpr_functions(bprdata=bpr_parameters_df)

    # To correct problem with assignment of performance

    return network


def read_OD(**kwargs):
    isl.factory.NetworkGenerator().read_OD(**kwargs)


def sparsify_OD(Q, prop_od_pairs=0):
    indices = np.argwhere(Q > 0)

    sample_num = int(prop_od_pairs * indices.shape[0])

    random_indices = indices[np.random.choice(range(len(indices)), sample_num, replace=False)]

    Q[random_indices[:, 0], random_indices[:, 1]] = 0

    return Q


def read_paths(network, **kwargs) -> None:
    isl.factory.PathsGenerator().read_paths(network=network, **kwargs)


def load_k_shortest_paths(network, k, update_incidence_matrices=False, **kwargs):
    isl.factory.PathsGenerator().load_k_shortest_paths(network=network,
                                                       k=k,
                                                       update_incidence_matrices=False,
                                                       **kwargs)
    if update_incidence_matrices:
        paths_od = network.paths_od
        network.D = network.generate_D(paths_od=paths_od, links=network.links)
        network.M = network.generate_M(paths_od=paths_od)
        network.C = network.generate_C(network.M)
