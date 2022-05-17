import isuelogit as isl
import pandas as pd
from isuelogit.printer import block_output, printIterationBar
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .aesue import TNetwork, Equilibrator

def simulate_features(n_periods, **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i in range(1, n_periods + 1):
        df_day = linkdata_generator.simulate_features(**kwargs)
        df_day.insert(0, 'period', i)
        df_list.append(df_day)

    df = pd.concat(df_list)

    return df


def convert_multiperiod_df_to_tensor(df, n_days, n_links, features, n_hours=1):
    '''
    Convert to a tensor of dimensions (n_days, n_hours, n_links, n_features).
    df is a dataframe that contains the feature data
    '''

    return tf.constant(np.array(df[features]).reshape(n_days, n_hours, n_links, len(features)))


def simulate_features_tensor(**kwargs):
    return convert_multiperiod_df_to_tensor(df=simulate_features(**kwargs),
                                            n_links=len(kwargs['links']),
                                            features=kwargs['features_Z'],
                                            n_days=kwargs['n_days']
                                            )


def simulate_suelogit_data(periods: List,
                           features_data: pd.DataFrame,
                           network: TNetwork,
                           equilibrator: Equilibrator,
                           **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i, period in enumerate(periods):
        printIterationBar(i + 1, len(periods), prefix='periods:', length=20)

        # linkdata_generator.simulate_features(**kwargs)
        df_period = features_data[features_data.period == period]

        network.load_features_data(linkdata=df_period)

        with block_output(show_stdout=False, show_stderr=False):
            counts, _ = linkdata_generator.simulate_counts(
                network=network,
                equilibrator=equilibrator,
                noise_params={'mu_x': 0, 'sd_x': 0},
                coverage=1)

        network.load_traffic_counts(counts=counts)

        df_period['traveltime'] = [link.true_traveltime for link in network.links]

        df_period['counts'] = network.observed_counts_vector

        df_list.append(df_period)

    df = pd.concat(df_list)

    return df


def get_design_tensor(Z: pd.DataFrame = None,
                      y: pd.DataFrame = None,
                      **kwargs) -> tf.Tensor:
    """
    return tensor with dimensions (n_days, n_links, 1+n_features)
    """

    if Z is None:
        df = y
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
    elif y is None:
        df = Z
    else:
        df = pd.concat([y, Z], axis=1)

    return convert_multiperiod_df_to_tensor(df=df, features=df.columns, **kwargs)


def get_y_tensor(y: pd.DataFrame, **kwargs):
    return convert_multiperiod_df_to_tensor(y, features=y.columns, **kwargs)