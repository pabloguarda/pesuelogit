import isuelogit as isl
import pandas as pd
from isuelogit.printer import block_output, printIterationBar
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .networks import TransportationNetwork, Equilibrator

def simulate_features(n_days, **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i in range(1, n_days + 1):
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


def simulate_suelogit_data(days: List,
                           features_data: pd.DataFrame,
                           network: TransportationNetwork,
                           equilibrator: Equilibrator,
                           **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i, period in enumerate(days):
        printIterationBar(i + 1, len(days), prefix='days:', length=20)

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

def traveltime_imputation(raw_data: pd.DataFrame):

    # Set free flow travel time based on the minimum of the minimums travel times in the set of measurements
    # collected for each link
    # raw_data['tt_ff'] = raw_data['tf_inrix']
    raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_ff'] == 0), 'tt_ff'] = tf.float64.max

    raw_data = raw_data.assign(tt_ff=raw_data.groupby(['link_key'])['tt_ff'].transform(min))
    # raw_data[raw_data['link_key'] == "(0, 1621, '0')"]

    # Impute links with zero average travel time at a given hr-day with average travel times over days-hrs at that link
    tt_avg_imputation = raw_data[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] != 0)].groupby(['link_key'])[
        'tt_avg'].mean()
    raw_data = pd.merge(raw_data, pd.DataFrame({'link_key': tt_avg_imputation.index,
                                                'tt_avg_imputed': tt_avg_imputation.values}), on='link_key', how='left')

    # indices = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg']==0)].index
    # raw_data.loc[indices,'tt_avg'] = raw_data.loc[indices,'tt_avg_imputed']

    indices = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] == 0)
                           & ~(raw_data['tt_avg_imputed'].isna())].index

    raw_data.loc[indices, 'tt_avg'] = raw_data.loc[indices, 'tt_avg_imputed'].copy()

    return raw_data.copy()

def impute_average_traveltime(data: pd.DataFrame):

    indices = (data['link_type'] == 'LWRLK') & (data['tt_avg'].isna())

    # # We impute the average as 2 times the value of free flow travel time but we may determine this
    # # factor based on a regression or correlation
    factor_ff_to_avg = 2
    data.loc[indices, 'tt_avg'] = factor_ff_to_avg*data.loc[indices, 'tt_ff']

    return data


def data_curation(raw_data: pd.DataFrame):

    raw_data.loc[raw_data['counts'] <= 0, "counts"] = np.nan

    # Replace free flow travel times with nans
    indices = (raw_data['link_type'] == 'LWRLK') & (raw_data['tt_ff'].isna())

    # with travel time reported in original nework files
    # raw_data.loc[indices, 'tt_ff'] = raw_data.loc[indices, 'tf']
    #Or alternatively, use average free flow speed:
    average_reference_speed = raw_data[raw_data['speed_ref_avg']>0]['speed_ref_avg'].mean()
    raw_data.loc[indices, 'tt_ff'] = raw_data.loc[indices, 'length']/average_reference_speed

    # raw_data = traveltime_imputation(raw_data)

    raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_ff'] == 0), 'tt_ff'] \
        = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_ff'] != 0), 'tt_ff'].mean()

    raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] == 0), 'tt_avg'] \
        = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] != 0), 'tt_avg'].mean()

    #raw_data[['tt_ff', 'tt_avg', 'tt_avg_imputed', 'link_type']]#

    # Travel time average cannot be lower than free flow travel time or to have nan entries
    indices = (raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg']< raw_data['tt_ff'])
    # raw_data.loc[indices,'tt_avg'] = float('nan')
    # alternatively, we reduce the average travel times in the conflicting observations to match the free flow travel time
    raw_data.loc[indices, 'tt_avg'] = raw_data.loc[indices, 'tt_ff']

    # raw_data = impute_average_traveltime(raw_data)

    return raw_data