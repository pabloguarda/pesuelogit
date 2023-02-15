import isuelogit as isl
import pandas as pd
from isuelogit.printer import block_output, printIterationBar
import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from .networks import TransportationNetwork, Equilibrator

def simulate_features(n_days, daytoday_variation = False, **kwargs):

    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i in range(1, n_days + 1):

        if i == 1 or daytoday_variation:
            df_day = linkdata_generator.simulate_features(**kwargs)
            df_day.insert(0, 'period', i)
        else:
            df_day = df_day.assign(period = i)

        df_list.append(df_day)

    df = pd.concat(df_list)

    return df


def convert_multiperiod_df_to_tensor(df, n_timepoints: int, n_links: int, features: List[str]):
    '''
    Convert to a tensor of dimensions (n_timepoints, n_links, n_features).
    df is a dataframe that contains the feature data
    '''

    return tf.constant(np.array(df[features]).reshape(n_timepoints, n_links, len(features)))

def temporal_split(X,Y, n_days = None):

    def splitter(A):
        B = A[0:len(A) // 2]
        C = A[len(A) // 2:]
        return (B, C)

    if n_days is None:
        n_days = X.shape[0]

    train_idxs, test_idxs = splitter(range(0, n_days))

    X_train, X_test = X[train_idxs, :, :], X[test_idxs, :, :]
    Y_train, Y_test = Y[train_idxs, :, :, :], Y[test_idxs, :, :, :]

    return X_train, X_test, Y_train, Y_test

def simulate_features_tensor(**kwargs):
    return convert_multiperiod_df_to_tensor(df=simulate_features(**kwargs),
                                            n_links=len(kwargs['links']),
                                            features=kwargs['features_Z'],
                                            n_timepoints=kwargs['n_timepoints']
                                            )


def simulate_suelogit_data(days: List,
                           features_data: pd.DataFrame,
                           network: TransportationNetwork,
                           equilibrator: Equilibrator,
                           coverage = 1,
                           sd_x: float = 0,
                           sd_t: float = 0,
                           daytoday_variation = False,
                           **kwargs):
    linkdata_generator = isl.factory.LinkDataGenerator()

    df_list = []

    for i, period in enumerate(days):
        printIterationBar(i + 1, len(days), prefix='days:', length=20)

        # linkdata_generator.simulate_features(**kwargs)
        df_period = features_data[features_data.period == period]

        network.load_features_data(linkdata=df_period)

        if i == 0 or daytoday_variation:

            with block_output(show_stdout=False, show_stderr=False):
                counts, _ = linkdata_generator.simulate_counts(
                    network=network,
                    equilibrator=equilibrator, #{'mu_x': 0, 'sd_x': 0},
                    coverage=1)

            masked_counts, _ = linkdata_generator.mask_counts_by_coverage(
                original_counts=np.array(list(counts.values()))[:, np.newaxis], coverage=coverage)

        counts_day_true = np.array(list(counts.values()))[:, np.newaxis]
        counts_day_noisy = linkdata_generator.add_error_counts(
            original_counts=masked_counts, sd_x=sd_x)

        df_period['counts'] = counts_day_noisy
        df_period['true_counts'] = counts_day_true

        # Generate true travel times from true counts
        network.load_traffic_counts(counts=dict(zip(counts.keys(),counts_day_true.flatten())))
        df_period['true_traveltime'] =  [link.true_traveltime for link in network.links]

        # Put nan in links where no traffic count data is available
        df_period['traveltime'] = [link.true_traveltime if ~np.isnan(count) else float('nan')
                                        for link,count in zip(network.links, masked_counts)]

        df_period['traveltime'] = linkdata_generator.add_error_counts(
            original_counts=np.array(df_period['traveltime'])[:, np.newaxis], sd_x=sd_t)

        df_list.append(df_period)

    df = pd.concat(df_list)

    # df.groupby('link_key').agg('mean')

    return df




def get_design_tensor(Z: pd.DataFrame = None,
                      y: pd.DataFrame = None,
                      **kwargs) -> tf.Tensor:
    """
    return tensor with dimensions (n_timepoints, n_links, 1+n_features)
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

def add_period_id(df: pd.DataFrame, period_feature: str = None, varname = 'period_id'):

    if period_feature is None:
        df[varname] = 0

    if period_feature is not None:
        periods_keys = dict(zip(sorted(df[period_feature].unique()), range(len(sorted(df[period_feature].unique())))))

        df = df.merge(pd.DataFrame({period_feature: periods_keys.keys(), varname: periods_keys.values()}),
                      left_on=period_feature, right_on=period_feature)

    return df


def get_tensors_by_year(df, features_Z, network) -> Tuple[Dict[str, tf.Tensor],Dict[str, tf.Tensor]]:
    n_links = len(df.link_key.unique())
    X,Y = {}, {}

    for year in sorted(df['year'].unique()):
        df_year = df[df['year'] == year] #.sort_values(['date', 'hour']).copy()

        # n_dates, n_hours = len(df_year.date.unique()), len(df_year.hour.unique())
        #
        # n_timepoints = n_dates * n_hours
        n_timepoints = len(df_year[['date', 'hour']].drop_duplicates())

        # TODO: Add an assert to check the dataframe is properly sorted before reshaping it into a tensor
        df_year['link_key'] = df_year['link_key'].astype(str)
        df_year['link_key'] = pd.Categorical(df_year['link_key'], map(str,list(network.links_dict.keys())))
        df_year = df_year.sort_values(['period','link_key'])

        traveltime_data = get_y_tensor(y=df_year[['tt_avg']], n_links=n_links, n_timepoints=n_timepoints)
        flow_data = get_y_tensor(y=df_year[['counts']], n_links=n_links, n_timepoints=n_timepoints)

        Y[year] = tf.concat([traveltime_data, flow_data], axis=2)

        X[year] = get_design_tensor(Z=df_year[features_Z + ['period_id']], n_links=n_links, n_timepoints=n_timepoints)

    return X, Y
    
    
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

    # Replace values of free flow travel times that had nans
    indices = (raw_data['link_type'] == 'LWRLK') & (raw_data['tt_ff'].isna())

    # with travel time reported in original nework files
    # raw_data.loc[indices, 'tt_ff'] = raw_data.loc[indices, 'tf']
    #Or alternatively, use average free flow speed:
    average_reference_speed = raw_data[raw_data['speed_ref_avg']>0]['speed_ref_avg'].mean()
    raw_data.loc[indices, 'tt_ff'] = raw_data.loc[indices, 'length']/average_reference_speed

    # raw_data = traveltime_imputation(raw_data)

    # Imputation

    for feature in ['tt_ff', 'tt_avg']:
        if feature in raw_data.columns:
            raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data[feature] == 0), feature] \
                = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data[feature] != 0), feature].mean()
            raw_data.loc[(raw_data['link_type'] != 'LWRLK'), feature] = 0

    for feature in ['tt_sd','tt_sd_adj']:
        if feature in raw_data.columns:
            raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data[feature].isna()), feature] \
                = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (~raw_data[feature].isna()), feature].mean()
            raw_data.loc[(raw_data['link_type'] != 'LWRLK'), feature] = 0

    # raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] == 0), 'tt_avg'] \
    #     = raw_data.loc[(raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg'] != 0), 'tt_avg'].mean()

    #raw_data[['tt_ff', 'tt_avg', 'tt_avg_imputed', 'link_type']]#

    # Travel time average cannot be lower than free flow travel time or to have nan entries
    indices = (raw_data['link_type'] == 'LWRLK') & (raw_data['tt_avg']< raw_data['tt_ff'])
    # raw_data.loc[indices,'tt_avg'] = float('nan')
    # alternatively, we reduce the average travel times in the conflicting observations to match the free flow travel time
    raw_data.loc[indices, 'tt_avg'] = raw_data.loc[indices, 'tt_ff']

    # raw_data = impute_average_traveltime(raw_data)

    return raw_data