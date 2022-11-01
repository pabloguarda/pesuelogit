# =============================================================================
# 1) SETUP
# =============================================================================

# External modules
import ast
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import os
import glob

import warnings

warnings.simplefilter(action="ignore")

# Path management
from pathlib import Path

import itertools

# Get main project directory
main_dir = str(Path(os.path.abspath('')).parents[1])
os.chdir(main_dir)
print('main dir:', main_dir)

# Internal modules
import isuelogit as isl

isl.config.set_main_dir(main_dir)
isl.config.dirs['write_network_data'] = main_dir + "/output/network-data/fresno/"

# =============================================================================
# 2) NETWORK FACTORY
# ============================================================================
network_name = 'Fresno'

# =============================================================================
# a) READ FRESNO LINK DATA
# =============================================================================

# Reader of geospatial and spatio-temporal data
data_reader = isl.etl.DataReader(network_key=network_name, setup_spark=True)

# Read files
links_df, nodes_df = isl.reader.read_fresno_network(folderpath=isl.dirs['Fresno_network'])

# Add link key in dataframe
links_df['link_key'] = [(int(i), int(j), '0') for i, j in zip(links_df['init_node_key'], links_df['term_node_key'])]

# =============================================================================
# a) BUILD NETWORK
# =============================================================================

# Create Network Generator
network_generator = isl.factory.NetworkGenerator()

A = network_generator.generate_adjacency_matrix(links_keys=list(links_df.link_key.values))

fresno_network = \
    network_generator.build_fresno_network(A=A, links_df=links_df, nodes_df=nodes_df, network_name=network_name)

# =============================================================================
# f) OD
# =============================================================================

# - Periods (6 days of 15 minutes each)
data_reader.options['od_periods'] = [1, 2, 3, 4]

# Read OD from raw data
Q = isl.reader.read_fresno_dynamic_od(network=fresno_network,
                                      filepath=isl.dirs['Fresno_network'] + '/SR41.dmd',
                                      periods=data_reader.options['od_periods'])

network_generator.write_OD_matrix(network=fresno_network, sparse=True, overwrite_input=False)

# =============================================================================
# c) LINK FEATURES FROM NETWORK FILE
# =============================================================================

# Extract data on link features from network file
link_features_df = links_df[['link_key', 'id', 'link_type', 'rhoj', 'lane', 'ff_speed', 'length']]

# Attributes
link_features_df['link_type'] = link_features_df['link_type'].apply(lambda x: x.strip())
link_features_df['rhoj'] = pd.to_numeric(link_features_df['rhoj'], errors='coerce', downcast='float')
link_features_df['lane'] = pd.to_numeric(link_features_df['lane'], errors='coerce', downcast='integer')
link_features_df['length'] = pd.to_numeric(link_features_df['length'], errors='coerce', downcast='float')

# Load features data
fresno_network.load_features_data(linkdata=link_features_df, link_key='link_key')

# =============================================================================
# d) LINK PERFORMANCE FUNCTIONS
# =============================================================================

options = {'tt_units': 'minutes'}

# Create two new features
if options['tt_units'] == 'minutes':
    # Weighting by 60 will leave travel time with minutes units, because speeds are originally in per hour units
    tt_factor = 60

if options['tt_units'] == 'seconds':
    tt_factor = 60 * 60

links_df['ff_speed'] = pd.to_numeric(links_df['ff_speed'], errors='coerce', downcast='float')
links_df['ff_traveltime'] = tt_factor * links_df['length'] / links_df['ff_speed']

bpr_parameters_df = pd.DataFrame({'link_key': links_df['link_key'],
                                  'alpha': 0.15,
                                  'beta': 4,
                                  'tf': links_df['ff_traveltime'],
                                  'k': pd.to_numeric(links_df['capacity'], errors='coerce', downcast='float')
                                  })

fresno_network.set_bpr_functions(bprdata=bpr_parameters_df, link_key='link_key')

# =============================================================================
# d) SPATIO-TEMPORAL LINK FEATURES AND TRAFFIC COUNTS
# =============================================================================

isl.config.dirs['input_folder'] = str(Path(os.path.abspath('')).parents[0]) + "/isuelogit/input/"
isl.config.dirs['read_network_data'] = isl.config.dirs['input_folder'] + "network-data/fresno/"

filespath = isl.config.dirs['input_folder'] + "private/Fresno/inrix/speed/by-day"

dates = [os.path.splitext(os.path.basename(file))[0] for file in glob.glob(filespath + '/*.csv')]
# hours = [16, 17, 18]

hours = range(6,21)

lwrlk_only = True

# TODO: select 16, 17, and 18 hours separately and generate files that include a column for the date and the hour

# List file names from external path using parent command

for date, hour in itertools.product(dates, hours):

    data_reader.select_period(date=date, hour=hour)

    # =============================================================================
    # SPATIO-TEMPORAL LINK FEATURES
    # =============================================================================

    spatiotemporal_features_df, spatiotemporal_features_list = data_reader.read_spatiotemporal_data_fresno(
        lwrlk_only=True,
        read_inrix_selected_date_data=True,
        # data_processing={'inrix_segments': False, 'inrix_data': False, 'census': True, 'incidents': False,
        #                  'bus_stops': False, 'streets_intersections': False},
        # When false, all INRIX data from October 2019 and 2020 is read. When true, only data from selected date is read
        network=fresno_network,
        selected_period_incidents={'year': [data_reader.options['selected_year']], },
        # 'month': [1,2,3,4,5,6,7,8,9,10]},
        # 'day_month': [data_reader.options['selected_day_month']]},
        selected_period_inrix={'year': [data_reader.options['selected_year']],
                               'month': [data_reader.options['selected_month']],
                               # 'day_month': [data_reader.options['selected_day_month']],
                               'day_week': [1, 2, 3, 4, 5],
                               'hour': data_reader.options['selected_hour']},
        buffer_size={'inrix': 100, 'bus_stops': 50, 'incidents': 50, 'streets_intersections': 50},
        tt_units=options['tt_units']
    )

    spatiotemporal_features_df['date'] = date
    spatiotemporal_features_df['hour'] = hour

    filepath = isl.config.dirs['write_network_data'] + 'links/' + str(data_reader.options['selected_date']) + '-' + str(
        data_reader.options['selected_hour']) + '-00-00'+ '-fresno-spatiotemporal-link-data.csv'

    spatiotemporal_features_df.to_csv(filepath, sep=',', encoding='utf-8', index=False, float_format='%.3f')

    # Test Reader
    spatiotemporal_features_df = pd.read_csv(filepath)

    fresno_network.load_features_data(spatiotemporal_features_df)

    # =============================================================================
    # d) FREE FLOW TRAVEL TIME OF LINK PERFORMANCE FUNCTIONS
    # =============================================================================

    # TODO: I may remove these lines and do the data processing of free flow travel times later

    for link in fresno_network.links:
        if link.link_type == 'LWRLK' and link.Z_dict['speed_ref_avg'] != 0:
            # Multiplied by 60 so speeds are in minutes
            # link.Z_dict['tf_inrix'] = tt_factor * link.Z_dict['length'] / link.Z_dict['speed_max']
            link.Z_dict['tf_inrix'] = tt_factor * link.Z_dict['length'] / link.Z_dict['speed_ref_avg']
        #             link.bpr.tf = tt_factor * link.Z_dict['length'] / link.Z_dict['speed_ref_avg']
        else:
            link.Z_dict['tf_inrix'] = link.bpr.tf
            # = links_df[links_df['link_key'].astype(str) == str(link.key)]['ff_traveltime']

    # =============================================================================
    # 3c) DATA CURATION
    # =============================================================================

    # a) Imputation to correct for outliers and observations with zero values because no GIS matching
    features_list = ['median_inc', 'intersections', 'incidents', 'bus_stops', 'median_age',
                     'tt_avg', 'tt_sd', 'tt_var', 'tt_cv',
                     'speed_ref_avg', 'speed_avg', 'speed_sd', 'speed_cv']

    for feature in features_list:
        fresno_network.link_data.feature_imputation(feature=feature, pcts=(0, 100))

    # b) Feature values in "connectors" links
    for key in features_list:
        for link in fresno_network.get_non_regular_links():
            link.Z_dict[key] = 0
    print('Features values of link with types different than "LWRLK" were set to 0')

    # a) Capacity adjustment

    # counts = isl.etl.adjust_counts_by_link_capacity(network = fresno_network, counts = counts)

    # b) Outliers

    # isl.etl.remove_outliers_fresno(fresno_network)

    # =============================================================================
    # 2.2) TRAFFIC COUNTS
    # =============================================================================

    # ii) Read data from PEMS count and perform matching GIS operations to combine station shapefiles

    date_pathname = data_reader.options['selected_date'].replace('-', '_')

    path_pems_counts = isl.dirs['input_folder'] + 'public/pems/counts/data/' + \
                       'd06_text_station_5min_' + date_pathname + '.txt.gz'

    # Read and match count data from a given period

    # Duration is set at 2 because the simulation time for the OD matrix was set at that value
    count_interval_df \
        = data_reader.read_pems_counts_by_period(
        filepath=path_pems_counts,
        selected_period={'hour': data_reader.options['selected_hour'],
                         'duration': int(len(data_reader.options['od_periods']) * 15)})

    # Generate a masked vector that fill out count values with no observations with nan
    counts = isl.etl.generate_fresno_pems_counts(links=fresno_network.links
                                                 , data=count_interval_df
                                                 # , flow_attribute='flow_total'
                                                 # , flow_attribute = 'flow_total_lane_1')
                                                 , flow_attribute='flow_total_lane'
                                                 , flow_factor=1  # 0.1
                                                 )
    # Write counts in csv
    filepath = isl.config.dirs['write_network_data'] + 'links/' + str(data_reader.options['selected_date']) \
               + '-' + str(data_reader.options['selected_hour'])+ '-00-00' + '-fresno-link-counts.csv'

    counts_df = pd.DataFrame({'link_key': counts.keys(),
                              'counts': counts.values(),
                              'pems_ids': [link.pems_stations_ids for link in fresno_network.links],
                              'date': date,
                              'hour': hour,
                              }
                             )

    counts_df.to_csv(filepath, sep=',', encoding='utf-8', index=False, float_format='%.3f')

    # Read counts from csv
    counts_df = pd.read_csv(filepath, converters={"link_key": ast.literal_eval})

    counts = dict(zip(counts_df['link_key'].values, counts_df['counts'].values))

    # Load counts
    fresno_network.load_traffic_counts(counts=counts)

    # # =============================================================================
    # # c) WRITE FILE WITH THE ADJUSTED GIS POSITIONS OF NODES
    # # =============================================================================
    #
    # # Update coordinates of raw file
    # nodes_df['lon'] = [node.position.get_xy()[0] for node in fresno_network.nodes]
    # nodes_df['lat'] = [node.position.get_xy()[1] for node in fresno_network.nodes]
    #
    # nodes_df.to_csv(isl.dirs['output_folder'] + 'fresno/network-data/nodes/' + 'fresno-nodes-data.csv',
    #                 sep=',', encoding='utf-8', index=False, float_format='%.3f')
    #
    # isl.geographer.write_nodes_gdf(nodes_df,
    #                                folderpath=isl.config.dirs['output_folder'] + 'gis/Fresno/network/nodes',
    #                                filename='Fresno_nodes.shp')

    # =============================================================================
    # d) WRITE FILE WITH LINK FEATURES AND COUNTS
    # =============================================================================

    summary_table_links_df = isl.descriptive_statistics.summary_table_links(links=fresno_network.links)

    summary_table_links_df['date'] = date
    summary_table_links_df['hour'] = hour

    summary_table_links_df.to_csv(isl.config.dirs['write_network_data'] + 'links/'
                                  + str(data_reader.options['selected_date']) \
                                  + '-' + str(data_reader.options['selected_hour']) + '-00-00'+ '-fresno-link-data.csv',
                                  sep=',', encoding='utf-8', index=False, float_format='%.3f')

    # folderpath = isl.dirs['output_folder'] + 'gis/Fresno/features/' + data_reader.options['selected_date']
    #
    # isl.geographer.write_links_features_map_shp(network=fresno_network,
    #                                             folderpath=folderpath,
    #                                             filename='link_features_' + data_reader.options[
    #                                                 'selected_date'] + '.shp'
    #                                             )

sys.exit()

# =============================================================================
# g) PATHS
# =============================================================================

# Create path generator
paths_generator = isl.factory.PathsGenerator()

# Generate and Load paths in network
paths_generator.load_k_shortest_paths(network=fresno_network, k=3)
#
# Write paths and incident matrices
paths_generator.write_paths(network=fresno_network, overwrite_input=False)

network_generator.write_incidence_matrices(network=fresno_network,
                                           matrices={'sparse_C': True, 'sparse_D': True, 'sparse_M': True},
                                           overwrite_input=False)

paths_generator.read_paths(network=fresno_network, update_incidence_matrices=True)

# =============================================================================
# Descriptive statistics
# =============================================================================

### Correlation between features

# features_dict = {'tf': 'travel time', 'tt_cv': 'cv of travel time',
#                 'median_inc': 'median income', 'incidents': 'incidents', 'bus_stops': 'bus stops',
#                 'intersections': 'intersections'}

features_dict = {'free_flow_speed': 'free flow speed\n[mi/hr]',
                 'speed_sd': 'standard deviation\nof speed [mi/hr]',
                 # 'speed_avg': 'average speed\n[mi/hr]',
                 'incidents': 'total incidents \n in the year',
                 'bus_stops': 'number of\nbus stops',
                 'median_inc': 'median income\n[1000 US$/month]',
                 'intersections': 'number of\nintersections', 'counts': 'traffic flow \n[vehicles/hour]'}

summary_table_links_dfs = {}

for date in dates:
    summary_table_links_dfs[date] = pd.read_csv(isl.dirs['output_folder'] + 'network-data/links/'
                                                + str(date) + '-fresno-link-data.csv',
                                                sep=',', encoding='utf-8')

    summary_table_links_dfs[date]['free_flow_speed'] \
        = tt_factor * summary_table_links_dfs[date]['length'] / summary_table_links_dfs[date]['tf_inrix']

    # summary_table_links_df = pd.DataFrame(
    #     preprocessing.MinMaxScaler().fit_transform(summary_table_links_raw_df[features_dict.keys()].values))

    summary_table_links_dfs[date].insert(0, 'date', date)

summary_table_links_df = pd.concat(summary_table_links_dfs.values())

summary_table_links_df.reset_index(inplace=True)

scatter_fig = isl.descriptive_statistics.scatter_plots_features(
    links_df=summary_table_links_df[summary_table_links_df.link_type == 'LWRLK'],
    hue='date',
    features=features_dict,
    normalized=False,
    folder=isl.config.dirs['output_folder'] + 'plots/',
    filename='features_correlations_plot_years.pdf'
)

plt.show()

# scatter_fig.savefig(estimation_reporter.dirs['estimation_folder'] + '/' + 'features_correlations_plot.pdf',
#             pad_inches=0.1, bbox_inches="tight")

### Daily traffic counts in selected pems stations

# TODO: Generate descriptive statistics of counts and travel times between years accross links.

# # data_reader.read_pems_counts_by_period(filepath=path_pems_counts
# #                                                   , selected_period = estimation_options['selected_period_pems_counts'])
#
#
# pems_counts_filepath = isl.dirs['input_folder'] + '/network-data/links/' \
#                             + str(dates[0]) + '-fresno-link-counts' + '.csv'
#
# # Read pems counts
#
# # Load pems ids
# fresno_network.links[0].pems_stations_ids
#
# selected_links_ids_pems_statistics = [link.pems_stations_ids for link in fresno_network.get_observed_links()]
#
# selected_links_ids_pems_statistics = list(np.random.choice(selected_links_ids_pems_statistics, 4, replace=False))
#
# distribution_pems_counts_figure = isl.descriptive_statistics.distribution_pems_counts(
#     filepath=pems_counts_filepath,
#     data_reader=data_reader,
#     selected_period={'year': data_reader.options['selected_year'],
#                      'month': data_reader.options['selected_month'],
#                      'day_month': data_reader.options['selected_day_month'],
#                      'hour': 6, 'duration': 900},
#     selected_links=selected_links_ids_pems_statistics
# )
#
# plt.show()
#
# # distribution_pems_counts_figure.savefig(estimation_reporter.dirs['estimation_folder'] + '/distribution_pems_counts.pdf',
# #                                         pad_inches=0.1, bbox_inches="tight")
#
# plt.show()
