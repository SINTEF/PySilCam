# -*- coding: utf-8 -*-
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pysilcam.postprocess as scpp
from pysilcam.config import PySilcamSettings


# This file can be run as a stand-alone file after having run
# python setup.py develop
# from the base directory of the silcam repository.

common_path = 'Z:\\302004868_COAP_Phase1\\Data\\Thor\\20200528'  # Path to folder containing merged NEPTUS logs
LOGS_PATH = "Neptus\\merge\\mra\\csv"  # Information on mission date used for locating files and understanding times
FOLDER = 'proc\\test\\test'  # Output folder, with start output names
INI_FILE = "config_crop_thresh97.ini"
SILCAM_DATAFILE = "proc\\SilCam_thresh97-STATS.csv"
CROP_STATS = (500, 500, 1750, 1750)  # The bounds of the cropping, if not being used give: None

LOGS_PATH = os.path.join(common_path, LOGS_PATH)
FOLDER = os.path.join(common_path, FOLDER)
INI_FILE = os.path.join(common_path, INI_FILE)
SILCAM_DATAFILE = os.path.join(common_path, SILCAM_DATAFILE)


def fix_ctd_time(ctd, hour_delay=0):
    """
    Reformats the timestamp column to a readable format in a new column
    """
    new_col_name = 'Time'
    output_time_format = "%Y-%m-%d %H:%M:%S.%f"

    ctd[new_col_name] = ctd.apply(
        lambda x: (datetime.utcfromtimestamp(x['timestamp']) + timedelta(hours=hour_delay)).strftime(output_time_format),
        axis=1)
    return ctd


def add_latlon_to_stats(stats, time, lat, lon):
    '''This is approximate because lat and lon are treated independently!!!!'''
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Latitude'] = np.interp(np.float64(sctime), np.float64(time), lat)
    stats['Longitude'] = np.interp(np.float64(sctime), np.float64(time), lon)
    return stats


def depth_timeseries_plot(ctd):
    plt.plot(ctd['Time'], ctd[' depth'], 'k.', markersize=4)
    plt.ylim(30, 0)
    plt.ylabel('Depth [m]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=20)
    plt.xticks(horizontalalignment='right')


def nd_plot(stats, settings):
    '''Particle number distribution plot'''
    dias, nd = scpp.nd_from_stats_scaled(stats, settings.PostProcess)
    plt.plot(dias, nd, 'k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(100, 10000)
    plt.xlabel('Equiv. diam. [um]')
    plt.ylabel('Number conc. [#/Arb.Vol./um]')


def nc_timeseries(stats, settings):
    ''' Calculates time series of d50 from stats

    Args:
        stats               : pandas DataFrame of particle statistics
        settings            : PySilCam settings

    Returns:
        nc                  : time series of number concentration
        time                : timestamps for concentration values

    '''

    from tqdm import tqdm

    stats.sort_values(by=['timestamp'], inplace=True)

    td = pd.to_timedelta('00:00:' + str(settings.window_size / 2.))
    nc = []
    time = []
    depth = []
    lat = []
    lon = []

    sample_volume = scpp.get_sample_volume(settings.pix_size, path_length=settings.path_length)

    u = pd.to_datetime(stats['timestamp']).unique()

    for t in tqdm(u):
        dt = pd.to_datetime(t)
        stats_ = stats[
            (pd.to_datetime(stats['timestamp']) < (dt + td))
            & (pd.to_datetime(stats['timestamp']) > (dt - td))]
        nc_ = len(stats_) / sample_volume
        nc.append(nc_)
        time.append(t)
        # all the location values below should be the same - use of mean is questionable
        depth.append(np.mean(stats_['Depth']))
        lat.append(np.mean(stats_['Latitude']))
        lon.append(np.mean(stats_['Longitude']))

    if len(time) == 0:
        nc = np.nan
        time = np.nan

    timeseries = pd.DataFrame(columns=[
        'Time', 'Depth [m]', 'Latitude', 'Longitude', 'Number Concentration [#/L]'])
    timeseries['Time'] = time
    timeseries['Depth [m]'] = depth
    timeseries['Latitude'] = lat
    timeseries['Longitude'] = lon
    timeseries['Number Concentration [#/L]'] = nc

    return timeseries


def neptus_csv_ctd(csv_path, start_time=None, end_time=None):
    '''
    takes exported neptus csv files and merges them into a DataFrame of 1-second time bins

    :param csv_path: folder location of exported neptus csv files
                    requires the following files to be exported from Neptus:
                     Depth, WaterDensity, Salinity, Temperature, Turbidity, Chlorophyll, EstimatedState
    :param start_time: optional timestamp of where to start
    :param end_time: optional timestamp of where to stop
    :return: pandas DataFrame of ctd data
    '''
    def date_fix(date):
        return pd.to_datetime(date, unit='s')

    depth = pd.read_csv(os.path.join(csv_path, 'Depth.csv'),
                        names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                        parse_dates=['timestamp'], date_parser=date_fix)
    depth = depth[depth[' entity '] == ' SmartX']

    density = pd.read_csv(os.path.join(csv_path, 'WaterDensity.csv'),
                          names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                          parse_dates=['timestamp'], date_parser=date_fix)
    density = density[density[' entity '] == ' SmartX']

    salinity = pd.read_csv(os.path.join(csv_path, 'Salinity.csv'),
                           names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                           parse_dates=['timestamp'], date_parser=date_fix)
    salinity = salinity[salinity[' entity '] == ' SmartX']

    temperature = pd.read_csv(os.path.join(csv_path, 'Temperature.csv'),
                              names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                              parse_dates=['timestamp'], date_parser=date_fix)
    temperature = temperature[temperature[' entity '] == ' SmartX']

    turbidity = pd.read_csv(os.path.join(csv_path, 'Turbidity.csv'),
                            names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                            parse_dates=['timestamp'], date_parser=date_fix)
    turbidity = turbidity[turbidity[' entity '] == ' SmartX']

    chlorophyll = pd.read_csv(os.path.join(csv_path, 'Chlorophyll.csv'),
                              names=['timestamp', 'system', ' entity ', 'value'], skiprows=1,
                              parse_dates=['timestamp'], date_parser=date_fix)
    chlorophyll = chlorophyll[chlorophyll[' entity '] == ' SmartX']

    estimated_state = pd.read_csv(os.path.join(csv_path, 'EstimatedState.csv'),
                                  parse_dates=['timestamp'], date_parser=date_fix)
    lat_deg = estimated_state[' lat (rad)'].apply(np.rad2deg)
    lon_deg = estimated_state[' lon (rad)'].apply(np.rad2deg)

    if start_time is None:
        start_time = min(depth['timestamp'])
        print(start_time)
    if end_time is None:
        end_time = max(depth['timestamp'])
    time_bins = pd.date_range(start=start_time, end=end_time, freq='S')
    time_mids = time_bins[0:-1] + pd.to_timedelta((time_bins[1] - time_bins[0]) / 2)

    depth_ = np.interp(np.float64(time_mids), np.float64(depth['timestamp']), depth['value'])
    density_ = np.interp(np.float64(time_mids), np.float64(density['timestamp']), density['value'])
    salinity_ = np.interp(np.float64(time_mids), np.float64(salinity['timestamp']), salinity['value'])
    temperature_ = np.interp(np.float64(time_mids), np.float64(temperature['timestamp']), temperature['value'])
    turbidity_ = np.interp(np.float64(time_mids), np.float64(turbidity['timestamp']), turbidity['value'])
    chlorophyll_ = np.interp(np.float64(time_mids), np.float64(chlorophyll['timestamp']), chlorophyll['value'])
    lat_deg_ = np.interp(np.float64(time_mids), np.float64(estimated_state['timestamp']), lat_deg)
    lon_deg_ = np.interp(np.float64(time_mids), np.float64(estimated_state['timestamp']), lon_deg)

    # not possible to have negative turbidity. convert to relative min(NTU)
    turbidity_ -= min(turbidity_)

    ctd = pd.DataFrame()
    ctd['timestamp'] = time_mids
    ctd['depth'] = depth_
    ctd['salinity'] = salinity_
    ctd['density'] = density_
    ctd['temperature'] = temperature_
    ctd['turbidity'] = turbidity_
    ctd['chlorophyll'] = chlorophyll_
    ctd['lat_deg'] = lat_deg_
    ctd['lon_deg'] = lon_deg_

    return ctd


def add_neptus_to_stats(stats, ctd):
    '''
    takes a silcam STATS DataFrame and a neptus CTD DataFrame (use neptus_csv_ctd() to get this),
    and interpolates the ctd data into the STATS. This gives ctd, chl, turbidity and potition data for every particle.

    :return: a modified STATS DataFrame
    '''
    sctime = pd.to_datetime(stats['timestamp'])

    stats = add_latlon_to_stats(stats, pd.to_datetime(ctd['timestamp']), ctd['lat_deg'],
                                ctd['lon_deg'])  # merge location data into particle stats

    # interpolate data into the SilCam times
    stats['Depth'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['depth'])
    stats['salinity'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['salinity'])
    stats['density'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['density'])
    stats['temperature'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['temperature'])
    stats['turbidity'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['turbidity'])
    stats['chlorophyll'] = np.interp(
        np.float64(sctime), np.float64(pd.to_datetime(ctd['timestamp'])), ctd['chlorophyll'])
    return stats


if __name__ == "__main__":

    outfilename = FOLDER + '-AUV-STATS.csv'

    settings = PySilcamSettings(INI_FILE)  # load the settings used for processing

    if not os.path.isfile(outfilename):
        print('Loading CSV file')
        # read the ctd data from the exported NEPTUS logs
        ctd = neptus_csv_ctd(LOGS_PATH)

        print('Loading SilCam STATS data')
        stats = pd.read_csv(SILCAM_DATAFILE)  # load the stats file

        # Apply workaround cropping of stats due to small window
        print('Cropping stats')
        stats = scpp.filter_stats(stats, CROP_STATS)

        print('Adding position and ctd data to stats')
        stats = add_neptus_to_stats(stats, ctd)

        print(stats.columns)

        print('Saving', outfilename)
        stats.to_csv(outfilename)
    else:
        print(outfilename, 'already exists')

        print('Loading', outfilename)
        stats = pd.read_csv(outfilename)  # load the stats file

        print('Calculating timeseries:')
        timeseries = nc_timeseries(stats, settings.PostProcess)

        ts_filename = (FOLDER + '-TimeSeries.csv')
        print('Saving timeseries file:', ts_filename)
        timeseries.to_csv(ts_filename, index=False)
