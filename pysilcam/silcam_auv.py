# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import os

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt

import pysilcam.postprocess as scpp
import pysilcam.plotting as scplt
from pysilcam.config import PySilcamSettings

datapath = '../NEPTUS' # path to folder containing merged NEPTUS logs
FOLDER = '250718' # information on mission date used for locating files and understanding times


def fix_ctd_time(ctd):
    newtime = []
    for t in ctd[' gmt time']:
        newtime.append(pd.to_datetime(FOLDER[0:2] + '-' + FOLDER[2:4] + '-' + FOLDER[4:6] + t))

    ctd['Time'] = newtime
    return ctd

def extract_middle(stats):
    '''
    Temporary cropping solution due to small window in AUV
    '''
    print('initial stats length:', len(stats))
    r = np.array(((stats['maxr'] - stats['minr'])/2) + stats['minr'])
    c = np.array(((stats['maxc'] - stats['minc'])/2) + stats['minc'])

    points = []
    for i in range(len(c)):
        points.append([(r[i], c[i])])

    pts = np.array(points)
    pts = pts.squeeze()

    # plt.plot(pts[:, 0], pts[:, 1], 'k.')
    # plt.axis('equal')

    ll = np.array([500, 500]) # lower-left
    ur = np.array([1750, 1750])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    inbox = pts[inidx]
    print('inbox shape:', inbox.shape)

    stats = stats[inidx]

    # plt.plot(inbox[:,0], inbox[:,1], 'r.')
    # plt.axis('equal')
    
    print('len stats', len(stats))
    return stats


def montager(stats):
    '''
    Wrapper for pysilcam montage maker
    '''

    ## Make montages of processed particle images
    maxlength = 5000000
    minlength = 100
    msize = 2048
    roidir = '../DATA/' + FOLDER + '/export_backup'

    stats = stats[~np.isnan(stats['major_axis_length'])]
    stats = stats[(stats['major_axis_length'] *
            settings.PostProcess.pix_size) < maxlength]
    stats = stats[(stats['major_axis_length'] *
            settings.PostProcess.pix_size) > minlength]
    
    stats = stats[stats['Depth'] > 2] # only look for things below 2m depth

#     stats = stats[stats['probability_other']>0.8] # can filter classified probabilities like this

    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)
    roifiles = scpp.gen_roifiles(stats, auto_scaler=msize)

    montage = scpp.montage_maker(roifiles, roidir, settings.PostProcess.pix_size, msize, eyecandy=True)
    return montage


def add_latlon_to_stats(stats, time, lat, lon):
    '''This is approximate because lat and lon are treated independently!!!!'''
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Latitude'] = np.interp(np.float64(sctime), np.float64(time), lat)
    stats['Longitude'] = np.interp(np.float64(sctime), np.float64(time), lon)
    return stats


def silcam_montage_plot(montage, settings):
    scplt.montage_plot(montage, settings.PostProcess.pix_size)


def depth_timeseries_plot(ctd):
    plt.plot(ctd['Time'], ctd[' depth'],'k.', markersize=4)
    plt.ylim(30, 0)
    plt.ylabel('Depth [m]')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=20)
    plt.xticks(horizontalalignment='right')
    #     plt.gcf().autofmt_xdate()


def nd_plot(stats, settings):
    '''Particle number distribution plot'''
    dias, nd = scpp.nd_from_stats_scaled(stats, settings.PostProcess)
    plt.plot(dias, nd, 'k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(100, 10000)
    plt.xlabel('Equiv. diam. [um]')
    plt.ylabel('Number conc. [#/Arb.Vol./um]')


def map_plot(ctd, request):
    ax = plt.gca()
    gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator(np.arange(9,11,0.05))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

#     ax.set_extent([10.32, 10.45, 63.40, 63.51])
    ax.set_extent([10.3, 10.5, 63.425, 63.5])

    ax.plot(np.array(ctd[' lon (corrected)']), np.array(ctd[' lat (corrected)']),
            'k.', markersize=4, transform=ccrs.Geodetic())
    
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation='vertical')

    ax.add_image(request, 12)


def summary_figure(ctd, montage, stats, settings):
    '''wrapper for above plotting functions'''
    f = plt.figure(figsize=(12, 12))
    
    request = cimgt.StamenTerrain()
    ax1 = plt.subplot(221, projection=request.crs)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    ax4 = plt.subplot(224)

    plt.sca(ax1)
    map_plot(ctd, request)
    plt.title(ctd['Time'].min().strftime('%Y-%m-%d %H:%M') +
              ' - ' +
              ctd['Time'].max().strftime('%Y-%m-%d %H:%M') + 
              '\n' +
              'Max Depth: {:0.0f} [m]'.format(ctd[' depth'].max()) + 
              ' | Raw SilCam images analysed: {:0.0f}'.format(scpp.count_images_in_stats(stats)) + 
              '\n' +
              'Particles analysed: {:0.0f}'.format(len(stats))
             , loc='left')

    plt.sca(ax2)
    depth_timeseries_plot(ctd)

    plt.sca(ax3)
    silcam_montage_plot(montage, settings)

    plt.sca(ax4)
    nd_plot(stats, settings)


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

    td = pd.to_timedelta('00:00:' + str(settings.window_size/2.))
    nc = []
    time = []
    depth = []
    lat = []
    lon = []

    sample_volume = scpp.get_sample_volume(settings.pix_size, path_length=settings.path_length)

    u = pd.to_datetime(stats['timestamp']).unique()

    for t in tqdm(u):
        dt = pd.to_datetime(t)
        stats_ = stats[(pd.to_datetime(stats['timestamp'])<(dt+td)) & (pd.to_datetime(stats['timestamp'])>(dt-td))]
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

    timeseries = pd.DataFrame(columns=['Time','Depth [m]',
        'Latitude','Longitude','Number Concentration [#/L]'])
    timeseries['Time'] = time
    timeseries['Depth [m]'] = depth
    timeseries['Latitude'] = lat
    timeseries['Longitude'] = lon
    timeseries['Number Concentration [#/L]'] = nc

    return timeseries


if __name__ == "__main__":
    
    outfilename = FOLDER + '-AUV-STATS.csv'

    settings = PySilcamSettings('../DATA/config.ini') # load the settings used for processing

    if not os.path.isfile(outfilename):
        print('Loading CSV file')
        # read the ctd data from the exported NEPTUS logs
        ctd = pd.read_csv(os.path.join(datapath, FOLDER + '_merged_logs/mra/csv/CTD.csv'), index_col=False)
        ctd = fix_ctd_time(ctd) # make the ctd time information useable

        SilCamDataFile = ('../DATA/' + FOLDER + '/proc_backup/RAW-STATS.csv')

        print('Loading SilCam STATS data')
        stats = pd.read_csv(SilCamDataFile) # load the stats file

        print('Cropping stats')
        stats = extract_middle(stats) # apply temporary (workaround) cropping of stats due to small window

        print('Adding depth and location to stats')
        stats = scpp.add_depth_to_stats(stats, ctd['Time'], ctd[' depth']) # merge ctd data into particle stats
        stats = add_latlon_to_stats(stats, ctd['Time'], ctd[' lat (corrected)'], ctd[' lon (corrected)']) # merge location data into particle stats
        
        print(stats.columns)

        print('Saving', outfilename)
        stats.to_csv(outfilename)
    else:
        print(outfilename, 'already exists')

        print('Loading', outfilename)
        stats = pd.read_csv(outfilename) # load the stats file

        print('Calculating timeseries:')
        timeseries = nc_timeseries(stats, settings.PostProcess)

        ts_filename = (FOLDER + '-TimeSeries.csv')
        print('Saving timeseries file:', ts_filename)
        timeseries.to_csv(ts_filename, index=False)
