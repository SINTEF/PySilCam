# -*- coding: utf-8 -*-
'''
module for processing Oil and Gas SilCam data
'''
import pysilcam.postprocess as sc_pp
from pysilcam.config import PySilcamSettings
import itertools
import pandas as pd
import numpy as np
import os
import http.server
import socketserver
from multiprocessing import Process, Queue
import serial
import serial.tools.list_ports
import glob
import sys
from tqdm import tqdm
import logging
from matplotlib import colors
import cmocean
import matplotlib.pyplot as plt

solidityThresh = 0.95
logger = logging.getLogger(__name__)

def getListPortCom():
    try:
        if sys.platform.startswith('win'):
            com_list = [comport.device for comport in serial.tools.list_ports.comports()]
        elif sys.platform.startswith('linux'):
            com_list = glob.glob('/dev/tty[A-Za-z]*')
    except AttributeError:
        com_list = []

    return com_list


def extract_gas(stats, THRESH=0.85):
    ma = stats['minor_axis_length'] / stats['major_axis_length']
    stats = stats[ma > 0.3]  # cannot have a deformation more than 0.3
    stats = stats[stats['solidity'] > solidityThresh]
    ind = np.logical_or((stats['probability_bubble'] > stats['probability_oil']),
                        (stats['probability_oily_gas'] > stats['probability_oil']))

    # ind2 = np.logical_or((stats['probability_bubble'] > THRESH),
    # (stats['probability_oily_gas'] > THRESH))

    ind2 = stats['probability_bubble'] > THRESH

    ind = np.logical_and(ind, ind2)

    stats = stats[ind]
    return stats


def extract_oil(stats, THRESH=0.85):
    ma = stats['minor_axis_length'] / stats['major_axis_length']
    stats = stats[ma > 0.3]  # cannot have a deformation more than 0.3
    stats = stats[stats['solidity'] > solidityThresh]
    ind = np.logical_or((stats['probability_oil'] > stats['probability_bubble']),
                        (stats['probability_oil'] > stats['probability_oily_gas']))

    ind2 = (stats['probability_oil'] > THRESH)

    ind = np.logical_and(ind, ind2)

    stats = stats[ind]
    return stats


def gor_timeseries(stats, settings):
    from tqdm import tqdm

    u = stats['timestamp'].unique()
    td = pd.to_timedelta('00:00:' + str(settings.window_size / 2.))

    sample_volume = sc_pp.get_sample_volume(settings.pix_size, path_length=settings.path_length)

    gor = []
    time = []

    for t in tqdm(u):
        dt = pd.to_datetime(t)
        stats_ = stats[
            (pd.to_datetime(stats['timestamp']) < (dt + td)) & (pd.to_datetime(stats['timestamp']) > (dt - td))]

        oilstats = extract_oil(stats_)
        dias, vd_oil = sc_pp.vd_from_stats(oilstats, settings)
        nims = sc_pp.count_images_in_stats(oilstats)
        sv = sample_volume * nims
        vd_oil /= sv

        gasstats = extract_gas(stats_)
        dias, vd_gas = sc_pp.vd_from_stats(gasstats, settings)
        nims = sc_pp.count_images_in_stats(gasstats)
        sv = sample_volume * nims
        vd_gas /= sv

        gor_ = sum(vd_gas) / sum(vd_oil)

        time.append(pd.to_datetime(t))
        gor.append(gor_)

    if (len(gor) == 0) or (np.isnan(max(gor))):
        gor = np.nan
        time = np.nan

    return gor, time


class rt_stats():
    '''
    Class for maintining realtime statistics
    '''

    def __init__(self, settings):
        self.stats = pd.DataFrame
        self.settings = settings
        self.dias = []
        self.vd_oil = []
        self.vd_gas = []
        self.gor = np.nan
        self.oil_d50 = np.nan
        self.gas_d50 = np.nan
        self.saturation = np.nan

    def update(self):
        '''
        Updates the rt_stats to remove data from before the specified window of seconds
        given in the config ini file, here: settings.PostProcess.window_size
        '''
        self.stats = sc_pp.extract_latest_stats(self.stats,
                                                self.settings.PostProcess.window_size)

        # extract seperate stats on oil and gas
        self.oil_stats = extract_oil(self.stats)
        self.gas_stats = extract_gas(self.stats)

        # calculate d50
        self.oil_d50 = sc_pp.d50_from_stats(self.oil_stats,
                                            self.settings.PostProcess)
        self.gas_d50 = sc_pp.d50_from_stats(self.gas_stats,
                                            self.settings.PostProcess)

        # First calculate volume distributsion in units of volume concentration per sample volume
        self.dias, self.vd_oil = sc_pp.vd_from_stats(self.oil_stats,
                                                     self.settings.PostProcess)
        self.dias, self.vd_gas = sc_pp.vd_from_stats(self.gas_stats,
                                                     self.settings.PostProcess)

        # Then calculate correct concentrations for re-scaling the volume distributions:
        nc, volume_concentration_oil, sample_volume, junge = sc_pp.nc_vc_from_stats(self.stats, self.settings.PostProcess,
                                                                                    oilgas=sc_pp.outputPartType.oil)
        # re-scale the volume distribution to the correct oil concentraiton
        self.vd_oil = (self.vd_oil / np.sum(self.vd_oil)) * volume_concentration_oil
        nc, volume_concentration_gas, sample_volume, junge = sc_pp.nc_vc_from_stats(self.stats, self.settings.PostProcess,
                                                                                    oilgas=sc_pp.outputPartType.gas)
        # re-scale the volume distribution to the correct gas concentraiton
        self.vd_gas = (self.vd_gas / np.sum(self.vd_gas)) * volume_concentration_gas

        if (volume_concentration_oil + volume_concentration_gas) == 0:
            self.gor = np.nan
        else:
            self.gor = np.float64(volume_concentration_gas / (volume_concentration_oil + volume_concentration_gas)) * 100

        self.saturation = np.max(self.stats.saturation)

    def to_csv(self, filename):
        '''
        Writes the rt_stats data to a csv file
        
        Args:
            filename (str) : filename of the csv file to write to
        '''
        df = pd.DataFrame()
        df['Oil d50[um]'] = [self.oil_d50]
        df['Gas d50[um]'] = [self.gas_d50]
        df['saturation [%]'] = [self.saturation]
        df.to_csv(filename, index=False, mode='w')  # do not append to this file


class ServerThread(Process):
    '''
    Class for managing http server for sharing realtime data
    '''

    def __init__(self, ip):
        '''
        Setup the server
        
        Args:
            ip (str) : string defining the local ip address of the machine that will run this function
        '''
        super(ServerThread, self).__init__()
        self.ip = ip
        self.go()

    def run(self):
        '''
        Start the server on port 8000
        '''
        PORT = 8000
        # address = '192.168.1.2'
        Handler = http.server.SimpleHTTPRequestHandler
        with socketserver.TCPServer((self.ip, PORT), Handler) as httpd:
            logger.info("serving at port: {0}".format(PORT))
            httpd.serve_forever()

    def go(self):
        self.start()


def ogdataheader():
    '''
    Possibly a redundant function....
    '''
    ogheader = 'Y, M, D, h, m, s, '

    bin_mids_um, bin_limits_um = sc_pp.get_size_bins()

    for i in bin_mids_um:
        ogheader += str(i) + ', '
        # print(str(i) + ', ')

    ogheader += 'd50, ProcessedParticles'

    return ogheader


def cat_data(timestamp, stats, settings):
    '''
    Possibly a redundant function....
    '''
    dias, vd = sc_pp.vd_from_stats(stats, settings.PostProcess)
    d50 = sc_pp.d50_from_vd(vd, dias)

    data = [[timestamp.year, timestamp.month, timestamp.day,
             timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond /
             1e6],
            vd, [d50, len(stats)]]
    data = list(itertools.chain.from_iterable(data))

    return data


def cat_data_pj(timestamp, vd, d50, nparts):
    '''
    cat data into PJ-readable format (readable by the old matlab SummaryPlot exe)
    '''
    timestamp = pd.to_datetime(timestamp)

    data = [[timestamp.year, timestamp.month, timestamp.day,
             timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond /
             1e6],
            vd, [d50, nparts]]
    data = list(itertools.chain.from_iterable(data))

    return data


def convert_to_pj_format(stats_file, config_file):
    '''converts stats files into a total, and gas-only time-series csvfile which can be read by the old matlab
    SummaryPlot exe'''

    settings = PySilcamSettings(config_file)
    logger.info('Loading stats....')
    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    base_name = stats_file.replace('-STATS.h5', '-PJ.csv')
    gas_name = base_name.replace('-PJ.csv', '-PJ-GAS.csv')

    ogdatafile = DataLogger(base_name, ogdataheader())
    ogdatafile_gas = DataLogger(gas_name, ogdataheader())

    stats['timestamp'] = pd.to_datetime(stats['timestamp'])
    u = stats['timestamp'].unique()
    sample_volume = sc_pp.get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    logger.info('Analysing time-series')
    for s in tqdm(u):
        substats = stats[stats['timestamp'] == s]
        nims = sc_pp.count_images_in_stats(substats)
        sv = sample_volume * nims

        oil = extract_oil(substats)
        dias, vd_oil = sc_pp.vd_from_stats(oil, settings.PostProcess)
        vd_oil /= sv

        gas = extract_gas(substats)
        dias, vd_gas = sc_pp.vd_from_stats(gas, settings.PostProcess)
        vd_gas /= sv
        d50_gas = sc_pp.d50_from_vd(vd_gas, dias)

        vd_total = vd_oil + vd_gas
        d50_total = sc_pp.d50_from_vd(vd_total, dias)

        data_total = cat_data_pj(s, vd_total, d50_total, len(oil) + len(gas))
        ogdatafile.append_data(data_total)

        data_gas = cat_data_pj(s, vd_gas, d50_gas, len(gas))
        ogdatafile_gas.append_data(data_gas)

    logger.info('  OK.')

    logger.info('Deleting header!')
    with open(base_name, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(base_name, 'w') as fout:
        fout.writelines(data[1:])
    with open(gas_name, 'r') as fin:
        data = fin.read().splitlines(True)
    with open(gas_name, 'w') as fout:
        fout.writelines(data[1:])
    logger.info('Conversion complete.')


def realtime_summary(statsfile, config_file):
    if not os.path.isfile(statsfile):
        print(statsfile, 'nonexistent')
        return
    plt.ion()

    stats = pd.read_hdf(statsfile, 'ParticleStats/stats')
    settings = PySilcamSettings(config_file)

    nims = sc_pp.count_images_in_stats(stats)
    print(len(stats), 'particles in', nims, 'images')

    if nims < 2:
        return

    timeseries = sc_pp.make_timeseries_vd(stats, settings)

    dias = timeseries.iloc[:, 0:52].columns.values
    vdts = timeseries.iloc[:, 0:52].values

    plt.gca().cla()
    pcm = plt.pcolormesh(timeseries['Time'], dias, vdts.T, cmap=cmocean.cm.turbid, norm=colors.LogNorm())
    plt.yscale('log')
    plt.ylabel('Equivalent Circular Diameter [um]')
    plt.ylim(50, 10000)
    plt.xlabel('Time')
    plt.title('Last data: ' + str(max(timeseries['Time'])))
    plt.draw()
    plt.pause(0.01)


def gaussian_fit(xdata, ydata):
    mu = np.sum(xdata * ydata) / np.sum(ydata)
    sigma = np.sqrt(np.abs(np.sum((xdata - mu) ** 2 * ydata) / np.sum(ydata)))
    return mu, sigma


def gaussian(x, mu, sig):
    y = 1 / np.sqrt(2 * sig * sig * np.pi) * np.exp(-(x - mu) * (x - mu) / (2 * sig * sig))  #
    return y


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


def cos_check(dias, vd):
    mu, sig = gaussian_fit(np.arange(0, len(dias)), vd)

    y = gaussian(np.arange(0, len(dias)), mu, sig)
    y /= max(y)
    y *= max(vd)

    cos = cosine_similarity(y, vd)
    return cos


class DataLogger:
    def __init__(self, filename, header):
        self.filename = filename

        # Generate file with header
        with open(filename, 'w') as fh:
            fh.write(header + '\n')

    def append_data(self, data):
        # Append data line to file
        with open(self.filename, 'a') as fh:
            # Generate data line string for data list
            line = ','.join([str(el) for el in data])

            # Append line to file
            fh.write(line + '\n')
