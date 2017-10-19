# -*- coding: utf-8 -*-
'''
module for processing Oil and Gas SilCam data
'''
import pysilcam.postprocess as sc_pp 
import itertools
import pandas as pd
import numpy as np

def extract_gas(stats, THRESH=0.9):
    ind = np.logical_or((stats['probability_bubble']>stats['probability_oil']),
            (stats['probability_oily_gas']>stats['probability_oil']))

    ind2 = np.logical_or((stats['probability_bubble'] > THRESH),
            (stats['probability_oily_gas'] > THRESH))

    ind = np.logical_and(ind, ind2)

    stats = stats[ind]
    return stats


def extract_oil(stats, THRESH=0.9):
    ind = np.logical_or((stats['probability_oil']>stats['probability_bubble']),
            (stats['probability_oil']>stats['probability_oily_gas']))

    ind2 = (stats['probability_oil'] > THRESH)

    ind = np.logical_and(ind, ind2)

    stats = stats[ind]
    return stats


class rt_stats():

    def __init__(self, settings):
        self.stats = pd.DataFrame
        self.settings = settings

    def update(self):
        # remove data from before the specified window of seconds
        # (settings.PostProcess.window_size)
        self.stats = sc_pp.extract_latest_stats(self.stats,
                self.settings.PostProcess.window_size)

        #extract seperate stats on oil and gas
        self.oil_stats = extract_oil(self.stats)
        self.gas_stats = extract_gas(self.stats)

        #calculate d50
        self.oil_d50 = sc_pp.d50_from_stats(self.oil_stats,
            self.settings.PostProcess)
        self.gas_d50 = sc_pp.d50_from_stats(self.gas_stats,
                self.settings.PostProcess)


def ogdataheader():

    ogheader = 'Y, M, D, h, m, s, '

    bin_mids_um, bin_limits_um = sc_pp.get_size_bins()

    for i in bin_mids_um:
        ogheader += str(i) + ', '
        #print(str(i) + ', ')
                
    ogheader += 'd50, ProcessedParticles'

    return ogheader


def cat_data(timestamp, stats, settings):
    dias, vd = sc_pp.vd_from_stats(stats, settings.PostProcess)
    d50 = sc_pp.d50_from_vd(vd, dias)

    data = [[timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond /
            1e6],
            vd, [d50, len(stats)]]
    data = list(itertools.chain.from_iterable(data))

    return data
