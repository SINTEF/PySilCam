# -*- coding: utf-8 -*-
'''
module for processing Oil and Gas SilCam data
'''
import pysilcam.postprocess as sc_pp 
import itertools


def ogdataheader():

    ogheader = 'Y, M, D, h, m, s, '

    bin_mids_um, bin_limits_um = sc_pp.get_size_bins()

    for i in bin_mids_um:
        ogheader += str(i) + ', '
        #print(str(i) + ', ')
                
    ogheader += 'd50, ProcessedParticles'

    return ogheader


def cat_data(timestamp, stats, settings):
    timestr = timestamp.strftime('%Y, %m, %d, %H, %M, %S.%f')
    dias, vd = sc_pp.vd_from_stats(stats, settings.PostProcess)
    d50 = sc_pp.d50_from_vd(vd, dias)

    data = [[timestamp.year, timestamp.month, timestamp.day,
            timestamp.hour, timestamp.minute, timestamp.second + timestamp.microsecond /
            1e6],
            vd, [d50, len(stats)]]
    data = list(itertools.chain.from_iterable(data))

    return data
