import pysilcam.process as scpr
import pysilcam.postprocess as scpp
import pysilcam.plotting as scplt
from pysilcam.config import load_config, PySilcamSettings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as skiio
import skimage
import seaborn as sns
sns.set_style('ticks')

#os.chdir('/mnt/PDrive/ENTICE/notebooks/')
#print(os.getcwd())

import sys
sys.path.insert(0, '/mnt/PDrive/ENTICE/notebooks/')

from entice import *

def particle_generator():


    print('load ctd')
    ctd_all = load_ctd()
    print(' ok')

    stn = 'STN12'
    mindepth = 1
    maxdepth = 70

    config_file = '/mnt/ARRAY/ENTICE/Data/configs/config.ini'
    stats_file = '/mnt/ARRAY/ENTICE/Data/proc/' + stn + '-STATS.h5'

    time = ctd_all['Date_Time']
    depth = ctd_all['Depth']

    conf = load_config(config_file)
    settings = PySilcamSettings(conf)

    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    stats = scpp.add_depth_to_stats(stats, time, depth)
    print('all stats:', len(stats))

    sstats = stats[(stats['Depth']>mindepth) & (stats['Depth']<maxdepth)]
    print('selected stats:', len(sstats))


    index = 0

    while True:

#        if np.random.choice([0,1]):
        sstats_ = scpp.extract_nth_largest(sstats,settings,n=index)
#        else:
#            sstats_ = scpp.extract_nth_longest(sstats,settings,n=index)
        print(sstats_)

        filename = os.path.join('/mnt/ARRAY/ENTICE/Data/export/',
            sstats_['export name'])

        im = skiio.imread(filename)

        im = scpp.explode_contrast(im)
        im = scpp.bright_norm(im)
        # scplt.show_imc(im)
        # plt.title(selected_stats['export name'] + ('\nDepth:
        # {0:.1f}'.format(selected_stats['depth'])) + 'm\n')

        index += 1

        yield im
