# -*- coding: utf-8 -*-
'''
Module for exporting particle images of specified characteristics
'''

from pysilcam.process import extract_roi
import matplotlib.pyplot as plt
import time
import numpy as np
import imageio
import os

def export_particles(imc,timestamp,stats,settings):

    inds = np.argwhere((settings.PostProcess.pix_size *
            stats['major_axis_length'])>settings.ExportParticles.min_length)

    extractable_particles = len(inds)
    print('EXTRACTING {0} IMAGES from {1}'.format(extractable_particles,len(stats['major_axis_length'])))
    bbox = np.zeros((4,1),dtype=int)

    filenames = ['not_exported'] * len(stats['major_axis_length'])

    for i in inds:
        bbox[0] = stats.iloc[i]['minr']
        bbox[1] = stats.iloc[i]['minc']
        bbox[2] = stats.iloc[i]['maxr']
        bbox[3] = stats.iloc[i]['maxc']
        roi = extract_roi(imc, bbox[:,0])

        filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

        filename = filename + '-PN' + str(i[0]) + '.tiff'
        imageio.imwrite(os.path.join(settings.ExportParticles.ouputpath,
filename), roi)

        filenames[int(i)] = filename

    return filenames
