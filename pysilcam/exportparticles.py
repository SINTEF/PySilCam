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
import pysilcam.silcam_classify as sccl
import h5py

def export_particles(imc,timestamp,stats,settings,nnmodel,class_labels):

    inds = np.argwhere(((settings.PostProcess.pix_size *
            stats['major_axis_length'])>settings.ExportParticles.min_length) &
            ((settings.PostProcess.pix_size * stats['minor_axis_length'])>2))

    extractable_particles = len(inds)
    print('EXTRACTING {0} IMAGES from {1}'.format(extractable_particles,len(stats['major_axis_length'])))
    bbox = np.zeros((4,1),dtype=int)

    if settings.ExportParticles.export_images:
        filenames = ['not_exported'] * len(stats['major_axis_length'])

    if settings.NNClassify.enable:
        predictions = np.zeros((len(stats['major_axis_length']),
            len(class_labels)),
            dtype='float64')
        predictions *= np.nan

    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')
    HDF5File = h5py.File(os.path.join(settings.ExportParticles.ouputpath, filename + ".h5"), "w")

    for i in inds:
        bbox[0] = stats.iloc[i]['minr']
        bbox[1] = stats.iloc[i]['minc']
        bbox[2] = stats.iloc[i]['maxr']
        bbox[3] = stats.iloc[i]['maxc']
        roi = extract_roi(imc, bbox[:,0])

        if settings.ExportParticles.export_images:
            filenames[int(i)] = filename + '-PN' + str(i[0])
            dset = HDF5File.create_dataset('PN' + str(i[0]), data = roi)

        if settings.NNClassify.enable:
            prediction = sccl.predict(roi, nnmodel)
            predictions[int(i),:] = prediction[0]

    HDF5File.close()
    
    if settings.NNClassify.enable:
        for n,c in enumerate(class_labels):
            stats['probability_' + c] = predictions[:,n]

    if settings.ExportParticles.export_images:
        stats['export name'] = filenames


    return stats
