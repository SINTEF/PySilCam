# -*- coding: utf-8 -*-
'''
Module for exporting particle images of specified characteristics
'''

from pysilcam.process import extract_roi
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os
import pysilcam.silcam_classify as sccl
import h5py

def export_particles(imc,timestamp,stats,settings,nnmodel,class_labels):
    '''
    function to export particle rois to HDF5

    @todo clean up all the unnesessary conditional statements in this
    '''

    # Find particles that match export criteria
    inds = np.argwhere(((settings.PostProcess.pix_size *
            stats['major_axis_length'])>settings.ExportParticles.min_length) &
            ((settings.PostProcess.pix_size * stats['minor_axis_length'])>2))

    extractable_particles = len(inds)
    print('EXTRACTING {0} IMAGES from {1}'.format(extractable_particles,len(stats['major_axis_length'])))

    # pre-allocation
    bbox = np.zeros((4,1),dtype=int)

    # pre-allocation
    if settings.ExportParticles.export_images:
        filenames = ['not_exported'] * len(stats['major_axis_length'])

    # pre-allocation
    if settings.NNClassify.enable:
        predictions = np.zeros((len(stats['major_axis_length']),
            len(class_labels)),
            dtype='float64')
        predictions *= np.nan

    # obtain the original image filename from the timestamp
    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

    # Make the HDF5 file
    HDF5File = h5py.File(os.path.join(settings.ExportParticles.ouputpath, filename + ".h5"), "w")

    # loop through all extractable particles and find their bounding boxes
    for i in inds:
        bbox[0] = stats.iloc[i]['minr']
        bbox[1] = stats.iloc[i]['minc']
        bbox[2] = stats.iloc[i]['maxr']
        bbox[3] = stats.iloc[i]['maxc']

        # extract the region of interest from the corrected colour image
        roi = extract_roi(imc, bbox[:,0])


        # add the roi to the HDF5 file
        if settings.ExportParticles.export_images:
            filenames[int(i)] = filename + '-PN' + str(i[0])
            dset = HDF5File.create_dataset('PN' + str(i[0]), data = roi)

        # run a prediction on what type of particle this might be
        if settings.NNClassify.enable:
            prediction = sccl.predict(roi, nnmodel)
            predictions[int(i),:] = prediction[0]

    # close the HDF5 file
    HDF5File.close()
    
    # add classification predictions to the particle statistics data
    if settings.NNClassify.enable:
        for n,c in enumerate(class_labels):
            stats['probability_' + c] = predictions[:,n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    if settings.ExportParticles.export_images:
        stats['export name'] = filenames


    return stats
