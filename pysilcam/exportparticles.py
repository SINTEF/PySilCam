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
import pandas as pd


def export_particles(imc,timestamp,settings,nnmodel,class_labels,region_properties):
    '''
    function to export particle rois to HDF5

    @todo clean up all the unnesessary conditional statements in this
    '''

    # pre-allocation
    if settings.ExportParticles.export_images:
        filenames = ['not_exported'] * len(region_properties)

    # pre-allocation
    if settings.NNClassify.enable:
        predictions = np.zeros((len(region_properties),
            len(class_labels)),
            dtype='float64')
        predictions *= np.nan

    # obtain the original image filename from the timestamp
    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

    # Make the HDF5 file
    HDF5File = h5py.File(os.path.join(settings.ExportParticles.ouputpath, filename + ".h5"), "w")

    # define the geometrical properties to be calculated from regionprops
    propnames = ['major_axis_length', 'minor_axis_length',
                 'equivalent_diameter']

    # pre-allocate some things
    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)
    bboxes = np.zeros((len(region_properties), 4), dtype=np.float64)
    nb_extractable_part = 0

    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]
        bboxes[i, :] = el.bbox

        # Find particles that match export criteria 
        if (((settings.PostProcess.pix_size *
            data[i, 0]) > settings.ExportParticles.min_length) &  #major_axis_length
            ((settings.PostProcess.pix_size * data[i, 1]) > 2)):  #minor_axis_length
            
            nb_extractable_part += 1
            # extract the region of interest from the corrected colour image
            roi = extract_roi(imc,bboxes[i, :].astype(int))
            
            # add the roi to the HDF5 file
            if settings.ExportParticles.export_images:
                filenames[int(i)] = filename + '-PN' + str(i)
                dset = HDF5File.create_dataset('PN' + str(i), data = roi)

            # run a prediction on what type of particle this might be
            if settings.NNClassify.enable:
                prediction = sccl.predict(roi, nnmodel)
                predictions[int(i),:] = prediction[0]

    # close the HDF5 file
    HDF5File.close()
 
 # @TODO  : create function that builds stats
#---------------------------------------------------------------------------
    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((data, bboxes))

    # put particle statistics into a DataFrame
    stats = pd.DataFrame(columns=column_names, data=cat_data)
#---------------------------------------------------------------------------
    print('EXTRACTING {0} IMAGES from {1}'.format(nb_extractable_part, len(stats['major_axis_length']))) 
    
    # add classification predictions to the particle statistics data
    if settings.NNClassify.enable:
        for n,c in enumerate(class_labels):
            stats['probability_' + c] = predictions[:,n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    if settings.ExportParticles.export_images:
        stats['export name'] = filenames

    return stats
