# -*- coding: utf-8 -*-

import time
import numpy as np
from skimage import morphology
from skimage import segmentation
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import measure
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy import ndimage as ndi
from scipy import signal
from scipy import interpolate
import skimage.exposure
import h5py
import os
import pysilcam.silcam_classify as sccl
from skimage.io import imsave
import traceback

'''
Module for processing SilCam data

TODO: add tests for this module
'''

# Get module-level logger
logger = logging.getLogger(__name__)


def image2blackwhite_accurate(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    Args:
        imc                         : background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)

    '''
    img = np.copy(imc)  # create a copy of the input image (not sure why)

    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = np.uint8(greythresh * np.percentile(img, 50))

    # create a segmented image using the crude threshold
    imbw1 = img < thresh

    # perform an adaptive historgram equalization to handle some
    # less-than-ideal lighting situations
    img_adapteq = skimage.exposure.equalize_adapthist(img,
                                                      clip_limit=(1 - greythresh),
                                                      nbins=256)

    # use the equalised image to estimate a second semi-automated threshold
    newthresh = np.percentile(img_adapteq, 0.75) * greythresh

    # create a second segmented image using newthresh
    imbw2 = img_adapteq < newthresh

    # merge both segmentation methods by selecting regions where both identify
    # something as a particle (everything else is water)
    imbw = imbw1 & imbw2

    return imbw


def image2blackwhite_fast(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (fixed scaling of greythresh is done inside)

    Args:
        imc                         : background-corrected image
        greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image
                                      histogram)

    Returns:
        imbw                        : segmented image (binary image)
    '''
    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = np.uint8(greythresh * np.percentile(imc, 50))
    imbw = imc < thresh  # segment the image

    return imbw


def clean_bw(imbw, min_area):
    ''' cleans up particles which are too small and particles touching the
    border

    Args:
        imbw                        : segmented image
        min_area                    : minimum number of accepted pixels for a particle

    Returns:
        imbw (DataFrame)           : cleaned up segmented image

    '''

    # remove objects that are below the detection limit defined in the config
    # file.
    # this min_area is usually 12 pixels
    imbw = morphology.remove_small_objects(imbw > 0, min_size=min_area)

    # remove particles touching the border of the image
    # because there might be part of a particle not recorded, and therefore
    # border particles will be incorrectly sized
    imbw = segmentation.clear_border(imbw, buffer_size=2)

    # remove objects smaller the min_area
    return imbw


def filter_bad_stats(stats, settings):
    ''' remove unacceptable particles from the stats

    Note that for oil and gas analysis, this filtering is handled by the functions in the pysilcam.oilgas module.

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings

    Returns:
        stats (DataFrame)           : particle statistics from silcam process
    '''

    # calculate minor-major axis ratio
    mmr = stats['minor_axis_length'] / stats['major_axis_length']
    # remove stats where particles are too deformed
    stats = stats[mmr > settings.Process.min_deformation]

    # remove particles that exceed the maximum dimention
    stats = stats[(stats['major_axis_length'] * settings.PostProcess.pix_size) <
                  settings.Process.max_length]

    return stats


def fancy_props(iml, imc, timestamp, settings, nnmodel, class_labels):
    '''Calculates fancy particle properties

    Args:
        iml                         : labelled segmented image
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        settings                    : PySilCam settings
        nnmodel                     : loaded tensorflow model from silcam_classify
        class_labels                : lables of particle classes in tensorflow model

    Return:
        stats                       : particle statistics

    '''

    region_properties = measure.regionprops(iml, cache=False)
    # build the stats and export to HDF5
    stats = extract_particles(imc, timestamp, settings, nnmodel, class_labels, region_properties)

    return stats


def concentration_check(imbw, settings):
    ''' Check saturation level of the sample volume by comparing area of
    particles with settings.Process.max_coverage

    Args:
        imbw                        : segmented image
        settings                    : PySilCam settings

    Returns:
        sat_check                   : boolean on if the saturation is acceptable. True if the image is acceptable
        saturation                  : percentage of maximum acceptable saturation defined in
                                      settings.Process.max_coverage
    '''

    # calcualte the area covered by particles in the binary image
    covered_area = float(imbw.sum())

    # measure the image size for correct area calcaultion
    r, c = np.shape(imbw)

    # calculate the percentage of the image covered by particles
    covered_pcent = covered_area / (r * c) * 100

    # convert the percentage covered to a saturation based on the maximum
    # acceptable coverage defined in the config
    saturation = covered_pcent / settings.Process.max_coverage * 100

    logger.info('{0:.1f}% saturation'.format(saturation))

    # check if the saturation is acceptable
    sat_check = saturation < 100

    return sat_check, saturation


def get_spine_length(imbw):
    ''' extracts the spine length of particles from a binary particle image
    (imbw is a binary roi)

    Args:
        imbw                : segmented particle ROI (assumes only one particle)

    Returns:
        spine_length        : spine length of particle (in pixels)
    '''
    skel = morphology.skeletonize(imbw)
    for i in range(2):
        skel = morphology.binary_dilation(skel)
    skel = morphology.skeletonize(skel)

    spine_length = np.sum(skel)
    return spine_length


def extract_roi(im, bbox):
    ''' given an image (im) and bounding box (bbox), this will return the roi

    Args:
        im                  : any image, such as background-corrected image (imc)
        bbox                : bounding box from regionprops [r1, c1, r2, c2]

    Returns:
        roi                 : image cropped to region of interest
    '''
    # refer to skimage regionprops documentation on how bbox is structured
    roi = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    return roi


def measure_particles(imbw, imc, settings, timestamp, nnmodel, class_labels):
    '''Measures properties of particles

    Args:
      imbw (full-frame binary image)
      imc (full-frame corrected raw image)
      image_index (some sort of tag for location matching)

    Returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)

    '''
    # check the converage of the image of particles is acceptable
    sat_check, saturation = concentration_check(imbw, settings)
    if (sat_check is False):
        logger.warning('....breached concentration limit! Skipping image.')
        imbw *= 0  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    # label the segmented image
    iml = morphology.label(imbw > 0)
    logger.info('  {0} particles found'.format(iml.max()))

    # if there are too many particles then do no proceed with analysis
    if (iml.max() > settings.Process.max_particles):
        logger.warning('....that''s way too many particles! Skipping image.')
        imbw *= 0  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    # calculate particle statistics
    stats = fancy_props(iml, imc, timestamp, settings, nnmodel, class_labels)

    return stats, saturation


def statextract(imc, settings, timestamp, nnmodel, class_labels):
    '''extracts statistics of particles in imc (raw corrected image)

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        settings                    : PySilCam settings
        nnmodel                     : loaded tensorflow model from silcam_classify
        class_labels                : lables of particle classes in tensorflow model

    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)
        imbw                        : segmented image
        saturation                  : percentage saturation of image
    '''
    logger.debug('segment')

    # simplyfy processing by squeezing the image dimensions into a 2D array
    # min is used for squeezing to represent the highest attenuation of all wavelengths
    img = np.uint8(np.min(imc, axis=2))

    if settings.Process.real_time_stats:
        imbw = image2blackwhite_fast(img, settings.Process.threshold)  # image2blackwhite_fast is less fancy but
    else:
        imbw = image2blackwhite_accurate(img, settings.Process.threshold)  # image2blackwhite_fast is less fancy but
    # image2blackwhite_fast is faster than image2blackwhite_accurate but might cause problems when trying to
    # process images with bad lighting

    logger.debug('clean')

    # clean segmented image (small particles and border particles)
    imbw = clean_bw(imbw, settings.Process.minimum_area)

    # fill holes in particles
    imbw = ndi.binary_fill_holes(imbw)

    write_segmented_images(imbw, imc, settings, timestamp)

    logger.debug('measure')
    # calculate particle statistics
    stats, saturation = measure_particles(imbw, imc, settings, timestamp, nnmodel, class_labels)

    return stats, imbw, saturation


def write_segmented_images(imbw, imc, settings, timestamp):
    '''writes binary images as bmp files to the same place as hdf5 files if loglevel is in DEBUG mode
    Useful for checking threshold and segmentation

    Args:
        imbw                        : segmented image
        settings                    : PySilCam settings
        timestamp                   : timestamp of image collection
    '''
    if (settings.General.loglevel == 'DEBUG') and settings.ExportParticles.export_images:
        fname = os.path.join(settings.ExportParticles.outputpath, timestamp.strftime('D%Y%m%dT%H%M%S.%f-SEG.bmp'))
        imbw_ = np.uint8(255 * imbw)
        imsave(fname, imbw_)
        fname = os.path.join(settings.ExportParticles.outputpath, timestamp.strftime('D%Y%m%dT%H%M%S.%f-IMC.bmp'))
        imsave(fname, imc)


def extract_particles(imc, timestamp, settings, nnmodel, class_labels, region_properties):
    '''extracts the particles to build stats and export particle rois to HDF5 files writted to disc in the location of
       settings.ExportParticles.outputpath

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        settings                    : PySilCam settings
        nnmodel                     : loaded tensorflow model from silcam_classify
        class_labels                : lables of particle classes in tensorflow model
        region_properties           : region properties object returned from regionprops (measure.regionprops(iml,
                                                                                                           cache=False))

    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)

    @todo clean up all the unnesessary conditional statements in this
    '''
    filenames = ['not_exported'] * len(region_properties)

    # pre-allocation
    predictions = np.zeros((len(region_properties),
                            len(class_labels)),
                           dtype='float64')
    predictions *= np.nan

    # obtain the original image filename from the timestamp
    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f')

    if settings.ExportParticles.export_images:
        # Make the HDF5 file
        hdf_filename = os.path.join(settings.ExportParticles.outputpath, filename + ".h5")
        HDF5File = h5py.File(hdf_filename, "w")
        # metadata
        meta = HDF5File.create_group('Meta')
        meta.attrs['Modified'] = str(pd.datetime.now())
        settings_dict = {s: dict(settings.config.items(s)) for s in settings.config.sections()}
        meta.attrs['Settings'] = str(settings_dict)
        meta.attrs['Timestamp'] = str(timestamp)
        meta.attrs['Raw image name'] = filename
        # @todo include more useful information in this meta data, e.g. possibly raw image location and background
        #  stack file list.

    # define the geometrical properties to be calculated from regionprops
    propnames = ['major_axis_length', 'minor_axis_length',
                 'equivalent_diameter', 'solidity']

    # pre-allocate some things
    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)
    bboxes = np.zeros((len(region_properties), 4), dtype=np.float64)
    nb_extractable_part = 0

    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]
        bboxes[i, :] = el.bbox

        # if operating in realtime mode, assume we only care about oil and gas and skip export of overly-derformed
        # particles
        if settings.Process.real_time_stats & (((data[i, 1] / data[i, 0]) < 0.3) | (data[i, 3] < 0.95)):
            continue
        # Find particles that match export criteria
        if ((data[i, 0] > settings.ExportParticles.min_length) &  # major_axis_length in pixels
                (data[i, 1] > 2)):  # minor length in pixels

            nb_extractable_part += 1
            # extract the region of interest from the corrected colour image
            roi = extract_roi(imc, bboxes[i, :].astype(int))

            # add the roi to the HDF5 file
            filenames[int(i)] = filename + '-PN' + str(i)
            if settings.ExportParticles.export_images:
                HDF5File.create_dataset('PN' + str(i), data=roi)
                # @todo also include particle stats here too.

            # run a prediction on what type of particle this might be
            prediction = sccl.predict(roi, nnmodel)
            predictions[int(i), :] = prediction[0]

    if settings.ExportParticles.export_images:
        # close the HDF5 file
        HDF5File.close()

    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((data, bboxes))

    # put particle statistics into a DataFrame
    stats = pd.DataFrame(columns=column_names, data=cat_data)

    logger.info('EXTRACTING {0} IMAGES from {1}'.format(nb_extractable_part, len(stats['major_axis_length'])))

    # add classification predictions to the particle statistics data
    for n, c in enumerate(class_labels):
        stats['probability_' + c] = predictions[:, n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    stats['export name'] = filenames

    return stats


def processImage(nnmodel, class_labels, image, settings, logger, gui):
    '''
    Proceses an image

    Args:
        nnmodel (tensorflow model object)   :  loaded using sccl.load_model()
        class_labels (str)                  :  loaded using sccl.load_model()
        image  (tuple)                      :  tuple contianing (i, timestamp, imc)
                                               where i is an int referring to the image number
                                               timestamp is the image timestamp obtained from passing the filename
                                               imc is the background-corrected image obtained using the backgrounder
                                               generator
        settings (PySilcamSettings)         :  Settings read from a .ini file
        logger (logger object)              :  logger object created using
                                               configure_logger()
        gui=None (Class object)             :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py

    Returns:
        stats_all (DataFrame)               :  stats dataframe containing particle statistics
    '''
    try:
        i = image[0]
        timestamp = image[1]
        imc = image[2]

        # time the full acquisition and processing loop
        start_time = time.time()

        logger.info('Processing time stamp {0}'.format(timestamp))

        # Calculate particle statistics
        stats_all, imbw, saturation = statextract(imc, settings, timestamp,
                                                  nnmodel, class_labels)

        # if there are not particles identified, assume zero concentration.
        # This means that the data should indicate that a 'good' image was
        # obtained, without any particles. Therefore fill all values with nans
        # and add the image timestamp
        if len(stats_all) == 0:
            print('ZERO particles identified')
            z = np.zeros(len(stats_all.columns)) * np.nan
            stats_all.loc[0] = z
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            # padding end of string required for HDF5 writing
            stats_all['export name'] = 'not_exported'

        # add timestamp to each row of particle statistics
        stats_all['timestamp'] = timestamp

        # add saturation to each row of particle statistics
        stats_all['saturation'] = saturation

        # Time the particle statistics processing step
        proc_time = time.time() - start_time

        # Print timing information for this iteration
        infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
        infostr = infostr.format(i, proc_time, 1.0 / proc_time)
        print(infostr)

        # ---- END MAIN PROCESSING LOOP ----
        # ---- DO SOME ADMIN ----

    except Exception as e:
        print(e)
        traceback.print_exc()
        infostr = 'Failed to process frame {0}, skipping.'.format(i)
        logger.warning(infostr, exc_info=True)
        return None

    return stats_all


def write_stats(
        datafilename,
        stats_all,
        append=True,
        export_name_len=40):
    '''
    Writes particle stats into the ouput file

    Args:
        datafilename (str):     filame prefix for -STATS.h5 file that may or may not include a path
        stats_all (DataFrame):  stats dataframe returned from processImage()
        append (bool):          if to allow append
        export_name_len (int):  max number of chars allowed for col 'export name'
    '''

    # create or append particle statistics to output file
    # if the output file does not already exist, create it
    # otherwise data will be appended
    # @todo accidentally appending to an existing file could be dangerous
    # because data will be duplicated (and concentrations would therefore
    # double) GUI promts user regarding this - directly-run functions are more dangerous.
    if 'export name' in stats_all.columns:
        min_itemsize = {'export name': export_name_len}
    else:
        min_itemsize = None

    with pd.HDFStore(datafilename + '-STATS.h5', 'a') as fh:
        stats_all.to_hdf(
            fh, 'ParticleStats/stats', append=append, format='t',
            data_columns=True, min_itemsize=min_itemsize)
