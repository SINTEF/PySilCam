# -*- coding: utf-8 -*-

import time
import numpy as np
from skimage import morphology
from skimage import segmentation
from skimage import measure
import pandas as pd
import logging
from scipy import ndimage as ndi
import skimage.exposure
import h5py
import os
import scipy.misc

'''
module for processing SilCam data

TODO: add tests for this module
'''

#Get module-level logger
logger = logging.getLogger(__name__)


def image2binary_accurate(imc, greythresh):
    """ converts corrected image (imc) to a binary image
   using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

   Args:
       imc                         : background-corrected image
       greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image histogram)

   Returns:
       imbw                        : segmented image (binary image)

   """

    img = np.copy(imc)

    # obtain a semi-automated threshold which can handle some flicker in the illumination by tracking the
    # 50th percentile of the image histogram
    thresh = np.uint8(greythresh * np.percentile(img, 50))

    # create a segmented image using the crude threshold
    im_binary1 = img < thresh

    # perform an adaptive historgram equalization to handle some
    # less-than-ideal lighting situations
    img_adapteq = skimage.exposure.equalize_adapthist(img, clip_limit=(1-greythresh), nbins=256)

    # use the equalised image to estimate a second semi-automated threshold
    new_thresh = np.percentile(img_adapteq, 0.75) * greythresh

    # create a second segmented image using newthresh
    im_binary2 = img_adapteq < new_thresh

    # merge both segmentation methods by selecting regions where both identify
    # something as a particle (everything else is water)
    im_binary = im_binary1 & im_binary2

    return im_binary


def image2binary_fast(imc, greythresh):
    """ converts corrected image (imc) to a binary image
   using greythresh as the threshold value (fixed scaling of greythresh is done inside)

   Args:
       imc                         : background-corrected image
       greythresh                  : threshold multiplier (greythresh is multiplied by 50th percentile of the image histogram)

   Returns:
       imbw                        : segmented image (binary image)
   """

    # obtain a semi-automated threshold which can handle some flicker in the illumination by tracking the
    # 50th percentile of the image histogram
    thresh = np.uint8(greythresh * np.percentile(imc, 50))
    imbw = imc < thresh  # segment the image

    return imbw


def clean_bw(im_binary, min_area):
    ''' cleans up particles which are too small and particles touching the
    border

    Args:
        im_binary                   : segmented image
        min_area                    : minimum number of accepted pixels for a particle

    Returns:
        im_binary (DataFrame)       : cleaned up segmented image

    '''

    # remove objects that are below the detection limit defined in the config file.
    # this min_area is usually 12 pixels
    im_binary = morphology.remove_small_objects(im_binary > 0, min_size=min_area)

    # remove particles touching the border of the image because there might be part of a particle not recorded,
    # and therefore border particles will be incorrectly sized
    im_binary = segmentation.clear_border(im_binary, buffer_size=2)

    # remove objects smaller the min_area
    return im_binary


def filter_bad_stats(stats, settings):
    """ remove unacceptable particles from the stats

   Note that for oil and gas analysis, this filtering is handled by the functions in the pysilcam.oilgas module.

   Args:
       stats (DataFrame)           : particle statistics from silcam process
       settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings

   Returns:
       stats (DataFrame)           : particle statistics from silcam process
   """

    # calculate minor-major axis ratio
    mmr = stats['minor_axis_length'] / stats['major_axis_length']
    # remove stats where particles are too deformed
    stats = stats[mmr > settings.Process.min_deformation]

    # remove particles that exceed the maximum dimension
    stats = stats[(stats['major_axis_length'] * settings.PostProcess.pix_size) < settings.Process.max_length]

    return stats


def concentration_check(im_binary, settings):
    """ Check saturation level of the sample volume by comparing area of
    particles with settings.Process.max_coverage

    Args:
        im_binary                   : segmented image
        settings                    : PySilCam settings

    Returns:
        sat_check                   : boolean on if the saturation is acceptable. True if the image is acceptable
        saturation                  : percentage of maximum acceptable saturation defined in settings.Process.max_coverage
    """

    # calculate the area covered by particles in the binary image
    covered_area = float(im_binary.sum())

    # measure the image size for correct area calculation
    r, c = np.shape(im_binary)

    # calculate the percentage of the image covered by particles
    covered_pcent = covered_area / (r * c) * 100

    # convert the percentage covered to a saturation based on the maximum acceptable coverage defined in the config
    saturation = covered_pcent / settings.Process.max_coverage * 100

    logger.info('{0:.1f}% saturation'.format(saturation))

    # check if the saturation is acceptable
    sat_check = saturation < 100

    return sat_check, saturation


def measure_particles(im_binary, imc, settings, timestamp, nnmodel, class_labels):
    '''Measures properties of particles

    Args:
      im_binary (full-frame binary image)
      imc (full-frame corrected raw image)
      image_index (some sort of tag for location matching)

    Returns:
      stats (list of particle statistics for every particle, according to Partstats class)

    '''
    # check the converage of the image of particles is acceptable
    sat_check, saturation = concentration_check(im_binary, settings)
    if sat_check == False:
        logger.warning('....breached concentration limit! Skipping image.')
        im_binary *= 0 # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    # label the segmented image
    iml = morphology.label(im_binary > 0)
    logger.info('  {0} particles found'.format(iml.max()))

    # if there are too many particles then do no proceed with analysis
    if iml.max() > settings.Process.max_particles:
        logger.warning('....that''s way too many particles! Skipping image.')
        im_binary *= 0  # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    # calculate particle statistics
    region_properties = measure.regionprops(iml, cache=False, coordinates='xy')
    stats = extract_particles(imc, timestamp, settings, nnmodel, class_labels, region_properties)

    return stats, saturation


def threshold_im(im_rgb, settings):
    '''im_rgb (raw corrected image) to binary image

    Args:
        im_rgb                      : background-corrected image
        settings                    : PySilCam settings

    Returns:
        im_binary                   : segmented image
    '''

    logger.debug('segment')

    # simplify processing by squeezing the image dimentions into a 2D array
    # min is used for squeezing to represent the highest attenuation of all wavelengths
    img = np.uint8(np.min(im_rgb, axis=2))

    if settings.Process.real_time_stats:
        im_binary = image2binary_fast(img, settings.Process.threshold) # image2binary_fast is less fancy but
    else:
        im_binary = image2binary_accurate(img, settings.Process.threshold) # image2binary_fast is less fancy but
    # image2binary_fast is faster than image2binary_accurate but might cause problems when trying to
    # process images with bad lighting

    logger.debug('clean')

    # clean segmented image (small particles and border particles)
    im_binary = clean_bw(im_binary, settings.Process.minimum_area)

    # fill holes in particles
    im_binary = ndi.binary_fill_holes(im_binary)

    logger.debug('measure')

    return im_binary


def extract_particles(imc, timestamp, settings, nnmodel, class_labels, region_properties):
    """extracts the particles to build stats and export particle rois to HDF5 files writted to disc in the location of settings.ExportParticles.outputpath

    Args:
        imc                         : background-corrected image
        timestamp                   : timestamp of image collection
        settings                    : PySilCam settings
        nnmodel                     : loaded tensorflow model from silcam_classify
        class_labels                : lables of particle classes in tensorflow model
        region_properties           : region properties object returned from regionprops (measure.regionprops(iml, cache=False))

    Returns:
        stats                       : (list of particle statistics for every particle, according to Partstats class)

    @todo clean up all the unnesessary conditional statements in this
    """

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
        # @todo include more useful information in this meta data, e.g. possibly raw image location and background stack file list.

    # define the geometrical properties to be calculated from regionprops
    propnames = ['major_axis_length', 'minor_axis_length', 'equivalent_diameter', 'solidity']

    # pre-allocate some things
    particle_data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)
    bboxes = np.zeros((len(region_properties), 4), dtype=np.uint16)
    filenames = ['not_exported'] * len(region_properties)
    rois = np.zeros((0, 32, 32, 3), dtype=np.float32)
    predictions = np.zeros((len(region_properties), len(class_labels)))
    particle_ids_export = []

    # Filter region properties for particles that are being exported and classified
    for i, el in enumerate(region_properties):
        data = [getattr(el, p) for p in propnames]
        particle_data[i, :] = data
        bboxes[i, :] = el.bbox

        # if operating in realtime mode, assume we only care about oil and gas and skip export of overly-derformed particles
        if settings.Process.real_time_stats and (data[1]/data[0] < 0.3 or data[3] < 0.95):
            continue

        # Find particles that match export criteria
        if data[0] < settings.ExportParticles.min_length or data[1] < 2:
            continue

        particle_ids_export.append(i)
        roi = imc[el.bbox[0]:el.bbox[2], el.bbox[1]:el.bbox[3]]

        # Rescale to neural net input size (32 x 32)
        roi_rz = scipy.misc.imresize(roi, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
        roi_rz = np.expand_dims(roi_rz, axis=0)
        rois = np.vstack((rois, roi_rz))

        filenames[i] = filename + '-PN' + str(i)

        # add the roi to the HDF5 file
        if settings.ExportParticles.export_images:
            HDF5File.create_dataset('PN' + str(i), data=roi)
            # @todo also include particle stats here too.

    if settings.ExportParticles.export_images:
        HDF5File.close()

    predictions[particle_ids_export, :] = nnmodel.predict(rois)

    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((particle_data, bboxes))

    # put particle statistics into a DataFrame
    stats = pd.DataFrame(columns=column_names, data=cat_data)

    logger.info('Extracting {0} rois from {1} possible'.format(len(particle_ids_export), len(stats['major_axis_length'])))

    # add classification predictions to the particle statistics data
    for n,c in enumerate(class_labels):
       stats['probability_' + c] = predictions[:, n]

    # add the filenames of the HDF5 file and particle number tag to the
    # particle statistics data
    stats['export name'] = filenames

    if settings.ExportParticles.export_images:
        # put a copy of the stats data for this image into the hdf5 file
        hdf_stats = pd.HDFStore(hdf_filename, 'r+')
        hdf_stats.put('Proc/STATS', stats)
        hdf_stats.close()

        HDF5File = h5py.File(hdf_filename, "r+")
        HDF5File['Proc'].attrs['Modified'] = str(pd.datetime.now())
        HDF5File['Proc'].attrs['Descriton'] = 'Pandas dataframe of particle stats data. Load it like this: stats = pd.read_hdf(h5file, "Proc/STATS")'
        HDF5File.close()

    return stats


def processImage(nnmodel, class_labels, image_data, settings):
    '''
    Proceses an image

    Args:
        nnmodel (tensorflow model object)   :  loaded using sccl.load_model()
        class_labels (str)                  :  loaded using sccl.load_model()
        image_data  (tuple)                 :  tuple contianing (i, timestamp, imc)
                                               where i is an int referring to the image number
                                               timestamp is the image timestamp obtained from passing the filename
                                               imc is the background-corrected image obtained using the backgrounder generator
        settings (PySilcamSettings)         :  Settings read from a .ini file

    Returns:
        stats_all (DataFrame)               :  stats dataframe containing particle statistics
    '''

    im_idx = image_data[0]
    im_timestamp = image_data[1]
    im_rgb = image_data[2]

    try:
        # time the full acquisition and processing loop
        start_time = time.clock()
        logger.info('Processing time stamp {0}'.format(im_timestamp))

        im_binary = threshold_im(im_rgb, settings)

        sat_check, saturation = concentration_check(im_binary, settings)
        if not sat_check:
            logger.warning('....breached concentration limit! Skipping image.')
            return None  # TODO return nan array here

        # label the segmented image
        im_labels = morphology.label(im_binary > 0)
        logger.info('  {0} particles found'.format(im_labels.max()))

        # if there are too many particles then do no proceed with analysis
        if im_labels.max() > settings.Process.max_particles:
            logger.warning('....that''s way too many particles! Skipping image.')
            return None  # TODO return nan array here

        # calculate particle statistics
        region_properties = measure.regionprops(im_labels, cache=False, coordinates='xy')
        stats_all = extract_particles(im_rgb, im_timestamp, settings, nnmodel, class_labels, region_properties)

        # if there are not particles identified, assume zero concentration.
        # This means that the data should indicate that a 'good' image was
        # obtained, without any particles. Therefore fill all values with nans
        # and add the image timestamp
        if len(stats_all) == 0:
            print('ZERO particles identified')
            stats_all.loc[0] = np.zeros(len(stats_all.columns)) * np.nan
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            if settings.ExportParticles.export_images:
                stats_all['export name'] = 'not_exported'

        stats_all['timestamp'] = im_timestamp  # add timestamp to each row of particle statistics
        stats_all['saturation'] = saturation  # add saturation to each row of particle statistics

        # Print processing time
        proc_time = time.clock() - start_time
        print('  Image {0} processed in {1:.2f} sec ({2:.1f} Hz).'.format(im_idx, proc_time, 1.0 / proc_time))

    except:
        infostr = 'Failed to process frame {0}, skipping.'.format(im_idx)
        logger.warning(infostr, exc_info=True)
        print(infostr)
        return None

    return stats_all


