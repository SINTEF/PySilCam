# -*- coding: utf-8 -*-

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

'''
module for processing SilCam data

TODO: add tests for this module
'''

#Get module-level logger
logger = logging.getLogger(__name__)


def im2bw_fancy(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    returns:
      imbw (binary image)
    '''
    img = np.copy(imc) # create a copy of the input image (not sure why)

    # obtain a semi-autimated treshold which can handle
    # some flicker in the illumination by tracking the 50th percentile of the
    # image histogram
    thresh = np.uint8(greythresh * np.percentile(img, 50))
    
    # create a segmented image using the crude threshold
    imbw1 = np.invert(img > thresh)

    # perform an adaptive historgram equalization to handle some
    # less-than-ideal lighting situations
    img_adapteq = skimage.exposure.equalize_adapthist(img,
            clip_limit=(1-greythresh),
            nbins=256)
    
    # use the equalised image to estimate a second semi-automated threshold
    newthresh = np.percentile(img_adapteq, 0.75) * greythresh

    # create a second segmented image using newthresh
    imbw2 = img_adapteq < newthresh

    # merge both segmentation methods by selecting regions where both identify
    # something as a particle (everything else is water)
    imbw = imbw1 & imbw2

    return imbw


def im2bw(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (fixed scaling of greythresh is done inside)

    returns:
      imbw (binary image)
    '''

    thresh = np.uint8(greythresh * 230)  # or use a faster less-good version
    imbw = imc < thresh  # segment the image

    return imbw


def clean_bw(imbw, min_area):
    '''cleans up particles which are too small and particles touching the
    border
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


def filter_bad_stats(stats,settings):
    ''' remove unacceptable particles from the stats
    '''
    # calculate minor-major axis ratio
    mmr = stats['minor_axis_length'] / stats['major_axis_length']   
    # remove stats where particles are too deformed
    stats = stats[mmr > settings.Process.min_deformation]

    # remove particles that exceed the maximum dimention
    stats = stats[(stats['major_axis_length'] * settings.PostProcess.pix_size) <
            settings.Process.max_length]

    # calculate the area of the bounding boxes
    bbox_area = (stats['maxr']-stats['minr']) * (stats['maxc']-stats['minc'])

    #Discard overlapping particles (approximate by solidity requirement)
    stats = stats[(stats['equivalent_diameter'] ** 2 / bbox_area) >
            (settings.Process.min_solidity)]
    # this is a crude, but fast approximate of solidity

    return stats


def fancy_props(iml, imc, settings):
    '''Calculates fancy particle properties

    return pandas.DataFrame

    partstats = fancy_props(iml, imc, settings)
    '''

    # define the geometrical properties to be calculated from regionprops
    propnames = ['major_axis_length', 'minor_axis_length',
                 'equivalent_diameter']

    # measure geometrical properties of particles
    region_properties = measure.regionprops(iml, cache=False)
    # cache=False is faster

    # pre-allocate some things
    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)
    bboxes = np.zeros((len(region_properties), 4), dtype=np.float64)

    # loop through each particle and extract the desired statistics
    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]
        bboxes[i, :] = el.bbox

    # build the column names for the outputed DataFrame
    column_names = np.hstack(([propnames, 'minr', 'minc', 'maxr', 'maxc']))

    # merge regionprops statistics with a seperate bounding box columns
    cat_data = np.hstack((data, bboxes))

    # put particle statistics into a DataFrame
    partstats = pd.DataFrame(columns=column_names, data=cat_data)

    return partstats


def concentration_check(imbw, settings):
    ''' Check saturation level of the sample volume by comparing area of
    particles with settings.Process.max_coverage

    sat_check, saturation = concentration_check(imbw, settings)

    set_check is a flag, which is True if the image is acceptable
    saturation is the percentaage saturated
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

    returns:
      spine_length
    '''
    skel = morphology.skeletonize(imbw)
    for i in range(2):
        skel = morphology.binary_dilation(skel)
    skel = morphology.skeletonize(skel)

    spine_length = np.sum(skel)
    return spine_length


def extract_roi(im, bbox):
    ''' given an image (im) and bounding box (bbox), this will return the roi

    returns:
      roi
    '''
    roi = im[bbox[0]:bbox[2], bbox[1]:bbox[3]] # yep, that't it.
    return roi


def measure_particles(imbw, imc, settings):
    '''Measures properties of particles

    Parameters:
      imbw (full-frame binary image)
      imc (full-frame corrected raw image)
      image_index (some sort of tag for location matching)

    Returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)

    '''

    # check the converage of the image of particles is acceptable
    sat_check, saturation = concentration_check(imbw, settings)
    if sat_check == False:
        logger.warn('....breached concentration limit! Skipping image.')
        imbw *= 0 # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found

    # label the segmented image
    iml = morphology.label(imbw > 0)
    logger.info('  {0} particles found'.format(iml.max()))

    # if there are too many particles then do no proceed with analysis
    if (iml.max() > settings.Process.max_particles):
        logger.warn('....that''s way too many particles! Skipping image.')
        imbw *= 0 # this is not a good way to handle this condition
        # @todo handle situation when too many particles are found


    # calculate particle statistics
    stats = fancy_props(iml, imc, settings)

    return stats, saturation


def statextract(imc, settings, fancy=False):
    '''extracts statistics of particles in imc (raw corrected image)

    returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)
    '''
    logger.debug('segment')

    if fancy:  # check is fancy (slow) processing is enabled
        # segment the image
        imbw = im2bw_fancy(imc, settings.Process.threshold)
    else:
        # segment the image
        imbw = im2bw(imc, settings.Process.threshold)

    logger.debug('clean')

    # clean segmented image (small particles and border particles)
    imbw = clean_bw(imbw, settings.Process.minimum_area)

    # fill holes in particles
    imbw = ndi.binary_fill_holes(imbw)

    logger.debug('measure')
    # calculate particle statistics
    stats, saturation = measure_particles(imbw, imc, settings)

    # remove bad particles from data
    stats = filter_bad_stats(stats, settings)

    return stats, imbw, saturation
