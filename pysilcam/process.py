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
from scipy import  ndimage as ndi

'''
module for processing SilCam data

TODO: add tests for this module
'''

#Get module-level logger
logger = logging.getLogger(__name__)


def im2bw(imc, greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    returns:
      imbw (binary image)
    '''

    #thresh = np.uint8(greythresh * np.median(img))  # determine auto-amazing theshold estimate
    thresh = np.uint8(greythresh * 230)  # or use a faster less-good version

    imbw = imc < thresh  # segment the image

    return imbw


def clean_bw(imbw, min_area):
    '''cleans up particles which are too small and particles touching the
    border
    '''
    imbw = morphology.remove_small_objects(imbw > 0, min_size=min_area)
    imbw = segmentation.clear_border(imbw, in_place=True)  # remove particles touching the border of the image

    # remove objects smaller the min_area
    return imbw


def fast_props(iml):

    propnames = ['major_axis_length', 'minor_axis_length',
                 'equivalent_diameter', 'solidity']

    region_properties = measure.regionprops(iml, cache=False)

    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)

    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]

    partstats = pd.DataFrame(columns=propnames, data=data)

    return partstats


def props(iml, image_index,im):
    '''populates stats dataframe with partstats given a labelled iamge
    (iml), some sort of image-specific tag for future location matching
    (image_index), and the corrected raw image (im)

    returns:
      partstats

    '''
    # this is crazy - i only want some of these attributes.....
    logger.debug('rprops')
    region_properties = measure.regionprops(iml, cache=False)
    logger.debug('  ok')
#     minor_axis = np.array([el.minor_axis_length for el in stats])

    partstats = pd.DataFrame(index=range(len(region_properties)), columns=['H',
        'S','V','spine length','area','major_axis_length','minor_axis_length',
        'convex area','equivalent_diameter','bbox rmin','bbox cmin','bbox rmax',
        'bbox cmax','perimeter','filled area'] )
    for i, el in enumerate (region_properties):
        hsv = get_color_stats(im,el.bbox,el.image)
        partstats['H'][i] = hsv[0]
        partstats['S'][i] = hsv[1]
        partstats['V'][i] = hsv[2]

        #partstats['spine length'][i] = get_spine_length(el.image)
        partstats['spine length'][i] = np.nan
        partstats['area'][i] = el.area
        partstats['major_axis_length'][i] = el.major_axis_length
        partstats['minor_axis_length'][i] = el.minor_axis_length
        partstats['convex area'][i] = el.convex_area
        partstats['equivalent_diameter'][i] = el.equivalent_diameter
        partstats['bbox rmin'][i] = el.bbox[0]
        partstats['bbox cmin'][i] = el.bbox[1]
        partstats['bbox rmax'][i] = el.bbox[2]
        partstats['bbox cmax'][i] = el.bbox[3]
        partstats['perimeter'][i] = el.perimeter
        partstats['filled area'][i] = el.filled_area

#        image_data = Partstats(get_spine_length(el.image),image_index, el.area,
#                el.major_axis_length, el.minor_axis_length,
#                el.convex_area, el.equivalent_diameter,
#                el.bbox,el.perimeter, el.filled_area,
#                hsv)
#        partstats.append(image_data)

#    partstats = []
#    for i, el in enumerate (region_properties):
#        hsv = get_color_stats(im,el.bbox,el.image)
#
#        image_data = Partstats(get_spine_length(el.image),image_index, el.area,
#                el.major_axis_length, el.minor_axis_length,
#                el.convex_area, el.equivalent_diameter,
#                el.bbox,el.perimeter, el.filled_area,
#                hsv)
#        partstats.append(image_data)

    return partstats


def get_spine_length(imbw):
    ''' extracts the spine length of particles from a binary particle image
    (imbw)

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
    roi = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return roi


def get_color_stats(im, bbox, imbw):
    '''extracts HSV averages inside a particle
    requries:
      im (the corrected raw image)
      bbox (bounding box of the particle image)
      imbw (segmented particle image - of shape determined by bbox)
    '''
    hsv = np.array([np.nan, np.nan, np.nan])

    #roi = np.uint8(extract_roi(im,bbox))
    #hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    #hsv_roi = roi

    #hsv_mask = imbw[:,:,None] * hsv_roi
    #hsv_mask = np.float64(hsv_mask)
    #hsv_mask[hsv_mask==0] = np.nan

    #hsv = np.array([0, 0, 0])
    #hsv[0] = np.nanmedian(hsv_mask[:,:,0])
    #hsv[1] = np.nanmedian(hsv_mask[:,:,1])
    #hsv[2] = np.nanmedian(hsv_mask[:,:,2])
    return hsv


def measure_particles(imbw, imc, max_particles):
    '''Measures properties of particles

    Parameters:
      imbw (full-frame binary image)
      imc (full-frame corrected raw image)
      image_index (some sort of tag for location matching)

    Returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)

    TODO: handle situation when too many particles are found
    TODO: handle situation when zero particles are found
    '''

    iml = morphology.label(imbw > 0)
    logger.info('  {0} particles found'.format(iml.max()))

    if (iml.max() > max_particles):
        logger.warn('....that''s way too many particles! Skipping image.')
        iml *= 0

    stats = fast_props(iml)
    
    return stats


def statextract(imc, settings):
    '''extracts statistics of particles in imc (raw corrected image)

    returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)
    '''
    logger.debug('segment')
    imbw = im2bw(imc, settings.threshold)

    logger.debug('clean')
    imbw = clean_bw(imbw, settings.minimum_area)

    imbw = ndi.binary_fill_holes(imbw)

    logger.debug('measure')
    stats = measure_particles(imbw, imc, settings.max_particles)

    return stats, imbw
