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
from scipy import signal
from scipy import interpolate

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
                 'equivalent_diameter']

    region_properties = measure.regionprops(iml, cache=False)

    data = np.zeros((len(region_properties), len(propnames)), dtype=np.float64)

    for i, el in enumerate(region_properties):
        data[i, :] = [getattr(el, p) for p in propnames]

    partstats = pd.DataFrame(columns=propnames, data=data)

    return partstats


def props_og(iml, imc, settings):

    region_properties = measure.regionprops(iml, cache=False)

    ecd = np.zeros(settings.Process.max_particles, dtype=np.float64)
    gas = np.copy(ecd)

    for i, el in enumerate (region_properties):

        mmr = el.minor_axis_length / el.major_axis_length
        if mmr < settings.Process.min_deformation:
            continue

        if el.solidity < settings.Process.min_solidity:
            continue

        ecd[i] = el.equivalent_diameter

        if settings.PostProcess.pix_size * ecd[i] > 150 and i > 246:
            roi = extract_roi(imc, el.bbox)

            prof = 1 / np.sum(roi, axis=0)
            y = np.copy(prof)
            x = np.arange(0, len(y))

            spl = interpolate.UnivariateSpline(x, y, s=1e-10, k=4)
            xs = np.linspace(0, len(y), 1000)

            yi = spl(xs)

            pinds = spl.derivative().roots()

            print('pinds:', pinds)

            if len(pinds)>1 and len(pinds)<4:
                gas[i] = 1

            plt.close('all')
            f, a = plt.subplots(2,1)
            a[0].plot(x, prof, 'k--')
            if gas[i]==1:
                a[0].plot(xs, yi, color='g')
            else:
                a[0].plot(xs, yi, color='r')
            a[1].imshow(roi, cmap='gray')
            print('particle',i)
            plt.show()

    ecd = ecd[ecd>0]

    partstats = pd.DataFrame(columns=['equivalent_diameter'], data=ecd)

    return partstats


def find_peaks(x, f):
    '''Find peaks in signal f(x)'''
    #wsize = 400
    wsize = np.max([5, len(x)/6]) * 4
    finterp = interpolate.UnivariateSpline(x, f, s=0.0, k=3)
    xi = np.linspace(x[0], x[-1], x.size*4)
    fi = finterp(xi)

    num_windows = max(1, xi.size//wsize)

    offset = 0
    peaks = []
    for xk, fk in zip(np.array_split(xi, num_windows), 
                      np.array_split(fi, num_windows)):
        peaks += [i+offset for i in signal.find_peaks_cwt(fk, np.arange(10,50))]
        offset += len(xk)

    return xi, fi, peaks


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


def measure_particles(imbw, imc, settings):
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

    if (iml.max() > settings.Process.max_particles):
        logger.warn('....that''s way too many particles! Skipping image.')
        iml *= 0

    #stats = fast_props(iml)
    stats = props_og(iml, imc, settings)
    
    return stats


def find_gas(imbw, imc, stats):
    pass


def is_gas():
    pass


def statextract(imc, settings):
    '''extracts statistics of particles in imc (raw corrected image)

    returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)
    '''
    logger.debug('segment')
    imbw = im2bw(imc, settings.Process.threshold)

    logger.debug('clean')
    imbw = clean_bw(imbw, settings.Process.minimum_area)

    imbw = ndi.binary_fill_holes(imbw)

    logger.debug('measure')
    stats = measure_particles(imbw, imc, settings)

    return stats, imbw
