# -*- coding: utf-8 -*-

import numpy as np
from skimage import morphology
from skimage import segmentation
from skimage.filters.rank import median
from skimage.morphology import disk
from skimage import measure
import pandas as pd
#import cv2
import matplotlib.pyplot as plt

'''
module for processing SilCam data

TODO: sort out settings
TODO: add tests for this module
'''

SETTINGS = dict(THRESH = 0.9, # higher THRESH is higher sensitivity
                min_area = 12,
                max_particles = 8000)

def im2bw(imc,greythresh):
    ''' converts corrected image (imc) to a binary image
    using greythresh as the threshold value (some auto-scaling of greythresh is done inside)

    returns:
      imbw (binary image)
    '''
    img = np.uint8(np.min(imc,2)) # sensibly squash RGB color space
    #img = median(img, disk(4)) # apply median filter to remove noise
    thresh = np.uint8(greythresh * np.median(img)) # determine auto-amazing theshold estimate

    imbw = np.invert(img > thresh) # segment the image
    # correct(ish) for blur in median filter for disk(4) - requires 2
    # iterations of a single pixel dilation for a disk size of 4
    #imbw = morphology.binary_erosion(imbw)
    #imbw = morphology.binary_erosion(imbw)

    return imbw

def clean_bw(imbw,min_area):
    '''cleans up particles which are too small and particles touching the
    border
    '''
    imbw = morphology.remove_small_objects(imbw>0,min_size=min_area)
    imbw = segmentation.clear_border(imbw, in_place=True) # remove particles touching the
    # border of the image

    # remove objects smaller the min_area
    return imbw


def fast_props(iml):

    propnames = ['major_axis_length','minor_axis_length',
        'equivalent_diameter']

    region_properties = measure.regionprops(iml,cache=False)

    data = np.zeros((len(region_properties), 3), dtype=np.float64)

    for i, el in enumerate (region_properties):
        data[i,:] = [getattr(el, p) for p in propnames]
        

    partstats = pd.DataFrame(columns=propnames, data=data)

    return partstats

def props(iml,image_index,im):
    '''population the Partstats class with partstats given a labelled iamge
    (iml), some sort of image-specific tag for future location matching
    (image_index), and the corrected raw image (im)

    returns:
      partstats

    '''
    # this is crazy - i only want some of these attributes.....
    print('rprops')
    region_properties = measure.regionprops(iml,cache=False)
    print('  ok')
#     minor_axis = np.array([el.minor_axis_length for el in stats])


    partstats = pd.DataFrame(index=range(len(region_properties)), columns=['H',
        'S','V','spine length','area','major axis','minor axis',
        'convex area','equiv diam','bbox rmin','bbox cmin','bbox rmax',
        'bbox cmax','perimeter','filled area'] )
    for i, el in enumerate (region_properties):
        hsv = get_color_stats(im,el.bbox,el.image)
        partstats['H'][i] = hsv[0]
        partstats['S'][i] = hsv[1]
        partstats['V'][i] = hsv[2]

        #partstats['spine length'][i] = get_spine_length(el.image)
        partstats['spine length'][i] = np.nan
        partstats['area'][i] = el.area
        partstats['major axis'][i] = el.major_axis_length
        partstats['minor axis'][i] = el.minor_axis_length
        partstats['convex area'][i] = el.convex_area
        partstats['equiv diam'][i] = el.equivalent_diameter
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

def extract_roi(im,bbox):
    ''' given an image (im) and bounding box (bbox), this will return the roi

    returns:
      roi
    '''
    roi = im[bbox[0]:bbox[2],bbox[1]:bbox[3]]
    return roi

def get_color_stats(im,bbox,imbw):
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

def measure_particles(imbw,imc,image_index):
    '''measures properties of particles
    requries:
      imbw (full-frame binary image)
      imc (full-frame corrected raw image)
      image_index (some sort of tag for location matching)

    returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)

    TODO: handle situation when too many particles are found
    TODO: handle situation when zero particles are found
    '''

    iml = morphology.label(imbw>0)
    print('  ',iml.max(),'particles found')

    if (iml.max()>SETTINGS['max_particles']):
        print('....that''s way too many particles! Skipping image.')
        stats = np.nan
    elif (iml.max()==0):
        stats = np.nan
    else:
        stats = fast_props(iml)
    
    return stats

def statextract(imc,image_index):
    '''extracts statistics of particles in imc (raw corrected image) with some
    sort of tag (image_index) used for location matching later

    returns:
      stats (list of particle statistics for every particle, according to
      Partstats class)
    '''
    print('segment')
    imbw = im2bw(imc,SETTINGS['THRESH'])

    print('clean')
    imbw = clean_bw(imbw,SETTINGS['min_area'])

    print('measure')
    stats = measure_particles(imbw,imc,image_index)

    return stats
