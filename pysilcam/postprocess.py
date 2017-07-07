import pandas as pd
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from skimage.filters.rank import median
from skimage.morphology import disk
import skimage
import pysilcam.process as scpr
from scipy import ndimage as ndi
import skimage
from skimage.exposure import rescale_intensity
import h5py
import logging


logger = logging.getLogger(__name__)
    logger.debug(iniparts - endparts,' particles removed.')
    logger.debug(endparts,' particles measured.')
def d50_from_stats(stats, settings):
    '''calculate the d50 from the stats and settings
    '''

    # the volume distribution needs calculating first
    dias, vd = vd_from_stats(stats, settings)

    # then the d50
    d50 = d50_from_vd(vd,dias)
    return d50

def d50_from_vd(vd,dias):
    ''' calculate d50 from a volume distribution
    d50 = d50_from_vd(vd,dias)
    '''

    # calcualte cumulative sum of the volume distribution
    csvd = np.cumsum(vd/np.sum(vd))

    # find the 50th percentile and interpolate if necessary
    d50 = np.interp(0.5,csvd,dias)
    return d50

def get_size_bins():
    '''retrieve size bins for PSD analysis
    bin_mids_um, bin_limits_um = get_size_bins()
    '''
    # pre-allocate
    bin_limits_um = np.zeros((53),dtype=np.float64)

    # define the upper limit of the smallest bin (same as LISST-100x type-c)
    bin_limits_um[0] = 2.72 * 0.91

    # loop through 53 size classes and calculate the bin limits
    for I in np.arange(1,53,1):
        # each bin is 1.18 * larger than the previous
        bin_limits_um[I] = bin_limits_um[I-1] * 1.180

    # pre-allocate
    bin_mids_um = np.zeros((52),dtype=np.float64)

    # define the middle of the smallest bin (same as LISST-100x type-c)
    bin_mids_um[0] = 2.72

    # loop through 53 size classes and calculate the bin mid-points
    for I in np.arange(1,52,1):
        # each bin is 1.18 * larger than the previous
        bin_mids_um[I]=bin_mids_um[I-1]*1.180

    return bin_mids_um, bin_limits_um

def vd_from_nd(count,psize,sv=1):
    ''' calculate volume concentration from particle count

    sv = sample volume size (litres)

    e.g:
    sample_vol_size=25*1e-3*(1200*4.4e-6*1600*4.4e-6); %size of sample volume in m^3
    sv=sample_vol_size*1e3; %size of sample volume in litres
    '''

    psize = psize *1e-6  # convert to m

    pvol = 4/3 *np.pi * (psize/2)**3  # volume in m^3

    tpvol = pvol * count * 1e9  # volume in micro-litres

    vd = tpvol / sv  # micro-litres / litre

    return vd


def nc_from_nd(count,sv):
    ''' calculate the number concentration from the count and sample volume
    '''
    nc = np.sum(count) / sv
    return nc

def nc_vc_from_stats(stats, settings):
    ''' calculate:
            number concentration
            volume concentration
            total sample volume
            junge distribution slope

    USEAGE: nc, vc, sample_volume, junge = nc_vc_from_stats(stats, settings)

    '''
    # calculate the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # get the path length from the config file
    path_length = settings.path_length

    # get pixel_size from config file
    pix_size = settings.pix_size

    # calcualte the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length=path_length, imx=2048, imy=2448)

    # count the number of images analysed
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images recorded
    sample_volume *= nims

    # calculate the volume distribution from the number distribution
    vd = vd_from_nd(necd, dias, sample_volume)

    # calculate the volume concentration
    vc = np.sum(vd)

    # calculate the number concentration
    nc = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    nd = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan

    # calcualte the junge distirbution slope
    junge = get_j(dias,nd)

    return nc, vc, sample_volume, junge


def nd_from_stats_scaled(stats, settings):
    ''' calcualte a scaled number distribution from stats and settings
    units of nd are in number per micron per litre
    '''
    # calculate the number distirbution (number per bin per sample volume) 
    dias, necd = nd_from_stats(stats,settings)

    # calculate the sample volume per image
    sample_volume = get_sample_volume(settings.pix_size,
            path_length=settings.path_length)

    # count the number of images
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images
    sample_volume *= nims

    # re-scale the units of the number distirbution into number per micron per
    # litre
    nd = nd_rescale(dias, necd, sample_volume)

    # nan the first bin in measurement because it will always be part full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan

    return dias, nd


def nd_from_stats(stats, settings):
    ''' calcualte  number distirbution from stats
    units are number per bin per sample volume
    '''

    # convert the equiv diameter from pixels into microns
    ecd = stats['equivalent_diameter'] * settings.pix_size

    # ignore nans
    ecd = ecd[~np.isnan(ecd)]

    # get the size bins into which particles will be counted
    dias, bin_limits_um = get_size_bins()

    # count particles into size bins
    necd, edges = np.histogram(ecd,bin_limits_um)

    # make it float so other operations are easier later
    necd = np.float64(necd)

    return dias, necd


def vd_from_stats(stats, settings):
    ''' calculate volume distribution from stats
    units of miro-litres per sample volume
    '''

    # obtain the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # convert the number distribution to volume in units of micro-litres per
    # sample volume
    vd = vd_from_nd(necd,dias)

    return dias, vd


class TimeIntegratedVolumeDist:
    ''' class used for summarising recent stats in real-time

    @todo - re-implement this later
    '''
    def __init__(self, settings):
        self.settings = settings
        self.window_size = settings.window_size
        self.times = []
        self.vdlist = []

        self.vd_mean = None
        self.dias = None

    def update_from_stats(self, stats, timestamp):
        '''Update size distribution from stats'''
        dias, vd = vd_from_stats(stats, self.settings)
        self.dias = dias

        #Add the new data
        self.times.append(timestamp)
        self.vdlist.append(vd)

        #Remove data until we are within window size
        while (timestamp - self.times[0]).seconds > self.window_size:
            self.times.pop(0)
            self.vdlist.pop(0)

        #Calculate time-integrated volume distribution
        if len(self.vdlist)>1:
            self.vd_mean = np.nanmean(self.vdlist, axis=0)
        else:
            self.vd_mean = self.vdlist[0]


def montage_maker(roifiles, roidir, pixel_size, msize=2048, brightness=255,
        tightpack=False):
    '''
    makes nice looking matages from a directory of extracted particle images

    use make_montage to call this function
    '''

    # pre-allocate an empty canvas
    montage = np.zeros((msize,msize,3),dtype=np.uint8())
    # pre-allocate an empty test canvas
    immap_test = np.zeros_like(montage[:,:,0])
    print('making a montage - this might take some time....')
    logger.debug('making a montage - this might take some time....')

    # loop through each extracted particle and attempt to add it to the canvas
    for files in roifiles:
        # get the particle image from the HDF5 file
        particle_image = export_name2im(files, roidir)

        # measure the size of this image
        [height, width] = np.shape(particle_image[:,:,0])

        # sanity-check on the particle image size
        if height >= msize:
            continue
        if width >= msize:
            continue

        # contrast exploding:
        particle_image = explode_contrast(particle_image)
        particle_image = np.float64(particle_image)

        # eye-candy normalization:
        peak = np.median(particle_image.flatten())
        bm = brightness - peak
        particle_image = np.float64(particle_image) + bm
        particle_image[particle_image>255] = 255


        # tighpack checks fitting within the canvas based on an approximation
        # of the particle area. If not tightpack, then the fitting will be done
        # based on bounding boxes instead
        if tightpack:
            imbw = scpr.im2bw_fancy(np.uint8(particle_image[:,:,0]), 0.95)
            imbw = ndi.binary_fill_holes(imbw)

            for J in range(5):
                imbw = skimage.morphology.binary_dilation(imbw)

        # initialise a counter
        counter = 0

        # try five times to fit the particle to the canvas by randomly moving
        # it around
        while (counter < 5):
            r = np.random.randint(1,msize-height)
            c = np.random.randint(1,msize-width)

            # tighpack checks fitting within the canvas based on an approximation
            # of the particle area. If not tightpack, then the fitting will be done
            # based on bounding boxes instead
            if tightpack:
                test = np.max(immap_test[r:r+height,c:c+width]+imbw)
            else:
                test = np.max(immap_test[r:r+height,c:c+width,None]+1)


            # if the new particle is overlapping an existing object in the
            # canvas, then try again and increment the counter
            if (test>1):
                counter += 1
            else:
                break

        # if we reach this point and there is still an overlap, then forget
        # this particle, and move on
        if (test>1):
            continue

        # if we reach here, then the particle has found a position in the
        # canvas with no overlap, and can then be inserted into the canvas
        montage[r:r+height,c:c+width,:] = np.uint8(particle_image)

        # update the testing canvas so it is ready for the next particle
        if tightpack:
            immap_test[r:r+height,c:c+width] = imbw
        else:
            immap_test[r:r+height,c:c+width,None] = immap_test[r:r+height,c:c+width,None]+1

    # now the montage is finished
    # here are some small eye-candy scaling things to tidy up
    montageplot = np.copy(montage)
    montageplot[montage>255] = 255
    montageplot[montage==0] = 255
    print('montage complete')
    logger.debug('montage complete')

    return montageplot


def make_montage(stats_csv_file, pixel_size, roidir, min_length=100,
        auto_scaler=500, msize=1024, max_length=5000):
    ''' wrapper function for montage_maker 
    '''

    # obtain particle statistics from the csv file
    stats = pd.read_csv(stats_csv_file)

    # remove nans because concentrations are not important here
    stats = stats[~np.isnan(stats['major_axis_length'])]

    # sort the particles based on their length
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)

    roifiles = stats['export name'].loc[
            (stats['major_axis_length'] * pixel_size > min_length) &
            (stats['major_axis_length'] * pixel_size < max_length)
            ].values

    # subsample the particles if necessary
    print('rofiles:',len(roifiles))
    logger.debug('rofiles:',len(roifiles))
    IMSTEP = np.max([np.int(np.round(len(roifiles)/auto_scaler)),1])
    print('reducing particles by factor of {0}'.format(IMSTEP))
    logger.debug('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0,len(roifiles),IMSTEP)]
    print('rofiles:',len(roifiles))
    logger.debug('rofiles:',len(roifiles))

    montage = montage_maker(roifiles, roidir, pixel_size, msize)

    return montage


def get_sample_volume(pix_size, path_length=10, imx=2048, imy=2448):
    ''' calculate the sample volume of one image
    '''
    sample_volume_litres = imx*pix_size/1000 * imy*pix_size/1000 * path_length*1e-6

    return sample_volume_litres


def get_j(dias, nd):
    ''' calculates the junge slope from a correctly-scale number distribution
    (number per micron per litre must be the units of nd)
    '''
    # conduct this calculation only on the part of the size distribution where
    # LISST-100 and SilCam data overlap
    ind = np.isfinite(dias) & np.isfinite(nd) & (dias<300) & (dias>150)

    # use polyfit to obtain the slope of the ditriubtion in log-space (which is
    # assumed near-linear in most parts of the ocean)
    p = np.polyfit(np.log(dias[ind]),np.log(nd[ind]),1)
    j = p[0]
    return j


def count_images_in_stats(stats):
    ''' count the number of raw images used to generate stats
    '''
    u = stats['timestamp'].unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats,settings,n=0):
    ''' return statistics of the nth largest particle
    '''
    stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def extract_nth_longest(stats,settings,n=0):
    ''' return statistics of the nth longest particle
    '''
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def explode_contrast(im):
    ''' eye-candy function for exploding the contrast of a particle iamge (roi)
    '''
    # make sure iamge is float
    im = np.float64(im)

    # re-scale the instensities in the image to chop off some ends
    p1, p2 = np.percentile(im, (0, 80))
    im_mod = rescale_intensity(im, in_range=(p1, p2))
                
    # set minimum value to zero
    im_mod -= np.min(im_mod)

    # set maximum value to one
    im_mod /= np.max(im_mod)

    # re-scale to match uint8 max
    im_mod *= 255

    # convert to unit8
    im_mod = np.uint8(im_mod)
    return im_mod


def bright_norm(im,brightness=255):
    ''' eye-candy function for normalising the image brightness
    '''
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im>255] = 255

    im =np.uint8(im)
    return im


def nd_rescale(dias, nd, sample_volume):
    ''' rescale a number distribution from
            number per bin per sample volume
        to
            number per micron per litre
    '''
    nd = np.float64(nd) / sample_volume # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    nd /= dd
    nd[nd<0] = np.nan # and nan impossible values!

    return nd

def add_depth_to_stats(stats, time, depth):
    ''' if you have a depth time-series, use this function to find the depth of
    each line in stats
    '''
    # get times
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def export_name2im(exportname, path):
    ''' returns an image from the export name string in the -STATS.csv file

    get the exportname like this: exportname = stats['export name'].values[0]
    '''

    # the particle number is defined after the time info
    pn = exportname.split('-')[1]
    # the name is the first bit
    name = exportname.split('-')[0] + '.h5'

    # combine the name with the location of the exported HDF5 files
    fullname = os.path.join(path, name)

    # open the H5 file
    fh = h5py.File(fullname ,'r')

    # extract the particle image of interest
    im = fh[pn]

    return im
