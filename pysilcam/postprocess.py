import pandas as pd
import numpy as np
import os
import imageio as imo
from scipy import ndimage as ndi
import skimage
from skimage.exposure import rescale_intensity
import h5py
from pysilcam.config import PySilcamSettings
from enum import Enum
from tqdm import tqdm
import logging
from pysilcam.process import write_stats

logger = logging.getLogger(__name__)


class outputPartType(Enum):
    '''
    Enum class for all (1), oil (2) or gas (3)
    '''
    all = 1
    oil = 2
    gas = 3


def d50_from_stats(stats, settings):
    '''
    Calculate the d50 from the stats and settings
    
    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings
        
    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # the volume distribution needs calculating first
    dias, vd = vd_from_stats(stats, settings)

    # then the d50
    d50 = d50_from_vd(vd, dias)
    return d50


def d50_from_vd(vd, dias):
    '''
    Calculate d50 from a volume distribution

    Args:
        vd            : particle volume distribution calculated from vd_from_stats()
        dias          : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        d50 (float)                 : the 50th percentile of the cumulative sum of the volume distributon, in microns
    '''

    # calculate cumulative sum of the volume distribution
    csvd = np.cumsum(vd / np.sum(vd))

    # find the 50th percentile and interpolate if necessary
    d50 = np.interp(0.5, csvd, dias)
    return d50


def get_size_bins():
    '''
    Retrieve size bins for PSD analysis

    Returns:
        bin_mids_um (array)     : mid-points of size bins
        bin_limits_um (array)   : limits of size bins
    '''
    # pre-allocate
    bin_limits_um = np.zeros((53), dtype=np.float64)

    # define the upper limit of the smallest bin (same as LISST-100x type-c)
    bin_limits_um[0] = 2.72 * 0.91

    # loop through 53 size classes and calculate the bin limits
    for bin_number in np.arange(1, 53, 1):
        # each bin is 1.18 * larger than the previous
        bin_limits_um[bin_number] = bin_limits_um[bin_number - 1] * 1.180

    # pre-allocate
    bin_mids_um = np.zeros((52), dtype=np.float64)

    # define the middle of the smallest bin (same as LISST-100x type-c)
    bin_mids_um[0] = 2.72

    # loop through 53 size classes and calculate the bin mid-points
    for bin_number in np.arange(1, 52, 1):
        # each bin is 1.18 * larger than the previous
        bin_mids_um[bin_number] = bin_mids_um[bin_number - 1] * 1.180

    return bin_mids_um, bin_limits_um


def filter_stats(stats, crop_stats):
    '''
    Filters stats file based on whether the particles are
    within a rectangle specified by crop_stats.
    A temporary cropping solution due to small window in AUV

    Args:
        stats (df)    : silcam stats file
        crop_stats (tuple) : 4-tuple of lower-left (row, column) then upper-right (row, column) coord of crop

    Returns:
        stats (df)    : cropped silcam stats file
    '''

    r = np.array(((stats['maxr'] - stats['minr']) / 2) + stats['minr'])  # pixel row of middle of bounding box
    c = np.array(((stats['maxc'] - stats['minc']) / 2) + stats['minc'])  # pixel column of middle of bounding box

    pts = np.array([[(r_, c_)] for r_, c_ in zip(r, c)])
    pts = pts.squeeze()

    ll = np.array(crop_stats[:2])  # lower-left
    ur = np.array(crop_stats[2:])  # upper-right

    inidx = np.all(np.logical_and(ll <= pts, pts <= ur), axis=1)
    stats = stats[inidx]

    return stats


def vd_from_nd(count, psize, sv=1):
    '''
    Calculate volume concentration from particle count

    sv = sample volume size (litres)

    e.g:
    sample_vol_size=25*1e-3*(1200*4.4e-6*1600*4.4e-6); %size of sample volume in m^3
    sv=sample_vol_size*1e3; %size of sample volume in litres

    Args:
        count (array) : particle number distribution
        psize (float) : pixel size of the SilCam contained in settings.PostProcess.pix_size from the config ini file
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations

    Returns:
        vd (array)    : the particle volume distribution
    '''

    psize = psize * 1e-6  # convert to m

    pvol = 4 / 3 * np.pi * (psize / 2)**3  # volume in m^3

    tpvol = pvol * count * 1e9  # volume in micro-litres

    vd = tpvol / sv  # micro-litres / litre

    return vd


def nc_from_nd(count, sv):
    '''
    Calculate the number concentration from the count and sample volume

    Args:
        count (array) : particle number distribution
        sv=1 (float)  : the volume of the sample which should be used for scaling concentrations

    Returns:
        nc (float)    : the total number concentration in #/L
    '''
    nc = np.sum(count) / sv
    return nc


def nc_vc_from_stats(stats, settings, oilgas=outputPartType.all):
    '''
    Calculates important summary statistics from a stats DataFrame

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings
        oilgas=oc_pp.outputPartType.all : the oilgas enum if you want to just make the figure for oil, or just gas (defaults to all particles)

    Returns:
        nc (float)            : the total number concentration in #/L
        vc (float)            : the total volume concentration in uL/L
        sample_volume (float) : the total volume of water sampled in L
        junge (float)         : the slope of a fitted juge distribution between 150-300um
    '''
    # get the path length from the config file
    path_length = settings.path_length

    # get pixel_size from config file
    pix_size = settings.pix_size

    # calculate the sample volume per image
    sample_volume = get_sample_volume(pix_size, path_length=path_length, imx=2048, imy=2448)

    # count the number of images analysed
    nims = count_images_in_stats(stats)

    # scale the sample volume by the number of images recorded
    sample_volume *= nims

    # extract only wanted particle stats
    if oilgas == outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
    elif oilgas == outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)

    # it is possible when in realtime mode and using multiprocess, to have data where there are no stats after extracting oil or
    # gas. Here which check for this and return zero concentration and a nan junge slope (as the PSD is non-existent)
    if len(stats) == 0:
        nc = 0
        vc = 0
        junge = np.nan
        return nc, vc, sample_volume, junge

    # calculate the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # calculate the volume distribution from the number distribution
    vd = vd_from_nd(necd, dias, sample_volume)

    # calculate the volume concentration
    vc = np.sum(vd)

    # calculate the number concentration
    nc = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    nd = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd > 0)
    nd[ind[0]] = np.nan

    # calcualte the junge distirbution slope
    junge = get_j(dias, nd)

    return nc, vc, sample_volume, junge


def nd_from_stats_scaled(stats, settings):
    ''' calcualte a scaled number distribution from stats and settings
    units of nd are in number per micron per litre

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings

    Returns:
        dias                        : mid-points of size bins
        nd                          : number distribution in number/micron/litre
    '''
    # calculate the number distirbution (number per bin per sample volume)
    dias, necd = nd_from_stats(stats, settings)

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
    ind = np.argwhere(nd > 0)
    nd[ind[0]] = np.nan

    return dias, nd


def nd_from_stats(stats, settings):
    ''' calcualte  number distirbution from stats
    units are number per bin per sample volume

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings

    Returns:
        dias                        : mid-points of size bins
        necd                        : number distribution in number/size-bin/sample-volume
    '''

    # convert the equiv diameter from pixels into microns
    ecd = stats['equivalent_diameter'] * settings.pix_size

    # ignore nans
    ecd = ecd[~np.isnan(ecd)]

    # get the size bins into which particles will be counted
    dias, bin_limits_um = get_size_bins()

    # count particles into size bins
    necd, edges = np.histogram(ecd, bin_limits_um)

    # make it float so other operations are easier later
    necd = np.float64(necd)

    return dias, necd


def vd_from_stats(stats, settings):
    ''' calculate volume distribution from stats
    units of miro-litres per sample volume

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        settings (PySilcamSettings) : settings associated with the data, loaded with PySilcamSettings

    Returns:
        dias                        : mid-points of size bins
        vd                          : volume distribution in micro-litres/sample-volume
    '''

    # obtain the number distribution
    dias, necd = nd_from_stats(stats, settings)

    # convert the number distribution to volume in units of micro-litres per
    # sample volume
    vd = vd_from_nd(necd, dias)

    return dias, vd


class TimeIntegratedVolumeDist:
    ''' class used for summarising recent stats in real-time

    Possibly redundant

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

        # Add the new data
        self.times.append(timestamp)
        self.vdlist.append(vd)

        # Remove data until we are within window size
        while (timestamp - self.times[0]).seconds > self.window_size:
            self.times.pop(0)
            self.vdlist.pop(0)

        # Calculate time-integrated volume distribution
        if len(self.vdlist) > 1:
            self.vd_mean = np.nanmean(self.vdlist, axis=0)
        else:
            self.vd_mean = self.vdlist[0]


def montage_maker(roifiles, roidir, pixel_size, msize=2048, brightness=255,
                  tightpack=False, eyecandy=True):
    '''
    makes nice looking matages from a directory of extracted particle images

    use make_montage to call this function

    Args:
        roifiles                : list of roi files obtained from gen_roifiles(stats, auto_scaler=auto_scaler)
        roidir                  : location of roifiles usually defined by settings.ExportParticles.outputpath
        pixel_size              : pixel size of system defined by settings.PostProcess.pix_size
        msize=2048              : size of canvas in pixels
        brightness=255          : brighness of packaged particles
        tightpack=False         : boolean which if True packs particles using segmented alpha-shapes instead of bounding boxes. This is slow, but packs particles more tightly.
        eyecandy=True           : boolean which if True will explode the contrast of packed particles (nice for natural particles, but not so good for oil and gas).

    Returns:
        montageplot             : a nicely-made montage in the form of an image, which can be plotted using plotting.montage_plot(montage, settings.PostProcess.pix_size)
    '''

    if tightpack:
        import pysilcam.process as scpr
    # pre-allocate an empty canvas
    montage = np.zeros((msize, msize, 3), dtype=np.uint8())
    # pre-allocate an empty test canvas
    immap_test = np.zeros_like(montage[:, :, 0])
    logger.info('making a montage - this might take some time....')

    # loop through each extracted particle and attempt to add it to the canvas
    for files in tqdm(roifiles):
        # get the particle image from the HDF5 file
        particle_image = export_name2im(files, roidir)

        # measure the size of this image
        [height, width] = np.shape(particle_image[:, :, 0])

        # sanity-check on the particle image size
        if height >= msize:
            continue
        if width >= msize:
            continue

        if eyecandy:
            # contrast exploding:
            particle_image = explode_contrast(particle_image)

            # eye-candy normalization:
            peak = np.median(particle_image.flatten())
            bm = brightness - peak
            particle_image = np.float64(particle_image) + bm
        else:
            particle_image = np.float64(particle_image)
        particle_image[particle_image > 255] = 255

        # tighpack checks fitting within the canvas based on an approximation
        # of the particle area. If not tightpack, then the fitting will be done
        # based on bounding boxes instead
        if tightpack:
            imbw = scpr.image2blackwhite_accurate(np.uint8(particle_image[:, :, 0]), 0.95)
            imbw = ndi.binary_fill_holes(imbw)

            for J in range(5):
                imbw = skimage.morphology.binary_dilation(imbw)

        # initialise a counter
        counter = 0

        # try five times to fit the particle to the canvas by randomly moving
        # it around
        while (counter < 5):
            r = np.random.randint(1, msize - height)
            c = np.random.randint(1, msize - width)

            # tighpack checks fitting within the canvas based on an approximation
            # of the particle area. If not tightpack, then the fitting will be done
            # based on bounding boxes instead
            if tightpack:
                test = np.max(immap_test[r:r + height, c:c + width] + imbw)
            else:
                test = np.max(immap_test[r:r + height, c:c + width, None] + 1)

            # if the new particle is overlapping an existing object in the
            # canvas, then try again and increment the counter
            if (test > 1):
                counter += 1
            else:
                break

        # if we reach this point and there is still an overlap, then forget
        # this particle, and move on
        if (test > 1):
            continue

        # if we reach here, then the particle has found a position in the
        # canvas with no overlap, and can then be inserted into the canvas
        montage[r:r + height, c:c + width, :] = np.uint8(particle_image)

        # update the testing canvas so it is ready for the next particle
        if tightpack:
            immap_test[r:r + height, c:c + width] = imbw
        else:
            immap_test[r:r + height, c:c + width, None] = immap_test[r:r + height, c:c + width, None] + 1

    # now the montage is finished
    # here are some small eye-candy scaling things to tidy up
    montageplot = np.copy(montage)
    montageplot[montage > 255] = 255
    montageplot[montage == 0] = 255
    logger.info('montage complete')

    return montageplot


def make_montage(stats_file, pixel_size, roidir,
                 auto_scaler=500, msize=1024, maxlength=100000,
                 oilgas=outputPartType.all, crop_stats=None):
    ''' wrapper function for montage_maker

    Args:
        stats_file              : location of the stats_csv file that comes from silcam process
        pixel_size                  : pixel size of system defined by settings.PostProcess.pix_size
        roidir                      : location of roifiles usually defined by settings.ExportParticles.outputpath
        auto_scaler=500             : approximate number of particle that are attempted to be pack into montage
        msize=1024                  : size of canvas in pixels
        maxlength=100000            : maximum length in microns of particles to be included in montage
        oilgas=outputPartType.all   : enum defining which type of particle to be selected for use in the montage
        crop_stats=None             : None or 4-tuple of lower-left then upper-right coord of crop

    Returns:
        montage (uint8)             : a nicely-made montage in the form of an image, which can be plotted using plotting.montage_plot(montage, settings.PostProcess.pix_size)
    '''

    # obtain particle statistics from the stats file
    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    if crop_stats is not None:
        stats = filter_stats(stats, crop_stats)

    # remove nans because concentrations are not important here
    stats = stats[~np.isnan(stats['major_axis_length'])]
    stats = stats[(stats['major_axis_length'] *
                  pixel_size) < maxlength]

    # extract only wanted particle stats
    if oilgas == outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
    elif oilgas == outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)

    # sort the particles based on their length
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)

    roifiles = gen_roifiles(stats, auto_scaler=auto_scaler)

    eyecandy = True
    if not (oilgas == outputPartType.all):
        eyecandy = False

    montage = montage_maker(roifiles, roidir, pixel_size, msize, eyecandy=eyecandy)

    return montage


def gen_roifiles(stats, auto_scaler=500):
    ''' generates a list of filenames suitable for making montages with

    Args:
        stats (DataFrame)           : particle statistics from silcam process
        auto_scaler=500             : approximate number of particle that are attempted to be pack into montage

    Returns:
        roifiles                    : a selection of filenames that can be passed to montage_maker() for making nice montages
    '''

    roifiles = stats['export name'][stats['export name'] != 'not_exported'].values

    # subsample the particles if necessary
    logger.info('rofiles: {0}'.format(len(roifiles)))
    IMSTEP = np.max([np.int(np.round(len(roifiles) / auto_scaler)), 1])
    logger.info('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0, len(roifiles), IMSTEP)]
    logger.info('rofiles: {0}'.format(len(roifiles)))

    return roifiles


def get_sample_volume(pix_size, path_length=10, imx=2048, imy=2448):
    ''' calculate the sample volume of one image

    Args:
        pix_size                    : size of pixels in microns (settings.PostProcess.pixel_size)
        path_length=10              : path length of sample volume in mm
        imx=2048                    : image x dimention in pixels
        imy=2448                    : image y dimention in pixels

    Returns:
        sample_volume_litres        : the volume of the sample volume in litres

    '''
    sample_volume_litres = imx * pix_size / 1000 * imy * pix_size / 1000 * path_length * 1e-6

    return sample_volume_litres


def get_j(dias, nd):
    ''' calculates the junge slope from a correctly-scale number distribution
    (number per micron per litre must be the units of nd)

    Args:
        dias                        : mid-point of size bins
        nd                          : number distribution in number per micron per litre

    Returns:
        j                           : Junge slope from fitting of psd between 150 and 300um

    '''
    # conduct this calculation only on the part of the size distribution where
    # LISST-100 and SilCam data overlap
    ind = np.isfinite(dias) & np.isfinite(nd) & (dias < 300) & (dias > 150)

    # use polyfit to obtain the slope of the ditriubtion in log-space (which is
    # assumed near-linear in most parts of the ocean)
    p = np.polyfit(np.log(dias[ind]), np.log(nd[ind]), 1)
    j = p[0]
    return j


def count_images_in_stats(stats):
    ''' count the number of raw images used to generate stats

    Args:
        stats                       : pandas DataFrame of particle statistics

    Returns:
        n_images                    : number of raw images

    '''
    u = pd.to_datetime(stats['timestamp']).unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats, n=0):
    ''' return statistics of the nth largest particle
    '''
    stats_sorted = stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=False)
    stats_sorted = stats_sorted.iloc[n]
    return stats_sorted


def extract_nth_longest(stats, n=0):
    ''' return statistics of the nth longest particle
    '''
    stats_sorted = stats.sort_values(by=['major_axis_length'], ascending=False, inplace=False)
    stats_sorted = stats_sorted.iloc[n]
    return stats_sorted


def d50_timeseries(stats, settings):
    ''' Calculates time series of d50 from stats

    Args:
        stats               : pandas DataFrame of particle statistics
        settings            : PySilCam settings

    Returns:
        d50                 : time series of median particle diameter (50th percentile of volume distribution)
        time                : timestamps for d50 values

    '''

    from tqdm import tqdm

    stats.sort_values(by=['timestamp'], inplace=True)

    td = pd.to_timedelta('00:00:' + str(settings.window_size / 2.))
    d50 = []
    time = []

    u = pd.to_datetime(stats['timestamp']).unique()

    for t in tqdm(u):
        dt = pd.to_datetime(t)
        stats_ = stats[(pd.to_datetime(stats['timestamp']) < (dt + td)) & (pd.to_datetime(stats['timestamp']) > (dt - td))]
        d50.append(d50_from_stats(stats_, settings))
        time.append(t)

    if len(time) == 0:
        d50 = np.nan
        time = np.nan

    return d50, time


def explode_contrast(im):
    ''' eye-candy function for exploding the contrast of a particle iamge (roi)

    Args:
        im   (uint8)       : image (normally a particle ROI)

    Returns:
        im_mod (uint8)     : image following exploded contrast

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


def bright_norm(im, brightness=255):
    ''' eye-candy function for normalising the image brightness

    Args:
        im   (uint8)    : image
        brightness=255  : median of histogram will be shifted to align with this value

    Return:
        im   (uint8)    : image with modified brightness

    '''
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im > 255] = 255

    im = np.uint8(im)
    return im


def nd_rescale(dias, nd, sample_volume):
    ''' rescale a number distribution from
            number per bin per sample volume
        to
            number per micron per litre

    Args:
        dias                : mid-points of size bins
        nd                  : unscaled number distribution
        sample_volume       : sample volume of each image

    Returns:
        nd                  : scaled number distribution (number per micron per litre)
    '''
    nd = np.float64(nd) / sample_volume  # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    nd /= dd
    nd[nd < 0] = np.nan  # and nan impossible values!

    return nd


def add_depth_to_stats(stats, time, depth):
    ''' if you have a depth time-series, use this function to find the depth of
    each line in stats

    Args:
        stats               : pandas DataFrame of particle statistics
        time                : time stamps associated with depth argument
        depth               : depths associated with the time argument

    Return:
        stats               : pandas DataFrame of particle statistics, now with a depth column
    '''
    # get times
    sctime = pd.to_datetime(stats['timestamp'])
    # interpolate depths into the SilCam times
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def export_name2im(exportname, path):
    ''' returns an image from the export name string in the -STATS.h5 file

    get the exportname like this: exportname = stats['export name'].values[0]

    Args:
        exportname              : string containing the name of the exported particle e.g. stats['export name'].values[0]
        path                    : path to exported h5 files

    Returns:
        im                      : particle ROI image

    '''

    # the particle number is defined after the time info
    pn = exportname.split('-')[1]
    # the name is the first bit
    name = exportname.split('-')[0] + '.h5'

    # combine the name with the location of the exported HDF5 files
    fullname = os.path.join(path, name)

    # open the H5 file
    fh = h5py.File(fullname, 'r')

    # extract the particle image of interest
    im = fh[pn]

    return im


def extract_latest_stats(stats, window_size):
    ''' extracts the stats data from within the last number of seconds specified
    by window_size.

    Args:
        stats                   : pandas DataFrame of particle statistics
        window_size             : number of seconds to extract from the end of the stats data

    Returns:
        stats dataframe (from the last window_size seconds)
    '''
    end = np.max(pd.to_datetime(stats['timestamp']))
    start = end - pd.to_timedelta('00:00:' + str(window_size))
    stats = stats[pd.to_datetime(stats['timestamp']) > start]
    return stats


def silc_to_bmp(directory):
    '''Convert a directory of silc files to bmp images

    Args:
        directory               : path of directory to convert

    '''
    files = [s for s in os.listdir(directory) if s.endswith('.silc')]

    for f in files:
        try:
            with open(os.path.join(directory, f), 'rb') as fh:
                im = np.load(fh, allow_pickle=False)
                fout = os.path.splitext(f)[0] + '.bmp'
            outname = os.path.join(directory, fout)
            imo.imwrite(outname, im)
        except:
            logger.warning('{0} failed!'.format(f))
            continue

    logger.info('Done.')


def make_timeseries_vd(stats, settings):
    '''makes a dataframe of time-series volume distribution and d50

    Args:
        stats (silcam stats dataframe): loaded from a *-STATS.h5 file
        settings (silcam settings): loaded from PySilCamSettings

    Returns:
        dataframe: of time series volume concentrations are in uL/L columns with number headings are diameter min-points
    '''

    from tqdm import tqdm

    stats['timestamp'] = pd.to_datetime(stats['timestamp'])

    u = stats['timestamp'].unique()

    sample_volume = get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    vdts = []
    d50 = []
    timestamp = []
    dias = []
    for s in tqdm(u):
        dias, vd = vd_from_stats(stats[stats['timestamp'] == s],
                                 settings.PostProcess)
        nims = count_images_in_stats(stats[stats['timestamp'] == s])
        sv = sample_volume * nims
        vd /= sv
        d50_ = d50_from_vd(vd, dias)
        d50.append(d50_)
        timestamp.append(pd.to_datetime(s))
        vdts.append(vd)

    if len(vdts) == 0:
        dias, limits = get_size_bins()
        vdts = np.zeros_like(dias) * np.nan

        time_series = pd.DataFrame(data=[np.squeeze(vdts)], columns=dias)

        time_series['D50'] = np.nan
        time_series['Time'] = np.nan

        return time_series

    time_series = pd.DataFrame(data=np.squeeze(vdts), columns=dias)

    time_series['D50'] = d50
    time_series['Time'] = pd.to_datetime(timestamp)

    time_series.sort_values(by='Time', inplace=True, ascending=False)

    return time_series


def stats_to_xls_png(config_file, stats_filename, oilgas=outputPartType.all):
    '''summarises stats in two excel sheets of time-series PSD and averaged
    PSD.

    Args:
        config_file (string)            : Path of the config file for this data
        stats_filename (string)         : Path of the stats csv file
        oilgas=oc_pp.outputPartType.all : the oilgas enum if you want to just make the figure for oil, or just gas (defaults to all particles)

    Returns:
        dataframe: of time series
        files: in the proc folder)
    '''
    settings = PySilcamSettings(config_file)

    stats = pd.read_csv(stats_filename)
    stats.sort_values(by='timestamp', inplace=True)
    oilgasTxt = ''

    if oilgas == outputPartType.oil:
        from pysilcam.oilgas import extract_oil
        stats = extract_oil(stats)
        oilgasTxt = 'oil'
    elif oilgas == outputPartType.gas:
        from pysilcam.oilgas import extract_gas
        stats = extract_gas(stats)
        oilgasTxt = 'gas'

    df = make_timeseries_vd(stats, settings)

    df.to_excel(stats_filename.replace('-STATS.h5', '') +
                '-TIMESERIES' + oilgasTxt + '.xlsx')

    sample_volume = get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

    dias, vd = vd_from_stats(stats,
                             settings.PostProcess)
    nims = count_images_in_stats(stats)
    sv = sample_volume * nims
    vd /= sv

    d50 = d50_from_vd(vd, dias)

    dfa = pd.DataFrame(data=[vd], columns=dias)
    dfa['d50'] = d50

    timestamp = np.min(pd.to_datetime(df['Time']))
    dfa['Time'] = timestamp

    dfa.to_excel(stats_filename.replace('-STATS.h5', '') + '-AVERAGE' + oilgasTxt + '.xlsx')

    return df


def statscsv_to_statshdf(stats_file):
    '''Convert old STATS.csv file to a STATS.h5 file

    Args:
        stats_file              : filename of stats file
    '''
    stats = pd.read_csv(stats_file, index_col=False)
    assert stats_file[-10:] == '-STATS.csv', f"Stats file {stats_file} should end in '-STATS.csv'."
    write_stats(stats_file[:-10], stats, append=False)


def trim_stats(stats_file, start_time, end_time, write_new=False, stats=[]):
    '''Chops a STATS.h5 file given a start and end time

    Args:
        stats_file              : filename of stats file
        start_time                  : start time of interesting window
        end_time                    : end time of interesting window
        write_new=False             : boolean if True will write a new stats csv file to disc
        stats=[]                    : pass stats DataFrame into here if you don't want to load the data from the stats_file given.
                                      In this case the stats_file string is only used for creating the new output datafilename.

    Returns:
        trimmed_stats       : pandas DataFram of particle statistics
        outname             : name of new stats csv file written to disc
    '''
    if len(stats) == 0:
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')

    start_time = pd.to_datetime(start_time)
    end_time = pd.to_datetime(end_time)

    trimmed_stats = stats[
        (pd.to_datetime(stats['timestamp']) > start_time) & (pd.to_datetime(stats['timestamp']) < end_time)]

    if np.isnan(trimmed_stats.equivalent_diameter.max()) or len(trimmed_stats) == 0:
        logger.info('No data in specified time range!')
        outname = ''
        return trimmed_stats, outname

    actual_start = pd.to_datetime(trimmed_stats['timestamp'].min()).strftime('D%Y%m%dT%H%M%S.%f')
    actual_end = pd.to_datetime(trimmed_stats['timestamp'].max()).strftime('D%Y%m%dT%H%M%S.%f')

    path, name = os.path.split(stats_file)

    outname = os.path.join(path, name.replace('-STATS.h5', '')) + '-Start' + str(actual_start) + '-End' + str(
        actual_end)

    if write_new:
        write_stats(outname, trimmed_stats, append=False)
        #trimmed_stats.to_csv(outname)

    return trimmed_stats, outname


def add_best_guesses_to_stats(stats):
    '''
    Calculates the most likely tensorflow classification and adds best guesses
    to stats dataframe.

    Args:
        stats (DataFrame)           : particle statistics from silcam process

    Returns:
        stats (DataFrame)           : particle statistics from silcam process
                                      with new columns for best guess and best guess value
    '''

    cols = stats.columns

    p = np.zeros_like(cols) != 0
    for i, c in enumerate(cols):
        p[i] = str(c).startswith('probability')

    pinds = np.squeeze(np.argwhere(p))

    parray = np.array(stats.iloc[:, pinds[:]])

    stats['best guess'] = cols[pinds.min() + np.argmax(parray, axis=1)]
    stats['best guess value'] = np.max(parray, axis=1)

    return stats


def show_h5_meta(h5file):
    '''
    prints metadata from an exported hdf5 file created from silcam process

    Args:
        h5file              : h5 filename from exported data from silcam process
    '''

    with h5py.File(h5file, 'r') as f:
        keys = list(f['Meta'].attrs.keys())

        for k in keys:
            logger.info(k + ':')
            print(k + ':')
            logger.info('    ' + f['Meta'].attrs[k])
            print('    ' + f['Meta'].attrs[k])


def vd_to_nd(vd, dias):
    '''convert volume distribution to number distribution

    Args:
        vd (array)           : particle volume distribution calculated from vd_from_stats()
        dias (array)         : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        nd (array)           : number distribution as number per micron per bin (scaling is the same unit as the input vd)
    '''
    DropletVolume = ((4 / 3) * np.pi * ((dias * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
    nd = vd / (DropletVolume * 1e9)  # the number distribution in each bin
    return nd


def vd_to_nc(vd, dias):
    '''calculate number concentration from volume distribution

    Args:
        vd (array)           : particle volume distribution calculated from vd_from_stats()
        dias (array)         : mid-points in the size classes corresponding the the volume distribution,
                               returned from get_size_bins()

    Returns:
        nn (float)           : number concentration (scaling is the same unit as the input vd).
                               If vd is a 2d array [time, vd_bins], nc will be the concentration for row
    '''
    nd = vd_to_nd(dias, vd)
    if np.ndim(nd) > 1:
        nc = np.sum(nd, axis=1)
    else:
        nc = np.sum(nd)
    return nc
