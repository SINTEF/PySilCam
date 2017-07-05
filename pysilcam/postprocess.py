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


# PIX_SIZE = 35.2 / 2448 * 1000 # pixel size in microns (Med. mag)

def stats_from_csv(filename):
    stats = pd.read_csv(filename,index_col=0)
    return stats

def filter_stats(stats, settings):
    iniparts = len(stats)
    mmr = stats['minor_axis_length'] / stats['major_axis_length']
    stats = stats[mmr > settings.min_deformation]

    stats = stats[stats['solidity'] > settings.min_solidity]

    endparts = len(stats)
    print(iniparts - endparts,' particles removed.')
    print(endparts,' particles measured.')
    return stats

def d50_from_stats(stats, settings):

    dias, vd = vd_from_stats(stats, settings)

    d50 = d50_from_vd(vd,dias)
    return d50

def d50_from_vd(vd,dias):
    ''' calculate d50 from a volume distribution
    d50 = d50_from_vd(vd,dias)
    '''
    csvd = np.cumsum(vd/np.sum(vd))
    d50 = np.interp(0.5,csvd,dias)
    return d50

def get_size_bins():
    '''retrieve size bins for PSD analysis
    bin_mids_um, bin_limits_um = get_size_bins()
    '''
    bin_limits_um = np.zeros((53),dtype=np.float64)
    bin_limits_um[0] = 2.72 * 0.91

    for I in np.arange(1,53,1):
        bin_limits_um[I] = bin_limits_um[I-1] * 1.180

    bin_mids_um = np.zeros((52),dtype=np.float64)
    bin_mids_um[0] = 2.72

    for I in np.arange(1,52,1):
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
    nc = np.sum(count) / sv
    return nc

def nc_vc_from_stats(stats, settings):
    dias, necd = nd_from_stats(stats, settings)

    path_length = settings.path_length
    pix_size = settings.pix_size

    sample_volume = get_sample_volume(pix_size, path_length=path_length, imx=2048, imy=2448)
    nims = count_images_in_stats(stats)
    sample_volume *= nims

    vd = vd_from_nd(necd, dias, sample_volume)
    vc = np.sum(vd)

    nc = nc_from_nd(necd, sample_volume)

    # convert nd to units of nc per micron per litre
    nd = nd_rescale(dias, necd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan

    junge = get_j(dias,nd)

    return nc, vc, sample_volume, junge


def nd_from_stats_scaled(stats, settings):
    dias, necd = nd_from_stats(stats,settings)

    sample_volume = get_sample_volume(settings.pix_size,
            path_length=settings.path_length)
    nims = count_images_in_stats(stats)
    sample_volume *= nims

    nd = nd_rescale(dias, necd, sample_volume)
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan
    return dias, nd


def nd_from_stats(stats, settings):
    ecd = stats['equivalent_diameter'] * settings.pix_size
    ecd = ecd[~np.isnan(ecd)]

    dias, bin_limits_um = get_size_bins()
    necd, edges = np.histogram(ecd,bin_limits_um)
    necd = np.float64(necd)

    return dias, necd


def vd_from_stats(stats, settings):
    dias, necd = nd_from_stats(stats, settings)

    vd = vd_from_nd(necd,dias)

    return dias, vd


class TimeIntegratedVolumeDist:
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
    montage = np.zeros((msize,msize,3),dtype=np.uint8())
    immap_test = np.zeros_like(montage[:,:,0])
    print('making a montage - this might take some time....')

    for files in roifiles:
        print(files)
        particle_image = export_name2im(files, roidir)
        #particle_image = imageio.imread(files)

        #particle_rect = np.ones_like(particle_image)
        [height, width] = np.shape(particle_image[:,:,0])
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

        if tightpack:
            imbw = scpr.im2bw_fancy(np.uint8(particle_image[:,:,0]), 0.95)
            imbw = ndi.binary_fill_holes(imbw)

            for J in range(5):
                imbw = skimage.morphology.binary_dilation(imbw)

        #masked = particle_image * imbw[:,:,None]
        #masked[masked==0] = 255
        # masked = particle_image

        counter = 0
        while (counter < 5):
            r = np.random.randint(1,msize-height)
            c = np.random.randint(1,msize-width)

            if tightpack:
                test = np.max(immap_test[r:r+height,c:c+width]+imbw)
            else:
                test = np.max(immap_test[r:r+height,c:c+width,None]+1)

            if (test>1):
                counter += 1
            else:
                break

        if (test>1):
            continue

        montage[r:r+height,c:c+width,:] = np.uint8(particle_image)

        if tightpack:
            immap_test[r:r+height,c:c+width] = imbw
        else:
            immap_test[r:r+height,c:c+width,None] = immap_test[r:r+height,c:c+width,None]+1

    montageplot = np.copy(montage)
    montageplot[montage>255] = 255
    montageplot[montage==0] = 255
    print('montage complete')

    return montageplot


def make_montage(stats_csv_file, pixel_size, roidir, min_length=100,
        auto_scaler=500, msize=1024, max_length=5000):
    stats = pd.read_csv(stats_csv_file)

    stats = stats[~np.isnan(stats['major_axis_length'])]

    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)

    roifiles = stats['export name'].loc[
            (stats['major_axis_length'] * pixel_size > min_length) &
            (stats['major_axis_length'] * pixel_size < max_length)
            ].values

    print('rofiles:',len(roifiles))
    IMSTEP = np.max([np.int(np.round(len(roifiles)/auto_scaler)),1])
    print('reducing particles by factor of {0}'.format(IMSTEP))
    roifiles = roifiles[np.arange(0,len(roifiles),IMSTEP)]
    print('rofiles:',len(roifiles))

    #for i, f in enumerate(roifiles):
    #    roifiles[i] = os.path.join(roidir, f)

    montage = montage_maker(roifiles, roidir, pixel_size, msize)

    return montage


def get_sample_volume(pix_size, path_length=10, imx=2048, imy=2448):
    sample_volume_litres = imx*pix_size/1000 * imy*pix_size/1000 * path_length*1e-6

    return sample_volume_litres


def get_j(dias, nd):
    ind = np.isfinite(dias) & np.isfinite(nd) & (dias<300) & (dias>150)
    p = np.polyfit(np.log(dias[ind]),np.log(nd[ind]),1)
    j = p[0]
    return j


def count_images_in_stats(stats):
    u = stats['timestamp'].unique()
    n_images = len(u)

    return n_images


def extract_nth_largest(stats,settings,n=0):
    stats.sort_values(by=['equivalent_diameter'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def extract_nth_longest(stats,settings,n=0):
    stats.sort_values(by=['major_axis_length'], ascending=False, inplace=True)
    stats = stats.iloc[n]
    return stats


def explode_contrast(im):
    im = np.float64(im)
    p1, p2 = np.percentile(im, (0, 80))
    im_mod = rescale_intensity(im, in_range=(p1, p2))
                
    im_mod -= np.min(im_mod)
    im_mod /= np.max(im_mod)
    im_mod *= 255
    im_mod = np.uint8(im_mod)
    return im_mod


def explode_contrast_old(im):
    im = np.float64(im)
    im -= im.min()
    im /= im.max()
    im *= 255
    im = np.uint8(im)
    return im


def bright_norm(im,brightness=255):
    peak = np.median(im.flatten())
    bm = brightness - peak

    im = np.float64(im) + bm
    im[im>255] = 255

    im =np.uint8(im)
    return im


def nd_rescale(dias, nd, sample_volume):
    nd = np.float64(nd) / sample_volume # nc per size bin per litre

    # convert nd to units of nc per micron per litre
    dd = np.gradient(dias)
    nd /= dd
    nd[nd<0] = np.nan # and nan impossible values!

    return nd

def add_depth_to_stats(stats, time, depth):
    sctime = pd.to_datetime(stats['timestamp'])
    stats['Depth'] = np.interp(np.float64(sctime), np.float64(time), depth)
    return stats


def export_name2im(exportname, path):
    ''' returns an image from the export name string in the -STATS.csv file

    get the exportname like this: exportname = stats['export name'].values[0]
    '''
    pn = exportname.split('-')[1]
    name = exportname.split('-')[0] + '.h5'
    fullname = os.path.join(path, name)

    fh = h5py.File(fullname ,'r')

    im = fh[pn]
    return im
