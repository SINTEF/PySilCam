# -*- coding: utf-8 -*-
'''
Particle plotting functionality: PSD, D50, etc.
'''
import matplotlib.pyplot as plt
import pysilcam.postprocess as sc_pp
import numpy as np
import seaborn as sns
from pysilcam.config import load_config, PySilcamSettings
import pandas as pd

class ParticleSizeDistPlot:
    '''Plot particle size distribution information on 2x2 layout'''

    def __init__(self):
        sns.set_style('white')
        sns.set_context('notebook', font_scale=0.8)

        plt.ion()
        self.figure, self.ax = plt.subplots(2, 2)

    def plot(self, imc, imbw, times, d50_ts, vd_mean, display):
        '''Create plots from data'''

        #Plot image in upper left axis
        ax = self.ax[0, 0]
        if display==True:
            self.image = ax.imshow(np.uint8(imc), cmap='gray',
                                   interpolation='None', animated=True, vmin=0,
                                   vmax=255)

        #Plot segmented image in upper right axis
        ax = self.ax[0, 1]
        if display==True:
            self.image_bw = ax.imshow(np.uint8(imbw > 0), cmap='gray',
                                      interpolation='None', animated=True)

        #Plot D50 time series in lower left axis
        ax = self.ax[1, 0]
        self.d50_plot, = ax.plot(range(len(d50_ts)), d50_ts, '.')
        ax.set_xlabel('image #')
        ax.set_ylabel('d50 (um)')
        ax.set_xlim(0, 50)
        ax.set_ylim(10, 10000)
        ax.set_yscale('log')

        #Plot PSD in lower right axis
        norm = np.sum(vd_mean['total'].vd_mean)/100
        ax = self.ax[1, 1]
        self.line, = ax.plot(vd_mean['total'].dias, vd_mean['total'].vd_mean, color='k')
#        self.line_oil, = ax.plot(vd_mean['oil'].dias,
#                                  vd_mean['oil'].vd_mean, color='darkred')
#        self.line_gas, = ax.plot(vd_mean['gas'].dias,
#                                  vd_mean['gas'].vd_mean, color='royalblue')
        ax.set_xlim(1, 10000)
        ax.set_ylim(0, 20)
        ax.set_xscale('log')
        ax.set_xlabel('Equiv. diam (um)')
        ax.set_ylabel('Volume concentration (%/sizebin)')

        #Trigger initial full draw of the figure
        self.figure.canvas.draw()


    def update(self, imc, imbw, times, d50_ts, vd_mean, display):
        '''Update plot data without full replotting for speed'''

        if display==True:
            self.image.set_data(np.uint8(imc))
            self.image_bw.set_data(np.uint8(imbw>0))

        #Show the last 50 D50 values
        self.d50_plot.set_data(range(len(d50_ts[-50:])), d50_ts[-50:])

        norm = np.sum(vd_mean['total'].vd_mean)/100
        self.line.set_data(vd_mean['total'].dias, vd_mean['total'].vd_mean/norm)
#        self.line_oil.set_data(vd_mean['oil'].dias, vd_mean['oil'].vd_mean/norm)
#        self.line_gas.set_data(vd_mean['gas'].dias, vd_mean['gas'].vd_mean/norm)

        #Fast redraw of dynamic figure elements only
        if display==True:
            self.ax[0, 0].draw_artist(self.image)
            self.ax[0, 1].draw_artist(self.image_bw)
        self.ax[1, 0].draw_artist(self.d50_plot)
        self.ax[1, 1].draw_artist(self.line)
        #self.ax[1, 1].draw_artist(self.line_oil)
        #self.ax[1, 1].draw_artist(self.line_gas)
        self.figure.canvas.flush_events()


def psd(stats, settings, ax, line=None, c='k'):

    dias, vd = sc_pp.vd_from_stats(stats, settings)

    if line:
        line.set_data(dias, vd/np.sum(vd)*100)
    else:
        line, = ax.plot(dias,vd/np.sum(vd)*100, color=c)
        ax.set_xscale('log')
        ax.set_xlabel('Equiv. diam (um)')
        ax.set_ylabel('Volume concentration (%/sizebin)')
    ax.set_xlim(10, 10000)
    ax.set_ylim(0, max(vd/np.sum(vd)*100))

    #ax.axvline(sc_pp.d50_from_vd(vd,dias), color=c)

    return line


def nd_scaled(stats, settings, ax, c='k'):
    sv = sc_pp.get_sample_volume(settings.pix_size,
            path_length=settings.path_length,
            imx=2048, imy=2448) # sample volume per image
    # re-scale sample volume to number of images
    sv_total = sv * sc_pp.count_images_in_stats(stats) # total sampled water volume

    nd(stats, settings, ax, line=None, c='k', sample_volume=sv_total)
    return


def nd(stats, settings, ax, line=None, c='k', sample_volume=1.):

    # nc per size bin per sample volume
    dias, nd = sc_pp.nd_from_stats(stats, settings)

    nd = sc_pp.nd_rescale(dias, nd, sample_volume)

    # remove data from first bin which will be part-full
    ind = np.argwhere(nd>0)
    nd[ind[0]] = np.nan


    if line:
        line.set_data(dias, nd)
    else:
        line, = ax.plot(dias,nd, color=c)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Equiv. diam [um]')
        ax.set_ylabel('Number concentration [#/L/um]')
    ax.set_xlim(10, 10000)
#    ax.set_ylim(0, 100)

    #ax.axvline(sc_pp.d50_from_vd(vd,dias), color=c)

    return line


def show_imc(imc, mag=2):
    PIX_SIZE = 35.2 / 2448 * 1000
    r, c = np.shape(imc[:,:,0])

    if mag==1:
        PIX_SIZE = 67.4 / 2448 * 1000

    plt.imshow(np.uint8(imc),
            extent=[0,c*PIX_SIZE/1000,0,r*PIX_SIZE/1000],
            interpolation='nearest')
    plt.xlabel('mm')
    plt.ylabel('mm')

    return


def montage_plot(montage, pixel_size):

    msize = np.shape(montage[:,0,0])
    ex = pixel_size * np.float64(msize)/1000.

    ax = plt.gca()
    ax.imshow(montage, extent=[0,ex,0,ex])
    ax.set_xticks([1, 2],[])
    ax.set_xticklabels(['    1mm',''])
    ax.set_yticks([], [])
    ax.xaxis.set_ticks_position('bottom')


def summarise_fancy_stats(stats_csv_file,config_file,monitor=False):
    sns.set_style('ticks')

    conf = load_config(config_file)
    settings = PySilcamSettings(conf)

    min_length = settings.ExportParticles.min_length + 1

    #f,a = plt.subplots(2,2)
    ax1 = plt.subplot2grid((2,2),(0, 0))
    ax2 = plt.subplot2grid((2,2),(1, 0))
    ax3 = plt.subplot2grid((2,2), (0, 1), rowspan=2)

    while True:
        montage = sc_pp.make_montage(stats_csv_file,
                settings.PostProcess.pix_size, roidir='export',
                min_length=min_length, max_length=10000,
                auto_scaler=1600, msize=2048)

        stats = pd.read_csv(stats_csv_file)

        # average numer and volume concentrations
        nc, vc, sv_total, junge = sc_pp.nc_vc_from_stats(stats, settings.PostProcess)
        d50 = sc_pp.d50_from_stats(stats, settings.PostProcess)
        total_measured_particles = len(stats['major_axis_length'])

        plt.sca(ax1)
        plt.cla()
        psd(stats, settings.PostProcess, plt.gca())
        plt.title('Volume conc.: {0:.2f}uL/L  d50: {1:.0f}um'.format(vc, d50))

        plt.sca(ax2)
        plt.cla()
        nd(stats, settings.PostProcess, plt.gca(), sample_volume=sv_total)
        plt.title('Number conc.: {0:.0f}#/L  Junge exp.: {1:.2f}'.format(nc,
            junge))

        plt.sca(ax3)
        plt.cla()
        montage_plot(montage, settings.PostProcess.pix_size)
        plt.title('Volume processed: {0:.1f}L  {1:.0f} particles measured'.format(sv_total,
            total_measured_particles))

        plt.draw()
        if monitor:
            plt.pause(1)
        else:
            break
    #plt.show()
