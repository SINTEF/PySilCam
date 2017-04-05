# -*- coding: utf-8 -*-
'''
Particle plotting functionality: PSD, D50, etc.
'''
import matplotlib.pyplot as plt
import pysilcam.postprocess as sc_pp
import numpy as np
import seaborn as sns

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
                                   interpolation='None', animated=True)

        #Plot segmented image in upper right axis
        ax = self.ax[0, 1]
        if display==True:
            self.image_bw = ax.imshow(np.uint8(imbw > 0), cmap='gray', 
                                      interpolation='None', animated=True)

        #Plot D50 time series in lower left axis
        ax = self.ax[1, 0]
        self.d50_plot, = ax.plot(times, d50_ts, '.')
        ax.set_xlabel('image #')
        ax.set_ylabel('d50 (um)')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 1000)

        #Plot PSD in lower right axis
        norm = np.sum(vd_mean['total'].vd_mean)/100
        ax = self.ax[1, 1]
        self.line, = ax.plot(vd_mean['total'].dias, vd_mean['total'].vd_mean, color='k')
        self.line_oil, = ax.plot(vd_mean['oil'].dias, 
                                  vd_mean['oil'].vd_mean, color='darkred')
        self.line_gas, = ax.plot(vd_mean['gas'].dias,
                                  vd_mean['gas'].vd_mean, color='royalblue')
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
        self.d50_plot.set_data(times[-50:], d50_ts[-50:])

        norm = np.sum(vd_mean['total'].vd_mean)/100
        self.line.set_data(vd_mean['total'].dias, vd_mean['total'].vd_mean/norm)
        self.line_oil.set_data(vd_mean['oil'].dias, vd_mean['oil'].vd_mean/norm)
        self.line_gas.set_data(vd_mean['gas'].dias, vd_mean['gas'].vd_mean/norm)

        #Fast redraw of dynamic figure elements only
        if display==True:
            self.ax[0, 0].draw_artist(self.image)
            self.ax[0, 1].draw_artist(self.image_bw)
        self.ax[1, 0].draw_artist(self.d50_plot)
        self.ax[1, 1].draw_artist(self.line)
        self.ax[1, 1].draw_artist(self.line_oil)
        self.ax[1, 1].draw_artist(self.line_gas)
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
    ax.set_xlim(1, 10000)
    ax.set_ylim(0, 100)

    #ax.axvline(sc_pp.d50_from_vd(vd,dias), color=c)

    return line

def show_imc(imc, mag=2):
    PIX_SIZE = 35.2 / 2448 * 1000

    if mag==1:
        PIX_SIZE = 67.4 / 2448 * 1000
    
    plt.imshow(np.uint8(imc),
            extent=[0,2448*PIX_SIZE/1000,0,2048*PIX_SIZE/1000],
            interpolation='nearest')
    plt.xlabel('mm')
    plt.ylabel('mm')

    return

