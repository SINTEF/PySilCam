import matplotlib.pyplot as plt
import pysilcam.postprocess as sc_pp
import numpy as np

class ParticleSizeDistPlot:
    '''Plot particle size distribution information on 2x2 layout'''

    def __init__(self):
        self.figure, self.ax = plt.subplots(2, 2)
        plt.ion()

    def plot(self, imc, times, times, d50_ts, vd_mean, vd_mean_oil, vd_mean_gas):
        '''Create plots from data'''
        ax = self.ax[0, 0]
        self.image = ax.imshow(np.uint8(imc), cmap='gray', 
                               interpolation='nearest', animated=True)
        self.image_bw = ax.imshow(np.uint8(imbw > 0), cmap='gray', 
                                  interpolation='nearest', animated=True)

        ax = self.ax[1, 0]
        self.d50_plot, = ax.plot(times, d50_ts, '.')
        ax.set_xlabel('image #')
        ax.set_ylabel('d50 (um)')
        ax.set_xlim(0, times[-1])
        ax.set_ylim(0, max(100, np.max(d50_ts)))

        ax = self.ax[1, 1]
        norm = np.sum(vd_mean.vd_mean)/100
        self.line, = ax.plot(vd_mean.dias, vd_mean.vd_mean, color='k')
        self.line_oil, = ax.plot(vd_mean_oil.dias, 
                                  vd_mean_oil.vd_mean, color='darkred')
        self.line_gas, = ax.plot(vd_mean_gas.dias,
                                  vd_mean_gas.vd_mean, color='royalblue')
        ax[1,1].set_xscale('log')
        ax[1,1].set_xlabel('Equiv. diam (um)')
        ax[1,1].set_ylabel('Volume concentration (%/sizebin)')
 
 
    def update_plot(self, imc, times, d50_ts, vd_mean, vd_mean_oil, vd_mean_gas):
        '''Update plot data without full replotting for speed'''
        self.image.set_data(np.uint8(imc))
        self.image_bw.set_data(np.uint8(imc > 0))
        self.d50_plot.set_data(times, d50_ts)

        self.line.set_data(vd_mean.dias, vd_mean.vd_mean/norm)
        self.line_oil.set_data(vd_mean_oil.dias, vd_mean_oil.vd_mean/norm)
        self.line_gas.set_data(vd_mean_gas.dias, vd_mean_gas.vd_mean/norm)
        self.ax[1,1].set_xlim(1, 10000)
        self.ax[1,1].set_ylim(0, np.max(vd_mean.vd_mean/norm))

        plt.pause(0.01)
        fig.canvas.draw()


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

