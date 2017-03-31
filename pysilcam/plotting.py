import matplotlib.pyplot as plt
import pysilcam.postprocess as sc_pp
import numpy as np

class ParticleSizeDistPlot:
    def __init__(self):
        pass
        self.figure, self.ax = plt.subplots(2, 2)

    def update_plot(self, ):
        pass
        if settings.Process.display:
            plt.axes(ax[0,0])
            if i == 0:
                image = plt.imshow(np.uint8(imc), cmap='gray', interpolation='nearest', animated=True)
            image.set_data(np.uint8(imc))

            plt.axes(ax[0,1])
            if i==0:
                image_bw = plt.imshow(np.uint8(imbw > 0), cmap='gray', interpolation='nearest', animated=True)
            image_bw.set_data(np.uint8(imbw > 0))

            if i == 0:
                d50_plot, = ax[1, 0].plot(times, d50_ts, '.')
            else:
                d50_plot.set_data(times, d50_ts)
            ax[1, 0].set_xlabel('image #')
            ax[1, 0].set_ylabel('d50 (um)')
            ax[1, 0].set_xlim(0, times[-1])
            ax[1, 0].set_ylim(0, max(100, np.max(d50_ts)))

            if i == 0:
                line_t = None
                line_oil = None
                line_gas = None
            line_t = scplt.psd(stats, settings.PostProcess, ax=ax[1, 1], line=line_t, c='k')
            line_oil = scplt.psd(oil, settings.PostProcess, ax=ax[1, 1], line=line_oil, c='r')
            line_gas = scplt.psd(gas, settings.PostProcess, ax=ax[1, 1], line=line_gas, c='b')

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

