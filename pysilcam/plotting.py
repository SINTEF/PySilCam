import matplotlib.pyplot as plt
import pysilcam.postprocess as sc_pp
import numpy as np

def psd(stats, settings, c='k'):
    
    dias, vd = sc_pp.vd_from_stats(stats, settings)

    plt.plot(dias,vd/np.sum(vd)*100, color=c)
    plt.xscale('log')

    plt.axvline(sc_pp.d50_from_vd(vd,dias), color=c)
    plt.xlabel('Equiv. diam (um)')
    plt.ylabel('Volume concentration (%/sizebin)')

    return

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

