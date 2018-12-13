import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from skimage.draw import circle
from skimage import util

import pysilcam.postprocess as scpp
import imageio as imo
import pysilcam.process as scpr
import pysilcam.config as sccf
import pysilcam.silcam_classify as sccl
import pysilcam.plotting as scplt

def generate_report(report_name, PIX_SIZE = 28.758169934640524,
                    PATH_LENGTH=40, d50 = 400, TotalVolumeConcentration = 800,
                    MinD = 108):
    plt.close('all')
    pp = PdfPages(report_name)
    # d50 = 100 # d50(ish) (RR size)
     # TotalVolumeConcentration uL/L
    # MinD = 108
    # PIX_SIZE = 28.758169934640524 # microns
    # MinD = 108

    imx = 2448
    imy = 2048

    diams, limits = scpp.get_size_bins()

    # initial volume distribution
    vd = weibull(diams, n=d50)
    vd = vd/np.sum(vd)*TotalVolumeConcentration

    DropletVolume=((4/3)*np.pi*((diams*1e-6)/2)**3)
    nd=vd/(DropletVolume*1e9)
    nd[diams<MinD] = 0

    sv = scpp.get_sample_volume(PIX_SIZE, path_length=PATH_LENGTH, imx=imx, imy=imy)

    nd = nd*sv
    nc = int(sum(nd))

    vd2 = scpp.vd_from_nd(nd,diams,sv)
    vc_initial = sum(vd2)

    d50_theory = scpp.d50_from_vd(vd2, diams)

    plt.plot(diams, vd2, 'k', label='Initial')
    plt.plot(diams, vd, 'r:', label='Theoretical')
    plt.vlines(d50_theory, 0, max(vd2), linestyle='--')
    plt.xscale('log')
    plt.xlabel('ECD [um]')
    plt.xlabel('Volume distribution [uL/L/bin]')
    plt.legend()
    plt.title('Initial conditions:' +
             '\n\n' + str(nc) + ' particles per image volume' +
             '\n' + str(int(vc_initial)) + ' initial volume concentration [uL/L]' +
              '\n' + str(int(d50_theory)) + ' initial d50 [um]',
             horizontalalignment='left', loc='left')
    pp.savefig(bbox_inches='tight')


    nims = 40
    dias, bin_limits_um = scpp.get_size_bins()
    log_vd = np.zeros((nims,len(dias)))
    cvd = np.zeros(nims)
    cd50 = np.zeros(nims)

    for I in range(nims):
        nc = int(sum(nd))
        log_ecd = np.zeros(nc)
        rad = np.random.choice(diams/2, size=nc, p=nd/sum(nd)) / PIX_SIZE
        log_ecd = rad*2*PIX_SIZE

        necd, edges = np.histogram(log_ecd,bin_limits_um)
        log_vd[I,:] = scpp.vd_from_nd(necd,dias)
        cvd[I] = np.sum(np.mean(log_vd[0:I,:],axis=0))
        cd50[I] = scpp.d50_from_vd(np.mean(log_vd,axis=0), dias)


    f, a = plt.subplots(1,3,figsize=(16,4))

    plt.sca(a[0])
    plt.plot(diams, vd2, 'k')
    plt.plot(dias, log_vd[0,:]/sv, alpha=0.5, label='1 image')
    plt.plot(dias, np.mean(log_vd[0:4,:], axis=0)/sv, alpha=0.5, label='4 images')
    plt.plot(dias, np.mean(log_vd, axis=0)/sv, alpha=0.5, label=(str(nims) + ' images'))
    plt.xscale('log')
    plt.vlines(d50_theory, 0, max(vd2), linestyle='--')
    plt.xlabel('ECD [um]')
    plt.ylabel('Volume distribution [uL/L/bin]')
    plt.legend()
    plt.title('Statistical summaries')

    plt.sca(a[1])
    plt.plot(cvd/sv,'k')
    plt.hlines(vc_initial, 0, nims, linestyle='--')
    plt.xlabel('Image number')
    plt.ylabel('Volume concentration [uL/L]')

    plt.sca(a[2])
    plt.plot(cd50,'k')
    plt.hlines(d50_theory, 0, nims, linestyle='--')
    plt.xlabel('Image number')
    plt.ylabel('d50 [um]')

    pp.savefig(bbox_inches='tight')

    img, log_vd = synthesize(diams, nd, imx, imy, PIX_SIZE)

    plt.figure(figsize=(10,10))
    plt.imshow(img, vmin=0, vmax=255, extent=[0,imx*PIX_SIZE/1000,0,imy*PIX_SIZE/1000])
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.title('Synthetic image')
    pp.savefig(bbox_inches='tight')

    dias, vd, imbw, stat_extract_time = test_analysis(img, PIX_SIZE, PATH_LENGTH)


    f, a = plt.subplots(1,2,figsize=(10,4))

    plt.sca(a[0])
    plt.plot(diams, vd2, 'r:', label='Initial')
    plt.plot(dias, log_vd/sv ,'k', label='Statistical Best')
    plt.plot(dias, vd, 'g', alpha=0.5, label='PySilCam')
    plt.xscale('log')
    plt.xlabel('ECD [um]')
    plt.ylabel('Volume distribution [uL/L]')
    plt.legend()
    plt.title('Single image assessment:' +
              '\n\n' + 'Statextract took ' + str(stat_extract_time.seconds) + ' seconds',
             horizontalalignment='left', loc='left')

    plt.sca(a[1])
    plt.imshow(imbw, vmin=0, vmax=1, extent=[0,imx*PIX_SIZE/1000,0,imy*PIX_SIZE/1000])
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.title('imbw')

    pp.savefig(bbox_inches='tight')

    pp.close()

def weibull(x,n=250,a=2.8):
    n *= 1.566
    return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)


def synthesize(diams, nd, imx, imy, PIX_SIZE):
    nc = int(sum(nd))
    img = np.zeros((imy, imx, 3), dtype=np.uint8()) + 230

    log_ecd = np.zeros(nc)

    for i in range(nc):
        rad = np.int(np.random.choice(diams / 2, p=nd / sum(nd)) / PIX_SIZE)
        log_ecd[i] = rad * 2 * PIX_SIZE
        col = np.random.randint(1, high=imx - rad)
        row = np.random.randint(1, high=imy - rad)
        rr, cc = circle(row, col, rad)
        img[rr, cc, :] = 0

    dias, bin_limits_um = scpp.get_size_bins()
    necd, edges = np.histogram(log_ecd, bin_limits_um)
    log_vd = scpp.vd_from_nd(necd, dias)
    img = np.uint8(255 * util.random_noise(np.float64(img) / 255), var=0.01 ** 2)

    img = np.uint8(img)
    return img, log_vd


def test_analysis(img, PIX_SIZE, PATH_LENGTH):
    testconfig = os.path.split(sccf.default_config_path())[0]
    testconfig = os.path.join(testconfig, 'tests/config.ini')
    conf = sccf.load_config(testconfig)
    conf.set('ExportParticles', 'export_images', 'False')
    conf.set('PostProcess', 'pix_size', str(PIX_SIZE))
    conf.set('PostProcess', 'path_length', str(PATH_LENGTH))
    conf.set('Process', 'real_time_stats', 'True')
    conf.set('Process', 'threshold', '0.85')

    settings = sccf.PySilcamSettings(conf)

    nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

    start_time = pd.Timestamp.now()
    stats, imbw, saturation = scpr.statextract(img, settings, pd.Timestamp.now(),
                                               nnmodel, class_labels)
    end_time = pd.Timestamp.now()
    stat_extract_time = end_time - start_time

    dias, vd = scpp.vd_from_stats(stats, settings.PostProcess)
    sv = scpp.get_sample_volume(settings.PostProcess.pix_size,
                                path_length=settings.PostProcess.path_length)
    vd /= sv

    return dias, vd, imbw, stat_extract_time