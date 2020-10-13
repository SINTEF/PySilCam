import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import os
from skimage.draw import circle
from skimage import util
import pysilcam.postprocess as scpp
import pysilcam.process as scpr
import pysilcam.config as sccf
import pysilcam.silcam_classify as sccl


def generate_report(report_name, PIX_SIZE=28.758169934640524,
                    PATH_LENGTH=40, d50=400, TotalVolumeConcentration=800,
                    MinD=108, config_file=''):
    '''Create a report of the expected response of the silcam to the provided experimental setup

    Args:
      report_name   (str)                   :  The path and filename of a pdf to be created
      PIX_SIZE          (float)             :  pixel size of the setup [um]
      PATH_LENGTH (float)                   :  the path length of the setup [mm] Path length is the gap between housings
      d50   (float)                         :  the expected of the oil d50 (50th percentile of the cumulative sum of the
                                               volume distribution)
      TotalVolumeConcentration (float)      :  the expected concentration of oil in the sample volume [uL/L]
      MinD (float)                          :  minimum resolvable diameter of the setup [um]. this would usually scale
                                               with the pixel size. Synthesized particles smaller than this are also
                                               removed for speed purposes.
    '''
    plt.close('all')
    pp = PdfPages(report_name)

    # image dimensions (fixed always for GC2450 camera)
    imx = 2448
    imy = 2048

    # get diameters and limits of size bins
    diams, bin_limits_um = scpp.get_size_bins()

    # initial volume distribution, close to Oystein's MPB paper in 2013
    vd = weibull(diams, n=d50)
    vd = vd / np.sum(vd) * TotalVolumeConcentration  # scale the distribution according to concentration

    DropletVolume = ((4 / 3) * np.pi * ((diams * 1e-6) / 2) ** 3)  # the volume of each droplet in m3
    nd = vd / (DropletVolume * 1e9)  # the number distribution in each bin
    nd[diams < MinD] = 0  # remove small particles for speed purposes

    # calculate the sample volume of the SilCam specified
    sv = scpp.get_sample_volume(PIX_SIZE, path_length=PATH_LENGTH, imx=imx, imy=imy)

    nd = nd * sv  # scale the number distribution by the sample volume so resulting units are #/L/bin
    nc = int(sum(nd))  # calculate the total number concentration

    vd2 = scpp.vd_from_nd(nd, diams, sv)  # convert the number distribution to volume distribution in uL/L/bin
    vc_initial = sum(vd2)  # obtain the resulting concentration, now having remove small particles

    d50_theory = scpp.d50_from_vd(vd2, diams)  # calculate the d50 in um

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

    nims = 40  # the number of images to simulate
    # preallocate variables
    log_vd = np.zeros((nims, len(diams)))
    cvd = np.zeros(nims)
    cd50 = np.zeros(nims)

    for I in range(nims):
        # randomly select a droplet radius from the input distribution
        rad = np.random.choice(diams / 2, size=nc, p=nd / sum(nd)) / PIX_SIZE  # radius is in pixels
        log_ecd = rad * 2 * PIX_SIZE  # log this size as a diameter in um

        necd, edges = np.histogram(log_ecd, bin_limits_um)  # count particles into number distribution
        log_vd[I, :] = scpp.vd_from_nd(necd, diams)  # convert to volume distribution
        cvd[I] = np.sum(
            np.mean(log_vd[0:I, :], axis=0))  # calculated the cumulate volume distribution over image number
        cd50[I] = scpp.d50_from_vd(np.mean(log_vd, axis=0), diams)  # calcualte the cumulate d50 over image number

    f, a = plt.subplots(1, 3, figsize=(16, 4))

    plt.sca(a[0])
    plt.plot(diams, vd2, 'k')
    plt.plot(diams, log_vd[0, :] / sv, alpha=0.5, label='1 image')
    plt.plot(diams, np.mean(log_vd[0:4, :], axis=0) / sv, alpha=0.5, label='4 images')
    plt.plot(diams, np.mean(log_vd, axis=0) / sv, alpha=0.5, label=(str(nims) + ' images'))
    plt.xscale('log')
    plt.vlines(d50_theory, 0, max(vd2), linestyle='--')
    plt.xlabel('ECD [um]')
    plt.ylabel('Volume distribution [uL/L/bin]')
    plt.legend()
    plt.title('Statistical summaries')

    plt.sca(a[1])
    plt.plot(cvd / sv, 'k')
    plt.hlines(vc_initial, 0, nims, linestyle='--')
    plt.xlabel('Image number')
    plt.ylabel('Volume concentration [uL/L]')

    plt.sca(a[2])
    plt.plot(cd50, 'k')
    plt.hlines(d50_theory, 0, nims, linestyle='--')
    plt.xlabel('Image number')
    plt.ylabel('d50 [um]')

    pp.savefig(bbox_inches='tight')

    # synthesize an image, returning the segmented image and the inputted volume distribution
    img, log_vd = synthesize(diams, bin_limits_um, nd, imx, imy, PIX_SIZE)

    plt.figure(figsize=(10, 10))
    plt.imshow(img, vmin=0, vmax=255, extent=[0, imx * PIX_SIZE / 1000, 0, imy * PIX_SIZE / 1000])
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.title('Synthetic image')
    pp.savefig(bbox_inches='tight')

    vd = np.zeros_like(log_vd)
    imbw = np.zeros_like(img[:, :, 0])
    stat_extract_time = pd.Timedelta(seconds=0)
    # @todo this should be handles properly as part of testing
    try:
        diams, vd, imbw, stat_extract_time = test_analysis(img, PIX_SIZE, PATH_LENGTH, config_file=config_file)
    except:
        print('Analysis failed')
        pass

    f, a = plt.subplots(1, 2, figsize=(20, 8))

    plt.sca(a[0])
    plt.plot(diams, vd2, 'r:', label='Initial')
    plt.plot(diams, log_vd / sv, 'k', label='Statistical Best')
    plt.plot(diams, vd, 'g', alpha=0.5, label='PySilCam')
    plt.xscale('log')
    plt.xlabel('ECD [um]')
    plt.ylabel('Volume distribution [uL/L]')
    plt.legend()
    plt.title('Single image assessment:' +
              '\n\n' + 'Statextract took ' + str(stat_extract_time.seconds) + ' seconds',
              horizontalalignment='left', loc='left')

    plt.sca(a[1])
    plt.imshow(imbw, vmin=0, vmax=1, extent=[0, imx * PIX_SIZE / 1000, 0, imy * PIX_SIZE / 1000], cmap='gray')
    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.title('imbw')

    pp.savefig(bbox_inches='tight')

    pp.close()


def weibull(x, n=250., a=2.8):
    '''weibull distribution similar to Oystein's MPB 2013 paper'''
    n *= 1.566
    return (a / n) * (x / n) ** (a - 1) * np.exp(-(x / n) ** a)


def synthesize(diams, bin_limits_um, nd, imx, imy, PIX_SIZE):
    '''synthesize an image and measure droplets

    Args:
      diams   (array)                       :  size bins of the number distribution
      bin_limits_um (array)                 :  limits of the size bins where dias are the mid-points
      nd      (array)                       :  number of particles per size bin
      imx     (float)                       :  image width in pixels
      imy     (float)                       :  image height in pixels
      PIX_SIZE          (float)             :  pixel size of the setup [um]

    Returns:
      img (unit8)                         : segmented image from pysilcam
      log_vd (array)                      : a volume distribution of the randomly selected particles put into the
                                            synthetic image

    '''
    nc = int(sum(nd))  # number concentration

    # preallocate the image and logged volume distribution variables
    img = np.zeros((imy, imx, 3), dtype=np.uint8()) + 230  # scale the initial brightness down a bit
    log_ecd = np.zeros(nc)
    # randomly select a droplet radii from the input distribution
    rad = np.random.choice(diams / 2, size=nc, p=nd / sum(nd)) / PIX_SIZE  # radius is in pixels
    log_ecd = rad * 2 * PIX_SIZE  # log these sizes as a diameter in um
    for rad_ in rad:
        # randomly decide where to put particles within the image
        col = np.random.randint(1, high=imx - rad_)
        row = np.random.randint(1, high=imy - rad_)
        rr, cc = circle(row, col, rad_)  # make a cirle of the radius selected from the distribution
        img[rr, cc, :] = 0  # make the circle completely non-transmitting (i.e. black)

    necd, edges = np.histogram(log_ecd, bin_limits_um)  # count the input diameters into a number distribution
    log_vd = scpp.vd_from_nd(necd, diams)  # convert to a volume distribution

    # add some noise to the synthesized image
    img = np.uint8(255 * util.random_noise(np.float64(img) / 255), var=0.01 ** 2)

    img = np.uint8(img)  # convert to uint8
    return img, log_vd


def test_analysis(img, PIX_SIZE, PATH_LENGTH, config_file=''):
    '''wrapper for pysilcam processing

    Args:
      img (unit8)           : image to be processed (equivalent to the background-corrected image obtained from the
                              SilCam)
      PIX_SIZE (float)      : pixel size of the setup [um]
      PATH_LENGTH (float)   : path length of the setup [mm]

    Returns:
      dias (array)                      : mid points of the size distribution bins [um]
      vd (array)                        : volume concentration in each size bin [uL/L/bin]
      imbw (uint8)                      : segmented image
      stat_extract_time (timestamp)     : time taken to run statextract

    '''

    # administer configuration settings according to specified setup
    if config_file == '':
        testconfig = os.path.split(sccf.default_config_path())[0]
        testconfig = os.path.join(testconfig, 'tests/config.ini')
        import configparser
        conf = configparser.ConfigParser()
        conf.set('Process', 'real_time_stats', 'True')
        conf.set('Process', 'threshold', '0.85')
        conf.set('ExportParticles', 'export_images', 'False')
    else:
        testconfig = config_file
    conf = sccf.load_config(testconfig)
    conf.set('PostProcess', 'pix_size', str(PIX_SIZE))
    conf.set('PostProcess', 'path_length', str(PATH_LENGTH))

    settings = sccf.PySilcamSettings(conf)  # pass these settings without saving a config file to disc

    # load tensorflow model
    nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

    start_time = pd.Timestamp.now()  # time statextract
    # process the image
    stats, imbw, saturation = scpr.statextract(img, settings, pd.Timestamp.now(),
                                               nnmodel, class_labels)
    end_time = pd.Timestamp.now()  # time statextract
    stat_extract_time = end_time - start_time

    # calculate the volume distribution from the processed stats
    dias, vd = scpp.vd_from_stats(stats, settings.PostProcess)

    # scale the volume distribution according to the SilCam setup specified
    sv = scpp.get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)
    vd /= sv

    return dias, vd, imbw, stat_extract_time
