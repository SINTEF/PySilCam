# -*- coding: utf-8 -*-
import sys
import time
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile
import pstats
from io import StringIO
import logging
from pysilcam import __version__
from pysilcam.acquisition import acquire
from pysilcam.background import backgrounder
import pysilcam.process
from pysilcam.process import statextract
import pysilcam.postprocess as sc_pp
import pysilcam.plotting as scplt
from pysilcam.config import load_config, PySilcamSettings
from skimage import color

title = '''
 ____        ____  _ _  ____                
|  _ \ _   _/ ___|(_) |/ ___|__ _ _ __ ___  
| |_) | | | \___ \| | | |   / _` | '_ ` _ \ 
|  __/| |_| |___) | | | |__| (_| | | | | | |
|_|    \__, |____/|_|_|\____\__,_|_| |_| |_|
       |___/                                
'''

def configure_logger(settings):
    if settings.logfile:
        logging.basicConfig(filename=settings.logfile,
                            level=getattr(logging, settings.loglevel))
    else:
        logging.basicConfig(level=getattr(logging, settings.loglevel))


def silcam_acquire():
    '''Aquire images from the SilCam

    Usage:
      silcam-acquire
      silcam-acquire liveview
      silcam-acquire process <configfile>
      silcam-acquire -h | --help
      silcam-acquire --version

    Arguments:
        liveview    Display acquired images
        process     Process acquired images in real time

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    print(title)
    print('')
    args = docopt(silcam_acquire.__doc__, version='PySilCam {0}'.format(__version__))
    #print('Type \'silcam-acquire -h\' for help')

    if args['process']:
        pr = cProfile.Profile()
        pr.enable()
        s = StringIO()
        sortby = 'cumulative'
        silcam_process_realtime(args['<configfile>'])
        pr.disable()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
#        print(s.getvalue())
        ps.dump_stats('process_profile.cprof')

    elif args['liveview']:
        print('LIVEVIEW MODE')
        print('')
        print('----------------------\n')
        plt.ion()
        fig, ax = plt.subplots()
        t1 = time.time()

        for i, img in enumerate(acquire()):
            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2
            #ax.imshow(img[:,:,0], cmap=plt.cm.gray)
            ax.imshow(np.uint8(img))
            #plt.draw()
            plt.pause(0.05)

    else:
        t1 = time.time()

        count = 0
        for i, img in enumerate(acquire()):
            count += 1

            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2


def silcam_process_realtime(config_filename):
    '''Run processing of SilCam images in real time'''
    print(config_filename)

    print('REALTIME MODE')
    print('')
    print('----------------------\n')
    #Load the configuration, create settings object
    conf = load_config(config_filename)
    conf.write(sys.stdout)
    print('----------------------\n')
    settings = PySilcamSettings(conf)

    #Configure logging
    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_process_realtime')

    #Initialize the image acquisition generator
    aqgen = acquire()

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen)

    times = []
    d50_ts = []

    vd_mean = sc_pp.TimeIntegratedVolumeDist(settings.PostProcess)
    vd_mean_oil = sc_pp.TimeIntegratedVolumeDist(settings.PostProcess)
    vd_mean_gas = sc_pp.TimeIntegratedVolumeDist(settings.PostProcess)

    if settings.Process.display:
        rtplot = plotting.ParticleSizeDistPlot()

    print('* Commencing image acquisition and processing')
    for i, (timestamp, imc) in enumerate(bggen):
        logger.info('Processing time stamp {0}'.format(timestamp))
        #logger.debug('PROCESSING....')
        start_time = time.clock()

        nc = color.guess_spatial_dimensions(imc)
        if nc == None: # @todo FIX if there are ambiguous dimentions, assume RGB color space
            imc = imc[:,:,1] # and just use green

        stats, imbw = statextract(imc, settings)

        oil = stats[stats['gas']==0]
        gas = stats[stats['gas']==1]

        proc_time = time.clock() - start_time

        if settings.Process.display:
            plt.axes(ax[0,0])
            if i == 0:
                image = plt.imshow(np.uint8(imc), cmap='gray', interpolation='nearest', animated=True)
            image.set_data(np.uint8(imc))

            plt.axes(ax[0,1])
            if i==0:
                image_bw = plt.imshow(np.uint8(imbw > 0), cmap='gray', interpolation='nearest', animated=True)
            image_bw.set_data(np.uint8(imbw > 0))

        if stats is not np.nan:
            logger.debug('data has arrived!')
        else:
            continue

        #Calculate time-averaged volume distributions
        vd_mean.update_from_stats(stats, timestamp)
        vd_mean_oil.update_from_stats(oil, timestamp)
        vd_mean_gas.update_from_stats(gas, timestamp)

        #Calculate D50s from particle statistics
        d50 = sc_pp.d50_from_vd(vd_mean.vd_mean, vd_mean.dias)
        d50_oil = sc_pp.d50_from_vd(vd_mean_oil.vd_mean, vd_mean_oil.dias)
        d50_gas = sc_pp.d50_from_vd(vd_mean_gas.vd_mean, vd_mean_gas.dias)
        logger.info('d50: {0:.1f}, d50 oil: {1:.1f}, d50 gas: {2:1f}'.format(d50, d50_oil, d50_gas))
        d50_ts.append(d50)
        times.append(i)

        if settings.Process.display:
            if i == 0:
                d50_plot, = ax[1, 0].plot(times, d50_ts, '.')
            else:
                d50_plot.set_data(times, d50_ts)
            ax[1, 0].set_xlabel('image #')
            ax[1, 0].set_ylabel('d50 (um)')
            ax[1, 0].set_xlim(0, times[-1])
            ax[1, 0].set_ylim(0, max(100, np.max(d50_ts)))

            norm = np.sum(vd_mean.vd_mean)/100
            if i == 0:
                line, = ax[1, 1].plot(vd_mean.dias, vd_mean.vd_mean, color='k')
                line_oil, = ax[1, 1].plot(vd_mean_oil.dias, 
                                          vd_mean_oil.vd_mean, color='darkred')
                line_gas, = ax[1, 1].plot(vd_mean_gas.dias,
                                          vd_mean_gas.vd_mean, color='royalblue')
                ax[1,1].set_xscale('log')
                ax[1,1].set_xlabel('Equiv. diam (um)')
                ax[1,1].set_ylabel('Volume concentration (%/sizebin)')
            else:
                line.set_data(vd_mean.dias, vd_mean.vd_mean/norm)
                line_oil.set_data(vd_mean_oil.dias, vd_mean_oil.vd_mean/norm)
                line_gas.set_data(vd_mean_gas.dias, vd_mean_gas.vd_mean/norm)
            ax[1,1].set_xlim(1, 10000)
            ax[1,1].set_ylim(0, np.max(vd_mean.vd_mean/norm))

            plt.pause(0.01)
            fig.canvas.draw()

        tot_time = time.clock() - start_time

        #logger.info('PROCESSING DONE in {0} sec.'.format(proc_time))
        print('  Processing image {0} took {1} sec. out of {2} sec.'.format(i,
            proc_time, tot_time))

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
