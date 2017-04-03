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
import pysilcam.datalogger as datalogger
import pysilcam.oilgas as oilgas
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

    #Load the configuration, create settings object
    conf = load_config(config_filename)
    settings = PySilcamSettings(conf)

    #Print configuration to screen
    print('---- CONFIGURATION ----\n')
    conf.write(sys.stdout)
    print('-----------------------\n')

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

    #Volume size distribution for total, oil and gas
    vd_mean = dict(total=sc_pp.TimeIntegratedVolumeDist(settings.PostProcess),
                   oil=sc_pp.TimeIntegratedVolumeDist(settings.PostProcess),
                   gas=sc_pp.TimeIntegratedVolumeDist(settings.PostProcess))
    d50_ts = dict(total=[], oil=[], gas=[])

    if settings.Process.display:
        logger.info('Initializing real-time plotting')
        rtplot = scplt.ParticleSizeDistPlot()

    ogdatafile = datalogger.DataLogger(settings.General.datafile + '.csv',
            oilgas.ogdataheader())
    ogdatafile_gas = datalogger.DataLogger(settings.General.datafile + '-GAS.csv',
            oilgas.ogdataheader())

    def loop(i, timestamp, imc):
        logger.info('Processing time stamp {0}'.format(timestamp))

        #Time the full acquisition and processing loop
        start_time = time.clock()

        nc = color.guess_spatial_dimensions(imc)
        if nc == None: # @todo FIX if there are ambiguous dimentions, assume RGB color space
            imc = imc[:,:,1] # and just use green

        #Calculate particle statistics
        stats_all, imbw = statextract(imc, settings)
        stats = dict(total=stats_all,
                     oil=stats_all[stats_all['gas']==0],
                     gas=stats_all[stats_all['gas']==1])

        #Time the particle statistics processing step
        proc_time = time.clock() - start_time

        #Calculate time-averaged volume distributions and D50 from particle stats
        for key in stats.keys():
            vd_mean[key].update_from_stats(stats[key], timestamp)
            d50_ts[key].append(sc_pp.d50_from_vd(vd_mean[key].vd_mean, 
                                                 vd_mean[key].dias))
        times.append(i)

        #If real-time plotting is enabled, update the plots
        if settings.Process.display:
            if i == 0:
                rtplot.plot(imc, imbw, times, d50_ts['total'], vd_mean)
            else:
                rtplot.update(imc, imbw, times, d50_ts['total'], vd_mean)

        tot_time = time.clock() - start_time

        #Log particle stats data to file
        data_all = oilgas.cat_data(timestamp, stats['total'], settings)
        ogdatafile.append_data(data_all)
        data_gas = oilgas.cat_data(timestamp, stats['gas'], settings)
        ogdatafile_gas.append_data(data_gas)

        #Print timing information for this iteration
        plot_time = tot_time - proc_time
        infostr = '  Image {0} processed in {1:.2f} sec. '
        infostr += 'Statextract: {2:.2f}s ({3:.0f}%) Plot: {4:.2f}s ({5:.0f}%)'
        print(infostr.format(i, tot_time, proc_time, proc_time/tot_time*100, 
                             plot_time, plot_time/tot_time*100))


    print('* Commencing image acquisition and processing')
    for i, (timestamp, imc) in enumerate(bggen):
        try:
            loop(i, timestamp, imc)
        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr)
            print(infostr)
    

def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
