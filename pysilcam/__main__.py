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
import pysilcam.exportparticles as exportparts
from pysilcam.config import load_config, PySilcamSettings
from skimage import color
import imageio

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
      silcam-acquire fancyprocess <configfile>
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
        #pr = cProfile.Profile()
        #pr.enable()
        #s = StringIO()
        #sortby = 'cumulative'
#        while True:
#            try:
        silcam_process_realtime(args['<configfile>'])
#            except:
#                print('probably image loading error....')
#                time.sleep(1)
        #pr.disable()
        #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        #ps.print_stats()
#       # print(s.getvalue())
        #ps.dump_stats('process_profile.cprof')
    elif args['fancyprocess']:
        silcam_process_fancy(args['<configfile>'])

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
        while True:
            t1 = time.time()
            try:
                aqgen = acquire()
                for i, (timestamp, imraw) in enumerate(aqgen):
                    filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f.bmp')
                    imageio.imwrite(filename, imraw)
                    print('Written', filename)

                    t2 = time.time()
                    aq_freq = np.round(1.0/(t2 - t1), 1)
                    requested_freq = 16.0
                    rest_time = (1 / requested_freq) - (1 / aq_freq)
                    rest_time = np.max([rest_time, 0.])
                    time.sleep(rest_time)
                    actual_aq_freq = 1/(1/aq_freq + rest_time)
                    print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, actual_aq_freq))
                    #t1 = t2
                    t1 = time.time()
            except KeyboardInterrupt:
                print('User interrupt with ctrl+c, terminating PySilCam.')
                sys.exit(0)
            except:
                #infostr = 'Failed to acquire frame, restarting.'
                #print(infostr)
                etype, emsg, etrace = sys.exc_info()
                print('Exception occurred: {0}. Restarting acquisition.'.format(emsg))
 
def silcam_process_fancy(config_filename):
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
    logger = logging.getLogger(__name__ + '.silcam_process_fancy')

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

    psddatafile = datalogger.DataLogger(settings.General.datafile + '.csv',
            oilgas.ogdataheader())

    def loop(i, timestamp, imc):
        #Time the full acquisition and processing loop
        start_time = time.clock()

        logger.info('Processing time stamp {0}'.format(timestamp))

        nc = color.guess_spatial_dimensions(imc)
        if nc == None: # @todo FIX if there are ambiguous dimentions, assume RGB color space
        #    imc = imc[:,:,1] # and just use green
            #Calculate particle statistics
            stats_all, imbw, saturation = statextract(imc[:,:,1], settings)
        else:
            stats_all, imbw, saturation = statextract(imc, settings)

        stats = dict(total=stats_all)
        #stats_all, imbw, saturation = statextract(imc, settings)
        #stats = dict(total=stats_all,
        #             oil=stats_all[stats_all['gas']==0],
        #             gas=stats_all[stats_all['gas']==1])

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
                rtplot.plot(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)
            else:
                rtplot.update(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)

        #Log particle stats data to file
        data_all = oilgas.cat_data(timestamp, stats['total'], settings)
        psddatafile.append_data(data_all)

        if settings.ExportParticles.export_images:
            exportparts.export_particles(imc,timestamp,stats_all,settings)

        tot_time = time.clock() - start_time

        #Print timing information for this iteration
        plot_time = tot_time - proc_time
        infostr = '  Image {0} processed in {1:.2f} sec ({6:.1f} Hz). '
        infostr += 'Statextract: {2:.2f}s ({3:.0f}%) Plot: {4:.2f}s ({5:.0f}%)'
        print(infostr.format(i, tot_time, proc_time, proc_time/tot_time*100, 
                             plot_time, plot_time/tot_time*100, 1.0/tot_time))


    print('* Commencing image acquisition and processing')
    for i, (timestamp, imc) in enumerate(bggen):
        loop(i, timestamp, imc)
        continue
        try:
            loop(i, timestamp, imc)
        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr)
            print(infostr)


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
    tavoilfile = datalogger.DataLogger(settings.General.datafile + '-tavoil.csv',
            'tavd50, sat')

    def loop(i, timestamp, imc):
        #Time the full acquisition and processing loop
        start_time = time.clock()

        logger.info('Processing time stamp {0}'.format(timestamp))

        nc = color.guess_spatial_dimensions(imc)
        if nc == None: # @todo FIX if there are ambiguous dimentions, assume RGB color space
            imc = imc[:,:,1] # and just use green

        #Calculate particle statistics
        stats_all, imbw, saturation = statextract(imc, settings)
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
                rtplot.plot(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)
            else:
                rtplot.update(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)

        #Log particle stats data to file
        data_all = oilgas.cat_data(timestamp, stats['total'], settings)
        ogdatafile.append_data(data_all)
        data_gas = oilgas.cat_data(timestamp, stats['gas'], settings)
        ogdatafile_gas.append_data(data_gas)

        tavoilfile.append_data([d50_ts['total'][-1], saturation])

        tot_time = time.clock() - start_time

        #Print timing information for this iteration
        plot_time = tot_time - proc_time
        infostr = '  Image {0} processed in {1:.2f} sec ({6:.1f} Hz). '
        infostr += 'Statextract: {2:.2f}s ({3:.0f}%) Plot: {4:.2f}s ({5:.0f}%)'
        print(infostr.format(i, tot_time, proc_time, proc_time/tot_time*100, 
                             plot_time, plot_time/tot_time*100, 1.0/tot_time))


    print('* Commencing image acquisition and processing')
    for i, (timestamp, imc) in enumerate(bggen):
        loop(i, timestamp, imc)
        continue
        try:
            loop(i, timestamp, imc)
        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr)
            print(infostr)

def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
