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
import os

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

    if args['fancyprocess']:
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

    nnmodel = []
    if settings.NNClassify.enable:
        import pysilcam.silcam_classify as sccl
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

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

    datafilename = settings.General.datafile

    psddatafile = datalogger.DataLogger(datafilename + '.csv',
            oilgas.ogdataheader())

    def loop(i, timestamp, imc):
        #Time the full acquisition and processing loop
        start_time = time.clock()

        logger.info('Processing time stamp {0}'.format(timestamp))

        nc = color.guess_spatial_dimensions(imc)
        if nc == None: # @todo FIX if there are ambiguous dimentions, assume RGB color space
        #    imc = imc[:,:,1] # and just use green
            #Calculate particle statistics

            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            print('lighting std:',s)
            if s > 4:
                print('bad lighting')
                return

            img = np.uint8(np.min(imc, axis=2))
            stats_all, imbw, saturation = statextract(img, settings,
                    fancy=True)
        else:
            stats_all, imbw, saturation = statextract(imc, settings, fancy=True)

        if (settings.ExportParticles.export_images) or (settings.NNClassify.enable):
            stats_all = exportparts.export_particles(imc, timestamp, stats_all,
                    settings, nnmodel, class_labels)


        if len(stats_all) == 0:
            print('ZERO particles idenfitied')
            z = np.zeros(len(stats_all.columns)) * np.nan
            stats_all.loc[0] = z 
            stats_all['export name'] = 'not_exported'

        stats_all['timestamp'] = timestamp

        if not os.path.isfile(datafilename):
            stats_all.to_csv(datafilename +
                    '-STATS.csv', index_label='particle index') 
        else:
            stats_all.to_csv(datafilename + '-STATS.csv',
                    mode='a', header=False) 


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


        tot_time = time.clock() - start_time

        #Print timing information for this iteration
        plot_time = tot_time - proc_time
        infostr = '  Image {0} processed in {1:.2f} sec ({6:.1f} Hz). '
        infostr += 'Statextract: {2:.2f}s ({3:.0f}%) Plot: {4:.2f}s ({5:.0f}%)'
        print(infostr.format(i, tot_time, proc_time, proc_time/tot_time*100, 
                             plot_time, plot_time/tot_time*100, 1.0/tot_time))


#        import seaborn as sns
#        sns.set_style('ticks')
#
#        f, a = plt.subplots(1,2, figsize=(20,10))
#
#        plt.sca(a[0])
#        plt.imshow(imc)
#
#        plt.sca(a[1])
#        plt.imshow(imbw, interpolation='nearest')
#
#        filename__ = timestamp.strftime('D%Y%m%dT%H%M%S.%f')
#        plt.savefig('/home/emlynd/Desktop/dump/' + filename__ + '-imbw.png')
#        plt.close('all')



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
