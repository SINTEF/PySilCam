# -*- coding: utf-8 -*-
import sys
import time
import logging
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cProfile
import pstats
from io import StringIO
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
import pysilcam.silcam_classify as sccl


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

    # this is the standard processing method under development now
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

    else: # this is the standard acquisition method under development now
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
                    t1 = time.time()
            except KeyboardInterrupt:
                print('User interrupt with ctrl+c, terminating PySilCam.')
                sys.exit(0)
            except:
                etype, emsg, etrace = sys.exc_info()
                print('Exception occurred: {0}. Restarting acquisition.'.format(emsg))
 

# the standard processing method under active development
def silcam_process_fancy(config_filename):
    '''Run processing of SilCam images
    
    The goal is to make this as fast as possible so it can be used in real-time

    Function requires the filename (including path) of the config.ini file
    which contains the processing settings

    '''
    print(config_filename)

    print('REALTIME MODE')
    print('')

    #---- SETUP ----

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

    # load the model for particle classification and keep it for later
    nnmodel = []
    if settings.NNClassify.enable:
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

    #Initialize the image acquisition generator
    aqgen = acquire()

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen)


    # setup realtime plotting if display is activated in config
    if settings.Process.display:
        logger.info('Initializing real-time plotting')
        rtplot = scplt.ParticleSizeDistPlot()

    # @todo: make datafilename autogenerated to easier batch processing
    datafilename = settings.General.datafile

    #---- END SETUP ----


    #---- MAIN PROCESSING LOOP ----
    # processing function run on each image
    def loop(i, timestamp, imc):
        #Time the full acquisition and processing loop
        start_time = time.clock()

        logger.info('Processing time stamp {0}'.format(timestamp))

        # basic check of image quality
        r = imc[:, :, 0]
        g = imc[:, :, 1]
        b = imc[:, :, 2]
        s = np.std([r, g, b])
        print('lighting std:',s)
        # ignore bas images as if they were not obtained (i.e. do not affect
        # output statistics in any way)
        if s > 4:
            print('bad lighting')
            return

        # simplyfy processing by squeezing the image dimentions into a 2D array
        # min is used for squeezing to represent the highest attenuation of all wavelengths
        img = np.uint8(np.min(imc, axis=2))

        #Calculate particle statistics
        stats_all, imbw, saturation = statextract(img, settings,
                fancy=True)

        # Export particle images if enabled in config file
        # also use this function for the NN calssification (because it is the
        # fastest way to access particle ROIs used for classification
        # @todo tidy up this
        if (settings.ExportParticles.export_images) or (settings.NNClassify.enable):
            stats_all = exportparts.export_particles(imc, timestamp, stats_all,
                    settings, nnmodel, class_labels)

        # if there are not particles identified, assume zero concentration.
        # This means that the data should indicate that a 'good' image was
        # obtained, without any particles. Therefore fill all values with nans
        # and add the image timestamp
        if len(stats_all) == 0:
            print('ZERO particles idenfitied')
            z = np.zeros(len(stats_all.columns)) * np.nan
            stats_all.loc[0] = z
            # 'export name' should not be nan because then this column of the
            # DataFrame will contain multiple types, so label with string instead
            if settings.ExportParticles.export_image:
                stats_all['export name'] = 'not_exported'

        # add timestamp to each row of particle statistics
        stats_all['timestamp'] = timestamp

        # create or append particle statistics to output file
        # if the output file does not already exist, create it
        # otherwise data will be appended
        # @todo accidentally appending to an existing file could be dangerous
        # because data will be duplicated (and concentrations would therefore
        # double)
        if not os.path.isfile(datafilename + '-STATS.csv'):
            stats_all.to_csv(datafilename +
                    '-STATS.csv', index_label='particle index') 
        else:
            stats_all.to_csv(datafilename + '-STATS.csv',
                    mode='a', header=False) 

        #Time the particle statistics processing step
        proc_time = time.clock() - start_time

        #If real-time plotting is enabled, update the plots
        # @todo realtime plotting will no longer work because the averaged
        # time-series data is no longer calculated here. This should be fixed
        # by reading of the STATS.csv or storing stats in memory
        if settings.Process.display:
            if i == 0:
                rtplot.plot(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)
            else:
                rtplot.update(imc, imbw, times, d50_ts['total'], vd_mean,
                        settings.Process.display)

        # figure out how long this took
        tot_time = time.clock() - start_time

        #Print timing information for this iteration
        plot_time = tot_time - proc_time
        infostr = '  Image {0} processed in {1:.2f} sec ({6:.1f} Hz). '
        infostr += 'Statextract: {2:.2f}s ({3:.0f}%) Plot: {4:.2f}s ({5:.0f}%)'
        print(infostr.format(i, tot_time, proc_time, proc_time/tot_time*100, 
                             plot_time, plot_time/tot_time*100, 1.0/tot_time))

        #---- END MAIN PROCESSING LOOP ----

    #---- RUN PROCESSING ----

    print('* Commencing image acquisition and processing')
    # iterate on the bbgen generator to obtain images
    for i, (timestamp, imc) in enumerate(bggen):
        # handle errors if the loop function fails for any reason
        try:
            loop(i, timestamp, imc)
        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr)
            print(infostr)

    #---- END ----


def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
