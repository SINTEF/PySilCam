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
from pysilcam.config import load_config, PySilcamSettings
from skimage import color
import imageio
import os
import pysilcam.silcam_classify as sccl
import pysilcam.sclive as scl
import pysilcam.oilgas as scog


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


def silcam():
    '''Aquire/process images from the SilCam

    Usage:
      silcam acquire [-l | --liveview]
      silcam process <configfile> <datapath> [--nbimages=<number of images>]
      silcam -h | --help
      silcam --version

    Arguments:
        acquire     Acquire images
        process     Process images

    Options:
      --nbimages=<number of images>     Number of images to process.
      -h --help                         Show this screen.
      --version                         Show version.

    '''
    print(title)
    print('')
    args = docopt(silcam.__doc__, version='PySilCam {0}'.format(__version__))
    # this is the standard processing method under development now
    if args['process']:
        nbImages = args['--nbimages']
        if (nbImages != None):
            try:
                nbImages = int(nbImages)
            except ValueError:
                print('Expected type int for --nbimages.')
                sys.exit(0)
        silcam_process(args['<configfile>'],args['<datapath>'], nbImages)

    elif args['acquire']: # this is the standard acquisition method under development now
        if args['--liveview']:
            silcam_acquire(liveview=True)
        else:
            silcam_acquire()

def silcam_sim(datapath, gui):

    aqgen = acquire(datapath)
    bggen = backgrounder(15, aqgen)

    start_time = time.clock()
    for i, (timestamp, imc) in enumerate(bggen):
        proc_time = time.clock() - start_time
        start_time = time.clock()
        infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
        infostr = infostr.format(i, proc_time, 1.0/proc_time)
        print(infostr)
        guidata = {'imc': imc,
            'timestamp': timestamp,
            'infostr': infostr,
            'dias': 1,
            'vd_oil': 1,
            'vd_gas': 1
            }
        gui.put(guidata)


def silcam_acquire(liveview=False):
    if liveview:
        lv = scl.liveview()
    while True:
        t1 = time.time()
        try:
            aqgen = acquire()
            for i, (timestamp, imraw) in enumerate(aqgen):
                if liveview:
                    lv = lv.update(imraw, timestamp)
                    if not lv.record:
                        continue
                filename = timestamp.strftime('D%Y%m%dT%H%M%S.%f.silc')
                with open(filename, 'wb') as fh:
                    np.save(fh, imraw, allow_pickle=False)
                    fh.flush()
                    os.fsync(fh.fileno())
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
def silcam_process(config_filename, datapath, nbImages=None, gui=None):

    '''Run processing of SilCam images

    The goal is to make this as fast as possible so it can be used in real-time

    Function requires the filename (including path) of the config.ini file
    which contains the processing settings

    '''
    print(config_filename)

    print('PROCESS MODE')
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
    logger = logging.getLogger(__name__ + '.silcam_process')

    logger.info('Processing path: ' + datapath)

    # load the model for particle classification and keep it for later
    nnmodel = []
    if settings.NNClassify.enable:
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)

    # this can be initialised and then never used
    rts = scog.rt_stats(settings)

    if settings.Process.display:
        lv = scl.liveview()
    else:
        lv=[]

    #Initialize the image acquisition generator
    aqgen = acquire(datapath)

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen)


    # setup realtime plotting if display is activated in config
    #if settings.Process.display:
    #    logger.info('Initializing real-time plotting')
    #    rtplot = scplt.ParticleSizeDistPlot()

    # @todo: make datafilename autogenerated for easier batch processing
    procfoldername = os.path.split(datapath)[-1]
    datafilename = os.path.join(settings.General.datafile,procfoldername)
    logger.info('output stats to: ' + datafilename)

    if os.path.isfile(datafilename + '-STATS.csv'):
        logger.info('removing: ' + datafilename + '-STATS.csv')
        os.remove(datafilename + '-STATS.csv')

    #---- END SETUP ----


    #---- MAIN PROCESSING LOOP ----
    # processing function run on each image
    def loop(i, timestamp, imc, lv=None, rts=None, gui=None):
        #Time the full acquisition and processing loop
        start_time = time.clock()


        logger.info('Processing time stamp {0}'.format(timestamp))

        # basic check of image quality
        if False:
            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            print('lighting std:',s)
            # ignore bad images as if they were not obtained (i.e. do not affect
            # output statistics in any way)
            if s > 8:
                print('bad lighting')
                return

        #Calculate particle statistics
        stats_all, imbw, saturation = statextract(imc, settings, timestamp,
                                                  nnmodel, class_labels)

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
            if settings.ExportParticles.export_images:
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

        #Print timing information for this iteration
        infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
        infostr = infostr.format(i, proc_time, 1.0/proc_time)
        print(infostr)

        #---- END MAIN PROCESSING LOOP ----
        #---- DO SOME ADMIN ----
        if settings.Process.real_time_stats:
            try:
                rts.stats = rts.stats().append(stats_all)
            except:
                rts.stats = rts.stats.append(stats_all)
            rts.update()
            if settings.Process.display:
                lv.stats = rts.stats
                lv.oil_stats = rts.oil_stats
                lv.gas_stats = rts.gas_stats

        if settings.Process.display:
            lv = lv.update(imc, settings)

        if not gui==None:
            print('putting data on gui queue')
            dias, vd_oil = sc_pp.vd_from_stats(rts.oil_stats,
                    settings.PostProcess)
            dias, vd_gas = sc_pp.vd_from_stats(rts.gas_stats,
                    settings.PostProcess)
            guidata = {'imc': imc,
                    'timestamp': timestamp,
                    'infostr': infostr,
                    'dias': dias,
                    'vd_oil': vd_oil,
                    'vd_gas': vd_gas
                    }
            gui.put(guidata)
            print('  OK.')

    #---- RUN PROCESSING ----

    print('* Commencing image acquisition and processing')
    # iterate on the bbgen generator to obtain images
    for i, (timestamp, imc) in enumerate(bggen):
        # handle errors if the loop function fails for any reason
        if (nbImages != None):
            if (nbImages <= i):
                break
        loop(i, timestamp, imc, lv, rts, gui)
        continue
        try:
            loop(i, timestamp, imc, lv, rts, gui)
        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr, exc_info=True)
            print(infostr)

    #---- END ----


def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
