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
import multiprocessing

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
      silcam process <configfile> <datapath> [--nbimages=<number of images>] [--singleProcess]
      silcam -h | --help
      silcam --version

    Arguments:
        acquire     Acquire images
        process     Process images

    Options:
      --nbimages=<number of images>     Number of images to process.
      --singleProcess                   Deactivate multiprocessing
      -h --help                         Show this screen.
      --version                         Show version.

    '''
    print(title)
    print('')
    args = docopt(silcam.__doc__, version='PySilCam {0}'.format(__version__))
    # this is the standard processing method under development now
    if args['process']:
        multiProcess = True
        if args['--singleProcess']:
            multiProcess = False
        nbImages = args['--nbimages']
        if (nbImages != None):
            try:
                nbImages = int(nbImages)
            except ValueError:
                print('Expected type int for --nbimages.')
                sys.exit(0)
        silcam_process(args['<configfile>'],args['<datapath>'], multiProcess, nbImages)

    elif args['acquire']: # this is the standard acquisition method under development now
        silcam_acquire()

def silcam_acquire():
    while True:
        t1 = time.time()
        try:
            aqgen = acquire()
            for i, (timestamp, imraw) in enumerate(aqgen):
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
def silcam_process(config_filename, datapath, multiProcess, nbImages=None, gui=None):

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

    #Initialize the image acquisition generator
    aqgen = acquire(datapath)

    #Get number of images to use for background correction from config
    print('* Initializing background image handler')
    bggen = backgrounder(settings.Background.num_images, aqgen)

    # make datafilename autogenerated for easier batch processing
    if (not os.path.isdir(settings.General.datafile)):
       logger.info('Folder ' + settings.General.datafile + ' was not found and is created')
       os.mkdir(settings.General.datafile)

    procfoldername = os.path.split(datapath)[-1]
    datafilename = os.path.join(settings.General.datafile,procfoldername)
    logger.info('output stats to: ' + datafilename)

    if os.path.isfile(datafilename + '-STATS.csv'):
        logger.info('removing: ' + datafilename + '-STATS.csv')
        os.remove(datafilename + '-STATS.csv')

    # Create export directory if needed
    if settings.ExportParticles.export_images:
       if (not os.path.isdir(settings.ExportParticles.outputpath)):
          logger.info('Export folder ' + settings.ExportParticles.outputpath + ' was not found and is created')
          os.mkdir(settings.ExportParticles.outputpath)

    #---- END SETUP ----

    #---- RUN PROCESSING ----

    proc_list = list()
    inputQueue = multiprocessing.Queue()
    outputQueue = multiprocessing.Queue()

    if (multiProcess):
        distributor(logger, settings, config_filename, datafilename, inputQueue, outputQueue, conf, datapath, proc_list, gui, multiProcess)

    print('* Commencing image acquisition and processing')

    # iterate on the bbgen generator to obtain images
    for i, (timestamp, imc) in enumerate(bggen):
        # handle errors if the loop function fails for any reason
        if (nbImages != None):
            if (nbImages <= i):
                break
        inputQueue.put((i, timestamp, imc)) # the tuple (i, timestamp, imc) is added to the inputQueue

    if (multiProcess):
        for p in proc_list:
            inputQueue.put(None)
    else:
        inputQueue.put(None)
        loop(config_filename, conf, datafilename, inputQueue, outputQueue, 1, gui)
        
    collector(outputQueue, datafilename, proc_list)
    
    for p in proc_list:
        p.join()
        print ('%s.exitcode = %s' % (p.name, p.exitcode) )

    outputQueue.close()
    inputQueue.close()    

    #---- END ----


def loop(config_filename, conf, datafilename, inputQueue, outputQueue, nbCore, gui=None):
    '''
    Main processing loop, run for each image
    '''
    settings = PySilcamSettings(conf)

    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_process')

    # load the model for particle classification and keep it for later
    nnmodel = []
    if settings.NNClassify.enable:
        nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)
    
    while True:
        try:
            task = inputQueue.get()

            if task is None:
                outputQueue.put(None)  
                break

            imc = task[2]
            timestamp = task[1]
            i = task[0]
            
            #Time the full acquisition and processing loop
            start_time = time.clock()

            logger.info('Processing time stamp {0}'.format(timestamp))

            # basic check of image quality
            r = imc[:, :, 0]
            g = imc[:, :, 1]
            b = imc[:, :, 2]
            s = np.std([r, g, b])
            print('lighting std:',s)
            # ignore bad images as if they were not obtained (i.e. do not affect
            # output statistics in any way)
            if s > settings.Process.bad_lighting_limit:
                logger.info('bad lighting')
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

            outputQueue.put(stats_all)

            #Time the particle statistics processing step
            proc_time = time.clock() - start_time

            #Print timing information for this iteration
            infostr = '  Image {0} processed in {1:.2f} sec ({2:.1f} Hz). '
            infostr = infostr.format(i, proc_time, 1.0/proc_time)
            print(infostr)

            #---- END MAIN PROCESSING LOOP ----
            #---- DO SOME ADMIN ----
            if not gui==None:
                guidata = stats_all.to_dict()
                guidata['imc'] = imc
                gui.put(guidata)

        except:
            infostr = 'Failed to process frame {0}, skipping.'.format(i)
            logger.warning(infostr, exc_info=True)
            print(infostr)

def distributor(logger, settings, config_filename, datafilename, inputQueue, outputQueue, conf, datapath, proc_list, gui=None, multiProcess = True):
    '''
    distributes the images in the input queue to the different loop processes
    '''
    
    numCores = 1
    if (multiprocessing.cpu_count() > 2):
        numCores = multiprocessing.cpu_count() - 2

    for nbCore in range(numCores):
        proc = multiprocessing.Process(target=loop, args=(config_filename, conf, datafilename, inputQueue, outputQueue, nbCore, gui))
        proc_list.append(proc)
        proc.start()

def collector(outputQueue, datafilename, proc_list):
    '''
    collects all the results and write them into the stats.csv file
    '''
    countProcessFinished = 0
    while True:
        task = outputQueue.get()
        
        if (task is None):
            countProcessFinished = countProcessFinished + 1
            if (len(proc_list) == 0): # no multiprocessing
                break
            # The collector can be stopped only after all loop processes are finished
            elif (countProcessFinished == len(proc_list)):
                break
            continue

        # create or append particle statistics to output file
        # if the output file does not already exist, create it
        # otherwise data will be appended
        # @todo accidentally appending to an existing file could be dangerous
        # because data will be duplicated (and concentrations would therefore
        # double)
        if not os.path.isfile(datafilename + '-STATS.csv'):
            task.to_csv(datafilename +
                    '-STATS.csv', index_label='particle index')
        else:
            task.to_csv(datafilename + '-STATS.csv',
                    mode='a', header=False)

def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
