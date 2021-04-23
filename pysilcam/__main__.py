# -*- coding: utf-8 -*-
import datetime
import logging
import multiprocessing
import os
import sys
import warnings
from shutil import copyfile
import time

import numpy as np
import pandas as pd
import psutil
from docopt import docopt

import pysilcam.oilgas as scog
import pysilcam.silcam_classify as sccl
from pysilcam import __version__
from pysilcam.acquisition import Acquire, defineQueues
from pysilcam.background import Backgrounder
from pysilcam.config import PySilcamSettings, updatePathLength
from pysilcam.process import processImage, write_stats
from pysilcam.fakepymba import silcam_name2time

if not sys.warnoptions:
    warnings.simplefilter("ignore")

title = 'PySilCam'


def silcam():
    '''Main entry point function to acquire/process images from the SilCam.

    Use this function in command line arguments according to the below documentation.

    Usage:
      silcam acquire <configfile> <datapath>
      silcam process <configfile> <datapath> [--nbimages=<number of images>] [--nomultiproc] [--appendstats]
      silcam realtime <configfile> <datapath> [--discwrite] [--nomultiproc] [--appendstats] [--discread]
      silcam -h | --help
      silcam --version

    Arguments:
        acquire     Acquire images
        process     Process images
        realtime    Acquire images from the camera and process them in real time

    Options:
      --nbimages=<number of images>     Number of images to process.
      --discwrite                       Write images to disc.
      --nomultiproc                     Deactivate multiprocessing.
      --appendstats                     Appends data to output STATS.h5 file. If not specified, the STATS.h5 file will
                                        be overwritten!
      -h --help                         Show this screen.
      --version                         Show version.
      --discread                        emergency disc read version of realtime analysis, to be run seperately but at
                                        the same time as silcam acquire

    '''
    print(title)
    print('')
    args = docopt(silcam.__doc__, version='PySilCam {0}'.format(__version__))

    overwriteSTATS = True

    if args['<datapath>']:
        # The following is solving problems in transfering arguments from shell on windows
        # Remove ' characters
        datapath = os.path.normpath(args['<datapath>'].replace("'", ""))
        # Remove " characters at the end (occurs when user give \" at the end)
        while datapath[-1] == '"':
            datapath = datapath[:-1]

    # this is the standard processing method under development now
    if args['process']:
        multiProcess = True
        if args['--nomultiproc']:
            multiProcess = False
        nbImages = args['--nbimages']
        if nbImages is not None:
            try:
                nbImages = int(nbImages)
            except ValueError:
                print('Expected type int for --nbimages.')
                sys.exit(0)
        if args['--appendstats']:
            overwriteSTATS = False  # if you want to append to the stats file, then overwriting should be False
        silcam_process(args['<configfile>'], datapath, multiProcess=multiProcess, realtime=False,
                       nbImages=nbImages, overwriteSTATS=overwriteSTATS)

    elif args['acquire']:  # this is the standard acquisition method under development now
        try:
            pid = psutil.Process(os.getpid())
            if sys.platform == 'linux':
                pid.nice(-20)
            else:
                pid.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        except:
            print('Could not prioritise acquisition process!')
        silcam_acquire(datapath, args['<configfile>'], writeToDisk=True)

    elif args['realtime']:
        discWrite = False
        if args['--discwrite']:
            discWrite = True
        multiProcess = True
        if args['--nomultiproc']:
            multiProcess = False
        if args['--appendstats']:
            overwriteSTATS = False  # if you want to append to the stats file, then overwriting should be False
        if args['--discread']:
            os.environ['REALTIME_DISC'] = ''
            print('discWrite = False')
            discWrite = False
        silcam_process(args['<configfile>'], datapath, multiProcess=multiProcess, realtime=True,
                       discWrite=discWrite, overwriteSTATS=overwriteSTATS)


def silcam_acquire_simple(datapath, config_filename, writeToDisk=True, gui=None):
    '''Aquire images from the SilCam

    Args:
       datapath              (str)          :  Path to the image storage
       config_filename=None  (str)          :  Camera config file
       writeToDisk=True      (Bool)         :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
       gui=None          (Class object)     :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''

    # Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    # Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    if writeToDisk:
        # Copy config file
        configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
        copyfile(config_filename, os.path.join(datapath, configFile2Copy))

    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_acquire')

    # update path_length
    updatePathLength(settings, logger)

    acq = Acquire(USE_PYMBA=True, datapath=datapath, writeToDisk=writeToDisk, gui=gui)  # ini class

    acq.stream_images(camera_config_file=config_filename)


def silcam_acquire(datapath, config_filename, writeToDisk=True, gui=None):
    '''dev for processing with VimbaPython

    Args:
       datapath              (str)          :  Path to the image storage
       config_filename=None  (str)          :  Camera config file
       writeToDisk=True      (Bool)         :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
       gui=None          (Class object)     :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''

    # Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    # Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    if writeToDisk:
        # Copy config file
        configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
        copyfile(config_filename, os.path.join(datapath, configFile2Copy))

    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_acquire')

    # update path_length
    updatePathLength(settings, logger)

    # create a last in first out queue eventually to be part of the background module
    # manager = MyManager()
    # manager.start()
    # raw_image_queue = manager.LifoQueue(2)  # limit queue size to two images
    raw_image_queue = multiprocessing.Queue(2)
    print("raw image queue initialised.")

    backgrounder = Backgrounder(settings.Background.num_images,
                                bad_lighting_limit=settings.Process.bad_lighting_limit,
                                real_time_stats=settings.Process.real_time_stats)
    backgrounder_process = multiprocessing.Process(target=backgrounder.run, args=(config_filename, raw_image_queue, None))
    backgrounder_process.start()
    print("backgrounder_process started.")

    acq = Acquire(USE_PYMBA=False, datapath=datapath, writeToDisk=False, gui=gui, raw_image_queue=raw_image_queue)  # ini class

    acq.stream_images(config_file=config_filename)

    backgrounder_process.join()  # shut down subprocesses after stopping (needs checking how to do this)


# the standard processing method under active development
def silcam_process(config_filename, datapath, multiProcess=True, realtime=False,
                   discWrite=False, nbImages=None, gui=None, overwriteSTATS=True):
    '''Run processing of SilCam images

    Args:
      config_filename   (str)               :  The filename (including path) of the config.ini file
      datapath          (str)               :  Path to the data directory
      multiProcess=True (bool)              :  If True, multiprocessing is used
      realtime=False    (bool)              :  If True, a faster but less accurate methods is used for segmentation and
                                               rts stats become active
      discWrite=False   (bool)              :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
      nbImages=None     (int)               :  Number of images to process
      gui=None          (Class object)      :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''
    # ---- SETUP ----

    # Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    # Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    # Configure logging
    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.silcam_process')

    logger.info('Processing path: ' + datapath)

    if realtime:
        if discWrite:
            # copy config file into data path
            configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
            copyfile(config_filename, os.path.join(datapath, configFile2Copy))

        # update path_length
        updatePathLength(settings, logger)

    print(' realtime', realtime)

    # make datafilename autogenerated for easier batch processing
    if not os.path.isdir(settings.General.datafile):
        logger.info('Folder ' + settings.General.datafile + ' was not found and is created')
        os.mkdir(settings.General.datafile)

    procfoldername = os.path.split(datapath)[-1]
    datafilename = os.path.join(settings.General.datafile, procfoldername)
    logger.info('output stats to: ' + datafilename)

    sccl.check_model(settings.NNClassify.model_path)

    fakepymba_offset = 0
    datafile_hdf = datafilename + '-STATS.h5'
    if os.path.isfile(datafile_hdf):
        with pd.HDFStore(datafile_hdf, 'r') as f:
            datafile_keys = f.keys()
        # Remove old STATS file if it exists
        if overwriteSTATS:
            logger.info('removing: ' + datafile_hdf)
            print('Overwriting ' + datafile_hdf)
            os.remove(datafile_hdf)
        elif '/ParticleStats/stats' not in datafile_keys:
            logger.info('Stats file present, but no data written: fakepymba_offset = 0')
            pass
        else:
            # If we are starting from an existings stats file, update the
            # PYILSCAM_OFFSET environment variable
            fakepymba_offset = update_pysilcam_offset(logger, settings, datafilename, datapath)

    # Create new HDF store and write PySilcam version and
    # current datetime as root attribute metadata
    if not os.path.isfile(datafile_hdf):
        with pd.HDFStore(datafile_hdf, 'w') as fh:
            fh.root._v_attrs.timestamp = str(datetime.datetime.now())
            fh.root._v_attrs.pysilcam_version = str(__version__)

    # Create export directory if needed
    if settings.ExportParticles.export_images:
        if not os.path.isdir(settings.ExportParticles.outputpath):
            logger.info('Export folder ' + settings.ExportParticles.outputpath + ' was not found and is created')
            os.mkdir(settings.ExportParticles.outputpath)

    # ---- END SETUP ----

    # ---- RUN PROCESSING ----

    # ==== Start proc_image_queues (input/outputQueue) and processing workers:

    # FOR NOW, Ignoring case where not multiprocessing:
    # multiProcess = multiProcess and (multiprocessing.cpu_count() > 1)

    print('* Commencing image acquisition and processing')

    # initialise realtime stats class regardless of whether it is used later
    rts = scog.rt_stats(settings)

    proc_list = []
    mem = psutil.virtual_memory()
    memAvailableMb = mem.available >> 20
    distributor_q_size = np.min([int(memAvailableMb / 2 * 1 / 15), np.copy(multiprocessing.cpu_count() * 4)])

    distributor_q_size = 1

    logger.debug('setting up processing queues')
    proc_image_queue, outputQueue = defineQueues(realtime, distributor_q_size)

    logger.debug('setting up processing distributor')
    proc_list = distributor(proc_image_queue, outputQueue, config_filename, proc_list, gui)

    # ==== Setup raw_image_queue and start backgrounder process:

    logger.debug('Setting up raw image queue')
    raw_image_queue = multiprocessing.Queue(1)
    print("raw image queue initialised.")

    # There is something odd here with realtime and real_time_stats.
    backgrounder = Backgrounder(settings.Background.num_images,
                                raw_image_queue,
                                proc_image_queue=proc_image_queue,
                                bad_lighting_limit=settings.Process.bad_lighting_limit,
                                real_time_stats=realtime)  # real_time_stats=settings.Process.real_time_stats)
    bg_process = backgrounder.start_backgrounder()

    # FOR NOW, I'll ignore processing time tracking, as it's not goverened in the same way.
    # t2 = time.time()

    # ==== Start image acquisition and streaming to raw_image_queue
    # implement later:
    # if 'REALTIME_DISC' in os.environ.keys():
    #    print('acq = Acquire(USE_PYMBA=False)')
    #    aq = Acquire(USE_PYMBA=False)
    # else:
    acq = Acquire(USE_PYMBA=realtime, datapath=datapath, writeToDisk=discWrite,
                  raw_image_queue=raw_image_queue, gui=gui)
    logger.debug('__main__ acq.stream_images(config_file=config_filename)')
    acq_process = acq.stream_images(config_file=config_filename)

    # Gui stuff?????
    # if gui is not None:
    #     logger.debug('Putting data on GUI Queue')
    #     while (gui.qsize() > 0):
    #         try:
    #             gui.get_nowait()
    #             time.sleep(0.001)
    #         except:
    #             continue
    #     # try:
    #     rtdict = dict()
    #     rtdict = {'dias': rts.dias,
    #                 'vd_oil': rts.vd_oil,
    #                 'vd_gas': rts.vd_gas,
    #                 'gor': rts.gor,
    #                 'oil_d50': rts.oil_d50,
    #                 'gas_d50': rts.gas_d50,
    #                 'saturation': rts.saturation}
    #     gui.put_nowait((timestamp, imc, imraw, rtdict))
    #     logger.debug('GUI queue updated')

    # implement later
    # if 'REALTIME_DISC' in os.environ.keys():
    #   scog.realtime_summary(datafilename + '-STATS.h5', config_filename)

    logger.debug('!C: Running collector')
    while acq_process.is_alive():
        collector(proc_image_queue, outputQueue, datafilename, proc_list, False, settings, rts=rts)
        time.sleep(0.5)
    logger.debug('!C: Data collected')

    acq_process.join()
    logger.info('!C: acq_process.join(): %s.exitcode = %s' % (acq_process.name, acq_process.exitcode))

    # some images might still be waiting to be written to the csv file
    logger.debug('!C: Running collector on left over data')
    collector(proc_image_queue, outputQueue, datafilename, proc_list, True,
              settings, rts=rts)
    logger.debug('!C: All data collected')

    logger.debug(('!C: proc_list:', proc_list))
    for p in proc_list:
        p.join()
        logger.info('!C: proc_list.join(): %s.exitcode = %s' % (p.name, p.exitcode))
    logger.debug(('!C: proc_list joined'))

    bg_process.terminate()
    logger.info('!C: bg_process.terminate(): %s.exitcode = %s' % (bg_process.name, bg_process.exitcode))

    print('!C: PROCESSING COMPLETE.')

    # ---- END ----


def loop(config_filename, proc_image_queue, outputQueue, gui=None):
    '''
    Main processing loop, run for each image

    Args:
        config_filename (str)   : path of the config ini file
        proc_image_queue  ()          : queue where the images are added for processing
                                  initilised using defineQueues()
        outputQueue ()          : queue where information is retrieved from processing
                                  initilised using defineQueues()
        gui=None (Class object) : Queue used to pass information between process thread and GUI
                                  initialised in ProcThread within guicals.py
    '''
    settings = PySilcamSettings(config_filename)
    configure_logger(settings.General)
    logger = logging.getLogger(__name__ + '.loop')

    # load the model for particle classification and keep it for later
    logger.info('load the model for particle classification and keep it for later')
    nnmodel = []
    nnmodel, class_labels = sccl.load_model(model_path=settings.NNClassify.model_path)
    logger.info('sccl.load_model - OK.')

    while True:
        logger.debug('proc_image_tuple = proc_image_queue.get()')
        proc_image_tuple = proc_image_queue.get()
        logger.debug('proc_image_tuple = proc_image_queue.get()  OK.')
        if proc_image_tuple is None:
            proc_image_queue.put(None)  # put the None back on proc_image_queue so other processes can see
            logger.debug("!C: Received none from inputQueue, shutting down.")
            logger.debug('outputQueue.put(None)')
            outputQueue.put(None)
            logger.debug('!C: outputQueue.put(None) OK')
            break
        stats_all = processImage(nnmodel, class_labels, proc_image_tuple, settings, logger, gui)

        if stats_all is not None:
            logger.debug('outputQueue.put(stats_all)')
            outputQueue.put(stats_all)
            logger.debug('outputQueue.put(stats_all)  OK.')
        else:
            logger.info('No stats found.')


def distributor(proc_image_queue, outputQueue, config_filename, proc_list, gui=None):
    '''
    distributes the images in the input queue to the different loop processes
    Args:
        inputQueue  ()              : queue where the images are added for processing
                                      initilised using defineQueues()
        outputQueue ()              : queue where information is retrieved from processing
                                      initilised using defineQueues()
        proc_list   (list)          : list of multiprocessing objects
        gui=None (Class object)     : Queue used to pass information between process thread and GUI
                                      initialised in ProcThread within guicals.py
    '''

    numCores = max(1, multiprocessing.cpu_count() - 2)
    numCores = 2

    for nbCore in range(numCores):
        proc = multiprocessing.Process(target=loop, args=(config_filename, proc_image_queue, outputQueue, gui))
        proc_list.append(proc)
        proc.start()
    return proc_list


def collector(proc_image_queue, outputQueue, datafilename, proc_list, testInputQueue,
              settings, rts=None):
    '''
    collects all the results and write them into the stats.h5 file

    Args:
        proc_image_queue  ()              : queue where the images are added for processing
                                      initilised using defineQueues()
        outputQueue ()              : queue where information is retrieved from processing
                                      initilised using defineQueues()
        datafilename (str)          : filename where processed data are written to csv
        proc_list   (list)          : list of multiprocessing objects
        testInputQueue (Bool)       : if True function will keep collecting until inputQueue is empty
        settings (PySilcamSettings) : Settings read from a .ini file
        rts (Class):                : Class for realtime stats
    '''

    countProcessFinished = 0

    logger = logging.getLogger(__name__ + '.collector')

    while ((outputQueue.qsize() > 0) or (testInputQueue and proc_image_queue.qsize() > 0)):
        logger.debug(('__main__ outputQueue.qsize(): ', outputQueue.qsize()))
        logger.debug(('__main__  proc_image_queue.qsize():', proc_image_queue.qsize()))

        stats_all = outputQueue.get()
        logger.debug('__main__  got stats_all from outputQueue')

        if stats_all is None:
            logger.debug("received None from outputQueue, wrapping up")
            logger.debug(("len(proc_list)", len(proc_list)))
            countProcessFinished = countProcessFinished + 1
            if len(proc_list) == 0:  # no multiprocessing
                break
            # The collector can be stopped only after all loop processes are finished
            elif countProcessFinished == len(proc_list):
                break
            continue

        write_stats(datafilename, stats_all)
        collect_rts(settings, rts, stats_all)


def collect_rts(settings, rts, stats_all):
    '''
    Updater for realtime statistics

    Args:
        settings (PySilcamSettings) : Settings read from a .ini file
                                      settings.logfile is optional
                                       settings.loglevel mest exist
        rts (Class)                 :  Class for realtime stats
                                       initialised using scog.rt_stats()
        stats_all (DataFrame)       :  stats dataframe returned from processImage()
    '''
    if settings.Process.real_time_stats:
        try:
            rts.stats = rts.stats().append(stats_all)
        except:
            rts.stats = rts.stats.append(stats_all)
        rts.update()
        filename = os.path.join(settings.General.datafile,
                                'OilGasd50.csv')
        rts.to_csv(filename)


def check_path(filename):
    '''Check if a path exists, and create it if not

    Args:
        filename (str): filame that may or may not include a path
    '''

    file = os.path.normpath(filename)
    path = os.path.dirname(file)
    if path:
        if not os.path.isdir(path):
            try:
                os.makedirs(path)
            except:
                print('Could not create catalog:', path)


def configure_logger(settings):
    '''Configure a logger according to the settings.

    Args:
        settings (PySilcamSettings): Settings read from a .ini file
                                     settings.logfile is optional
                                     settings.loglevel mest exist
    '''
    if settings.logfile:
        check_path(settings.logfile)
        logging.basicConfig(filename=settings.logfile,
                            level=getattr(logging, settings.loglevel))

    else:
        logging.basicConfig(level=getattr(logging, settings.loglevel))


def update_pysilcam_offset(logger, settings, datafilename, datapath):
    '''
    Set the offset for image loading based on stats file.

    Args:
        logger          (logger object) : logger object created using configure_logger()
        datafilename    (str)           : name of the folder containing the -STATS.h5
        datapath        (str)           : name of the path containing the data

    '''

    datafile_hdf = datafilename + '-STATS.h5'
    logger.info('Loading old data from: ' + datafile_hdf)
    print('Loading old data from: ' + datafile_hdf)
    oldstats = pd.read_hdf(datafile_hdf, 'ParticleStats/stats')
    logger.info('  OK.')
    print('  OK.')
    last_time = pd.to_datetime(oldstats['timestamp'].max())

    logger.info('Calculating spooling offset')
    print('Calculating spooling offset')

    files = [f for f in sorted(os.listdir(datapath))
             if f.endswith('.silc') or f.endswith('.bmp')]
    offsetcalc = pd.DataFrame(columns=['files', 'times'])
    offsetcalc['files'] = files
    for i, f in enumerate(files):
        offsetcalc['times'].iloc[i] = silcam_name2time(f)
    # find the index of the first time in offsetcalc[]'times'] that is after last_time
    if last_time >= np.max(offsetcalc['times']):
        offset = len(files)
    else:
        offset = int(np.min(np.where(offsetcalc['times'] > last_time)))
    # subtract the number of background images, so we get data from the correct start point
    offset -= settings.Background.num_images
    # and check offset is still positive
    if offset < 0:
        offset = 0
    offset = str(offset)
    logger.info('offset set to: ' + offset)
    print('offset set to: ' + offset)
    offset = int(offset)

    return offset
