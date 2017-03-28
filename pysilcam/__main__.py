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
            ax.imshow(img[:,:,0], cmap=plt.cm.gray)
            #plt.draw()
            plt.pause(0.05)

    else:
        t1 = time.time()

        count = 0
        for i, img in enumerate(acquire()):
            #imageio.imwrite('../data/foo{0}.bmp'.format(count), img)
            #print("image {} ".format(count))
            count += 1

            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2





def silcam_process_realtime(config_filename):
    '''Run processing of SilCam images in real time'''
    print(config_filename)

    print('REALTIME MODE')
    print()
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

    d50_ts = []

#    plt.ion()
    fig, ax = plt.subplots(2,2)

    print('* Commencing image acquisition and processing')
    for i, imc in enumerate(bggen):
        #logger.debug('PROCESSING....')
        start_time = time.clock()
        stats = statextract(imc, settings.Process)
        proc_time = time.clock() - start_time

        plt.axes(ax[0,0])
#        plt.cla()
#        scplt.show_imc(imc)
        if i == 0:
            image = plt.imshow(np.uint8(imc), interpolation='nearest', animated=True)
        image.set_data(np.uint8(imc))

        if stats is not np.nan:
            logger.debug('data has arrived!')
        else:
#            plt.pause(0.05)
#            plt.draw()
#            tot_time = time.clock() - start_time
#            print('  Processing image {0} took {1} sec. out of {2} sec.'.format(i, proc_time, tot_time))

            continue
        stats = sc_pp.filter_stats(stats, settings.PostProcess)
        d50 = sc_pp.d50_from_stats(stats, settings.PostProcess)
        print('d50:', d50)

#        tot_time = time.clock() - start_time
#        print('  Processing image {0} took {1} sec. out of {2} sec.'.format(i,
#            proc_time, tot_time))
#        continue

        d50_ts.append(d50)

        plt.axes(ax[1,0])
        plt.cla()
        plt.plot(d50_ts,'.')
        plt.xlabel('image #')
        plt.ylabel('d50 (um)')

        plt.axes(ax[1,1])
        plt.cla()
        scplt.psd(stats, settings.PostProcess)
        plt.pause(0.01)
        plt.draw()
        tot_time = time.clock() - start_time

        #logger.info('PROCESSING DONE in {0} sec.'.format(proc_time))
        print('  Processing image {0} took {1} sec. out of {2} sec.'.format(i,
            proc_time, tot_time))

        if (i == 6000):
#            plt.savefig('/home/emlynd/Desktop/dump.png')
            break

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
