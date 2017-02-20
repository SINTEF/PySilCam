# -*- coding: utf-8 -*-
import time
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
from pysilcam import __version__
from pysilcam.acquisition import acquire
from pysilcam.background import backgrounder
from pysilcam.process import statextract
import pysilcam.postprocess as sc_pp
import pandas as pd
import cProfile
import pstats
from io import StringIO


def silcam_acquire():
    '''Aquire images from the SilCam

    Usage:
      silcam-acquire
      silcam-acquire -h | --help
      silcam-acquire --version
      silcam-acquire liveview
      silcam-acquire process

    Arguments:
        liveview    Display acquired images
        process     Process acquired images

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(silcam_acquire.__doc__, version='PySilCam {0}'.format(__version__))
    #print('Type \'silcam-acquire -h\' for help')

    if args['process']:
        pr = cProfile.Profile()
        pr.enable()
        s = StringIO()
        sortby = 'cumulative'
        silcam_process_realtime()
        pr.disable()
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
#        print(s.getvalue())
        ps.dump_stats('process_profile.cprof')

    elif args['liveview']:
        plt.ion()
        fig, ax = plt.subplots()
        t1 = time.time()
        for i, img in enumerate(acquire()):
            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2
            ax.imshow(img[:,:,0], cmap=plt.cm.gray)
            plt.draw()
            plt.pause(0.05)

    else:
        t1 = time.time()
        for i, img in enumerate(acquire()):
            t2 = time.time()
            aq_freq = np.round(1.0/(t2 - t1), 1)
            print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
            t1 = t2



def silcam_process_realtime():

    #Initialize the image acquisition generator
    aqgen = acquire()

    for i, imc in enumerate(backgrounder(10, aqgen)):
    #for i, imc in enumerate(acquire()):
#        plt.imshow(np.uint8(imc))
#        plt.show()
        print('PROCESSING....')
        start_time = time.clock()
        plt.figure()
        stats = statextract(imc, i)
        proc_time = time.clock() - start_time
        print('PROCESSING DONE in', proc_time, 'sec.')


#        stats.to_csv('/home/emlynd/Desktop/data/test-' + str(i) + '.csv')

        if stats is not np.nan:
            print('data has arrived!')
#            print(stats)
#            break
        d50 = sc_pp.d50_from_stats(stats)
        print('d50:', d50)
        plt.figure()
        plt.imshow(np.uint8(imc))
        plt.draw()
        plt.show()
        break

#    print('Placeholder for silcam-process-rt entry point')

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
