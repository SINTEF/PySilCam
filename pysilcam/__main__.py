# -*- coding: utf-8 -*-
import time
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.acquisition import acquire


def silcam_acquire():
    '''Aquire images from the SilCam

    Usage:
      silcam-acquire
      silcam-acquire -h | --help
      silcam-acquire --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(silcam_acquire.__doc__, version='PySilCam {0}'.format(__version__))
    #print('Type \'silcam-acquire -h\' for help')

    t1 = time.time()
    for i, img in enumerate(acquire()):
        t2 = time.time()
        aq_freq = np.round(1.0/(t2 - t1), 1)
        print('Image {0} acquired at frequency {1:.1f} Hz'.format(i, aq_freq))
        t1 = t2


def silcam_process_realtime():
    print('Placeholder for silcam-process-rt entry point')

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
