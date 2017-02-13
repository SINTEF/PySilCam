# -*- coding: utf-8 -*-
import time
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.aquire import aquire


def silcam_aquire():
    '''Aquire images from the SilCam

    Usage:
      silcam-aquire
      silcam-aquire -h | --help
      silcam-aquire --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(silcam_aquire.__doc__, version='PySilCam {0}'.format(__version__))
    #print('Type \'silcam-aquire -h\' for help')

    t1 = time.time()
    for i, img in enumerate(aquire()):
        t2 = time.time()
        aq_freq = np.round(1.0/(t2 - t1), 1)
        print('Image {0} aquired at frequency {1:.1f} Hz'.format(i, aq_freq))
        t1 = t2


def silcam_process_realtime():
    print('Placeholder for silcam-process-rt entry point')

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')
