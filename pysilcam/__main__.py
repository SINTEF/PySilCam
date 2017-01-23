# -*- coding: utf-8 -*-
from docopt import docopt
from pysilcam import __version__


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
    print('Type \'silcam-aquire -h\' for help')


def silcam_process_realtime():
    print('Placeholder for silcam-process-rt entry point')

    
def silcam_process_batch():
    print('Placeholder for silcam-process-batch entry point')

