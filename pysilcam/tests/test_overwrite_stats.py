# -*- coding: utf-8 -*-
from pysilcam.__main__ import silcam_process
from pysilcam.silcreport import silcam_report
from pysilcam.config import load_config
from pysilcam.postprocess import count_images_in_stats
from pysilcam.config import PySilcamSettings
import glob
import os
import unittest
import pandas as pd
import tempfile
from sys import platform

# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

# pytest on windows can't deal with multiprocessing, so switch it off if windows patform detected
multiProcess = True
if platform == "win32":
    multiProcess = False

print('ROOTPATH', ROOTPATH)
print('MODEL_PATH', MODEL_PATH)


@unittest.skipIf((ROOTPATH is None),
                 "test path not accessible")
def test_output_files():
    '''Testing that the appropriate STATS.h5 file is created'''

    conf_file = os.path.join(ROOTPATH, 'config.ini')
    conf_file_out = os.path.join(ROOTPATH, 'config_generated.ini')
    conf = load_config(conf_file)

    data_file = os.path.join(ROOTPATH, 'STN04')
    conf.set('General', 'loglevel', 'INFO')
    conf.set('General', 'datafile', os.path.join(data_file, 'proc'))
    conf.set('General', 'logfile', os.path.join(ROOTPATH, 'log.log'))
    conf.set('ExportParticles', 'outputpath', os.path.join(data_file, 'export'))
    if MODEL_PATH is not None:
        conf.set('NNClassify', 'model_path', MODEL_PATH)
    with open(conf_file_out, 'w') as conf_file_hand:
        conf.write(conf_file_hand)

    stats_file = os.path.join(data_file, 'proc', 'STN04-STATS.h5')

    # if STATS file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # start by processing 10 images. overwriteSTATS can be True or False here without affecting the result
    requested_images = 10
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False, nbImages=requested_images)
    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
    num_images = count_images_in_stats(stats)
    assert num_images == requested_images, 'incorrect number of images processed'

    # restart, asking for 20 images starting from where we got to before (by using overwriteSTATS=False)
    requested_images = 20
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False, nbImages=requested_images)

    stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
    num_images = count_images_in_stats(stats)
    assert num_images == requested_images, 'incorrect number of images processed'

    # process the rest of the dataset starting from where we stopped
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False)

    # attempt to process a fully-processed dataset
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False)
