# -*- coding: utf-8 -*-
import glob
from pysilcam.__main__ import silcam_process
from pysilcam.config import load_config
from pysilcam.postprocess import count_images_in_stats
import os
import unittest
import pandas as pd
import tempfile
from sys import platform
from pysilcam.config import PySilcamSettings

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
def test_overwrite_stats():
    '''Testing silcam_process with different combinations of overwriteSTATS and checking for expected output'''

    with tempfile.TemporaryDirectory() as tempdir:
        print('tempdir:', tempdir, 'created')

        conf_file = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards.ini')
        conf_file_out = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards_generated.ini')
        conf = load_config(conf_file)

        data_file = os.path.join(ROOTPATH, 'STANDARDS', 'StandardsBig')
        conf.set('General', 'loglevel', 'INFO')
        conf.set('General', 'datafile', os.path.join(tempdir, 'proc'))
        conf.set('General', 'logfile', os.path.join(tempdir, 'log.log'))
        conf.set('ExportParticles', 'outputpath', os.path.join(tempdir, 'export'))
        conf.set('Background', 'num_images', '5')
        if MODEL_PATH is not None:
            conf.set('NNClassify', 'model_path', MODEL_PATH)
        with open(conf_file_out, 'w') as conf_file_hand:
            conf.write(conf_file_hand)
        settings = PySilcamSettings(conf)

        stats_file = os.path.join(tempdir, 'proc', 'StandardsBig-STATS.h5')

        # if STATS file already exists, it has to be deleted
        if (os.path.isfile(stats_file)):
            os.remove(stats_file)

        # start by processing 10 images. overwriteSTATS can be True or False here without affecting the result
        requested_images = 10
        silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False, nbImages=requested_images)
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
        num_images = count_images_in_stats(stats)
        assert num_images == requested_images, 'incorrect number of images processed'

        # restart, asking for 10 additional images starting from where we got to before (by using overwriteSTATS=False)
        silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False, nbImages=requested_images)
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
        num_images = count_images_in_stats(stats)
        assert num_images == requested_images * 2, 'incorrect number of images processed'

        # process the rest of the dataset starting from where we stopped
        silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False)
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
        num_images = count_images_in_stats(stats)
        total_images = len(glob.glob(os.path.join(data_file, "*.bmp")))
        assert num_images == (total_images - settings.Background.num_images), 'incorrect number of images processed'

        # attempt to process a fully-processed dataset
        silcam_process(conf_file_out, data_file, multiProcess=multiProcess, overwriteSTATS=False)
        stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
        num_images = count_images_in_stats(stats)
        assert num_images == (total_images - settings.Background.num_images), 'incorrect number of images processed'
