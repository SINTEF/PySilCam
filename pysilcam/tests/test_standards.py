# -*- coding: utf-8 -*-
import os
from pysilcam.__main__ import silcam_process
import pysilcam.postprocess as scpp
import unittest
import pandas as pd
from pysilcam.config import PySilcamSettings
from pysilcam.config import load_config
from sys import platform

# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

# pytest on windows can't deal with multiprocessing, so switch it off if windows patform detected
multiProcess = True
if platform == "win32":
    multiProcess = False

@unittest.skipIf((ROOTPATH is None),
    "test path not accessible")
def test_big_standards():
    '''Testing that the large standards are sized correctly'''

    conf_file = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards.ini')
    conf_file_out = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards_generated.ini')
    conf = load_config(conf_file)

    data_file = os.path.join(ROOTPATH, 'STANDARDS', 'StandardsBig')
    conf.set('General', 'loglevel', 'INFO')
    conf.set('General', 'datafile', os.path.join(ROOTPATH, 'STANDARDS', 'proc'))
    conf.set('General', 'logfile', os.path.join(ROOTPATH, 'STANDARDS', 'log.log'))
    if MODEL_PATH is not None:
        conf.set('NNClassify', 'model_path', MODEL_PATH)
    conf_file_hand = open(conf_file_out,'w')
    conf.write(conf_file_hand)
    conf_file_hand.close()

    stats_file = os.path.join(ROOTPATH, 'STANDARDS', 'proc', 'StandardsBig-STATS.csv')

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # call process function
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, nbImages=10)

    # check that csv file has been created
    assert os.path.isfile(stats_file), 'stats_file not created'

    # check that csv file has been properly built
    csvfile = open(stats_file)
    lines = csvfile.readlines()
    numline = len(lines)
    assert numline > 1 , 'csv file empty'

    settings = PySilcamSettings(conf_file_out)
    stats = pd.read_csv(stats_file)
    d50 = scpp.d50_from_stats(stats, settings.PostProcess)
    print('Large d50:', d50)
    assert (d50 > 310 and d50 < 330), 'incorrect d50'


@unittest.skipIf((ROOTPATH is None),
    "test path not accessible")
def test_small_standards():
    '''Testing that the small standards are sized correctly'''
    path = os.path.dirname(__file__)
    conf_file = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards.ini')
    conf_file_out = os.path.join(ROOTPATH, 'STANDARDS', 'config_glass_standards_generated.ini')
    conf = load_config(conf_file)

    data_file = os.path.join(ROOTPATH, 'STANDARDS', 'StandardsSmall')
    conf.set('General', 'datafile', os.path.join(ROOTPATH, 'STANDARDS', 'proc'))
    conf.set('General', 'logfile', os.path.join(ROOTPATH, 'STANDARDS', 'log.log'))
    if MODEL_PATH is not None:
        conf.set('NNClassify', 'model_path', MODEL_PATH)
    conf_file_hand = open(conf_file_out,'w')
    conf.write(conf_file_hand)
    conf_file_hand.close()

    stats_file = os.path.join(ROOTPATH, 'STANDARDS', 'proc', 'StandardsSmall-STATS.csv')

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # call process function
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess, nbImages=10)

    # check that csv file has been created
    assert os.path.isfile(stats_file), 'stats_file not created'

    # check that csv file has been properly built
    csvfile = open(stats_file)
    lines = csvfile.readlines()
    numline = len(lines)
    assert numline > 1 , 'csv file empty'

    settings = PySilcamSettings(conf_file_out)
    stats = pd.read_csv(stats_file)
    d50 = scpp.d50_from_stats(stats, settings.PostProcess)
    print('Small d50:', d50)
    assert (d50 > 70 and d50 < 90), 'incorrect d50'
