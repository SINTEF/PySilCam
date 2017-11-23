# -*- coding: utf-8 -*-
import os
import sys
import logging
from pysilcam.__main__ import silcam_process
import pysilcam.postprocess as scpp
import unittest
import pandas as pd
import pysilcam.silcam_classify as sccl
from pysilcam.config import PySilcamSettings
import tensorflow as tf

@unittest.skipIf(not os.path.isdir(
    '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsBig'),
    "test path not accessible")
@unittest.skipIf(not os.path.isdir(
    '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsSmall'),
    "test path not accessible")

def test_big_standards():
    '''Testing that the large standards are sized correctly'''

    path = os.path.dirname(__file__)
    conf_file = os.path.join(path, 'config_glass_standards.ini')

    data_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsBig'
    stats_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/proc/StandardsBig-STATS.csv'

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # call process function
    silcam_process(conf_file, data_file, nbImages=10)

    # check that csv file has been created
    assert os.path.isfile(stats_file), 'stats_file not created'

    # check that csv file has been properly built
    csvfile = open(stats_file)
    lines = csvfile.readlines()
    numline = len(lines)
    assert numline > 1 , 'csv file empty'

    # check the columns
    assert lines[0] == 'particle index,major_axis_length,minor_axis_length,equivalent_diameter,solidity,minr,minc,maxr,maxc,'\
            'probability_oil,probability_other,probability_bubble,probability_faecal_pellets,probability_copepod,'\
            'probability_diatom_chain,probability_oily_gas,export name,timestamp\n', 'columns not properly built'

    settings = PySilcamSettings(conf_file)
    stats = pd.read_csv(stats_file)
    d50 = scpp.d50_from_stats(stats, settings.PostProcess)
    assert (d50 > 310 and d50 < 330), 'incorrect d50'


@unittest.skipIf(not os.path.isdir(
    '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsBig'),
    "test path not accessible")
@unittest.skipIf(not os.path.isdir(
    '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsSmall'),
    "test path not accessible")
def test_small_standards():
    '''Testing that the small standards are sized correctly'''

    path = os.path.dirname(__file__)
    conf_file = os.path.join(path, 'config_glass_standards.ini')

    data_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/StandardsSmall'
    stats_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STANDARDS/proc/StandardsSmall-STATS.csv'

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # call process function
    silcam_process(conf_file, data_file, nbImages=10)

    # check that csv file has been created
    assert os.path.isfile(stats_file), 'stats_file not created'

    # check that csv file has been properly built
    csvfile = open(stats_file)
    lines = csvfile.readlines()
    numline = len(lines)
    assert numline > 1 , 'csv file empty'

    # check the columns
    assert lines[0] == 'particle index,major_axis_length,minor_axis_length,equivalent_diameter,solidity,minr,minc,maxr,maxc,'\
            'probability_oil,probability_other,probability_bubble,probability_faecal_pellets,probability_copepod,'\
            'probability_diatom_chain,probability_oily_gas,export name,timestamp\n', 'columns not properly built'

    settings = PySilcamSettings(conf_file)
    stats = pd.read_csv(stats_file)
    d50 = scpp.d50_from_stats(stats, settings.PostProcess)
    assert (d50 > 70 and d50 < 90), 'incorrect d50'
