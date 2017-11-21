# -*- coding: utf-8 -*-
import os
import sys
import logging
from pysilcam.__main__ import silcam_process
import unittest

@unittest.skipIf(not os.path.isdir(
    '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STN04'), 
    "test path not accessible")
@unittest.skipIf(not os.path.isfile(
    'config.ini'), 
    "config file not accessible")
def test_csv_file():
    '''Testing that the appropriate STATS.csv file is created'''

    data_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STN04'
    conf_file = 'config.ini'
    stats_file = '//sintef.no/mk20/nasgul/Miljoteknologi/MK102013220_SILCAM_IPR_EJD/hello_silcam/unittest_entries/STN04-STATS.csv'
    
    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # call process function
    silcam_process(conf_file, data_file)

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

    nbimages = 0 # number of images in the csv file
    for line in lines:
        if line[0] == '0': # index of particule
            nbimages += 1
    assert nbimages == 5, 'images missing from csv file' # 5 images are used for the background, the 5 images left are processed
    
    
