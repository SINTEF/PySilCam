# -*- coding: utf-8 -*-
from pysilcam.__main__ import silcam_process
from pysilcam.silcreport import silcam_report
from pysilcam.config import load_config
from pysilcam.postprocess import count_images_in_stats
from pysilcam.silcam_classify import load_model, predict
from pysilcam.config import PySilcamSettings
from skimage.io import imread
import glob
import os
import numpy as np
import unittest
import pandas as pd


# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

print('ROOTPATH',ROOTPATH)
print('MODEL_PATH',MODEL_PATH)

@unittest.skipIf((ROOTPATH is None),
    "test path not accessible")
def test_classify():
    '''
    Basic check of classification prediction against the training database.
    Therefore, if correct positive matches are not high percentages, then something is wrong with the prediction.

    @todo include more advanced testing of the classification feks. assert values in a confusion matrix.
    '''

    # location of the training data
    database_path = os.path.join(ROOTPATH, 'silcam_classification_database')

    # Load the trained tensorflow model and class names
    model, class_labels = load_model(MODEL_PATH)

    # class_labels should match the training data
    classes = glob.glob(os.path.join(database_path, '*'))

    # @todo write a quick check that classes and class_labels agree before doing the proper test.

    def correct_positives(category):
        '''
        calculate the percentage positive matches for a given category
        '''

        # list the files in this category of the training data
        files = glob.glob(os.path.join(database_path, category, '*.tiff'))

        # start a counter of incorrectly classified images
        failed = 0

        # loop through the database images
        for file in files:

            img = imread(file) # load ROI
            prediction = predict(img, model) # run prediction from silcam_classify

            ind = np.argmax(prediction) # find the highest score

            # check if the highest score matches the correct category
            if not class_labels[ind] == category:
                # if not, the add to the failure count
                failed += 1

        # turn failed count into a success percent
        success = 100 - (failed / len(files)) * 100
        return success

    # loop through each category and calculate the success percentage
    for cat in classes:
        name = os.path.split(cat)[-1]
        success = correct_positives(name)
        print(name, success)
        assert success > 96, (name + ' was poorly classified at only ' + str(success) + 'percent.')



@unittest.skipIf((ROOTPATH is None),
    "test path not accessible")
def test_output_files():
    '''Testing that the appropriate STATS.csv file is created'''

    conf_file = os.path.join(ROOTPATH, 'config.ini')
    conf_file_out = os.path.join(ROOTPATH, 'config_generated.ini')
    conf = load_config(conf_file)

    data_file = os.path.join(ROOTPATH, 'STN04')
    conf.set('General', 'datafile', os.path.join(data_file, 'proc'))
    conf.set('General', 'logfile', os.path.join(ROOTPATH,'log.log'))
    conf.set('ExportParticles', 'outputpath', os.path.join(data_file, 'export'))
    if MODEL_PATH is not None:
        conf.set('NNClassify', 'model_path', MODEL_PATH)
    conf_file_hand = open(conf_file_out,'w')
    conf.write(conf_file_hand)
    conf_file_hand.close()

    stats_file = os.path.join(data_file, 'proc', 'STN04-STATS.csv')
    hdf_file = os.path.join(data_file, 'export/D20170509T172705.387171.h5')
    report_figure = os.path.join(data_file,'proc', 'STN04-Summary_all.png')

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # if hdf file file already exists, it has to be deleted
    if (os.path.isfile(hdf_file)):
        os.remove(hdf_file)

    # call process function
    silcam_process(conf_file_out, data_file, multiProcess=False)

    # check that csv file has been created
    assert os.path.isfile(stats_file), ('STATS csv file not created. should be here:' + stats_file)

    # check that csv file has been properly built
    csvfile = open(stats_file)
    lines = csvfile.readlines()
    numline = len(lines)
    assert numline > 1 , 'csv file empty'

    # check the columns
    assert lines[0] == 'particle index,major_axis_length,minor_axis_length,equivalent_diameter,solidity,minr,minc,maxr,maxc,'\
            'probability_oil,probability_other,probability_bubble,probability_faecal_pellets,probability_copepod,'\
            'probability_diatom_chain,probability_oily_gas,export name,timestamp,saturation\n', 'columns not properly built'

    # check the correct number of images have been processed
    stats = pd.read_csv(stats_file)
    settings = PySilcamSettings(conf_file_out)
    background_images = settings.Background.num_images
    number_processed = count_images_in_stats(stats)

    files = glob.glob(os.path.join(data_file, '*.bmp'))
    expected_processed = len(files) - background_images
    # assert (number_processed == expected_processed), 'number of images processed does not match the size of the dataset'


    # check that hdf file has been created
    assert os.path.isfile(hdf_file), ('hdf file not created. should be here:' + hdf_file)

    from pysilcam.postprocess import show_h5_meta
    show_h5_meta(hdf_file)
    from pysilcam.config import settings_from_h5
    Settings = settings_from_h5(hdf_file)
    # test a an appropriate settting after reading it back from the hdf5 file
    assert (Settings.ExportParticles.export_images == True), 'unexpected setting read from metadata in hdf5 file'

    # if report figure already exists, it has to be deleted
    if (os.path.isfile(report_figure)):
        os.remove(report_figure)

    silcam_report(stats_file, conf_file_out, dpi=10)
    assert os.path.isfile(report_figure), 'report figure file not created'

    # # test synthesizer
    #import pysilcam.tests.synthesizer as synth
    #reportdir = os.path.join(path, '../../test-report')
    #os.makedirs(reportdir, exist_ok=True)
    #synth.generate_report(os.path.join(reportdir, 'imagesynth_report.pdf'), PIX_SIZE=28.758169934640524,
    #                      PATH_LENGTH=10, d50=800, TotalVolumeConcentration=800,
    #                      MinD=108)