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
def test_debug_files():
    '''Testing that the debug images are created'''

    # do this whole test using a temporary export directory
    with tempfile.TemporaryDirectory() as tempdir:
        print('tempdir:', tempdir, 'created')

        conf_file = os.path.join(ROOTPATH, 'config.ini')
        conf_file_out = os.path.join(ROOTPATH, 'config_generated.ini')
        conf = load_config(conf_file)

        data_file = os.path.join(ROOTPATH, 'STN04')
        conf.set('General', 'loglevel', 'DEBUG')
        conf.set('General', 'datafile', os.path.join(data_file, 'proc'))
        conf.set('General', 'logfile', os.path.join(ROOTPATH, 'log.log'))
        conf.set('ExportParticles', 'outputpath', tempdir)
        if MODEL_PATH is not None:
            conf.set('NNClassify', 'model_path', MODEL_PATH)
        with open(conf_file_out, 'w') as conf_file_hand:
            conf.write(conf_file_hand)

        num_test_ims = 5  # number of images to test

        # call process function
        silcam_process(conf_file_out, data_file, multiProcess=multiProcess, nbImages=num_test_ims)

        imc_files = glob.glob(os.path.join(tempdir, '*-IMC*'))
        assert len(imc_files) == num_test_ims, 'unexpected number of IMC files'

        seg_files = glob.glob(os.path.join(tempdir, '*-SEG*'))
        assert len(seg_files) == num_test_ims, 'unexpected number of SEG files'


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
    # todo generate this hdf filename based on input data
    hdf_file = os.path.join(data_file, 'export', 'D20170509T172705.387171.h5')
    report_figure = os.path.join(data_file, 'proc', 'STN04-Summary_all.png')

    # if csv file already exists, it has to be deleted
    if (os.path.isfile(stats_file)):
        os.remove(stats_file)

    # if hdf file file already exists, it has to be deleted
    if (os.path.isfile(hdf_file)):
        os.remove(hdf_file)

    # call process function
    silcam_process(conf_file_out, data_file, multiProcess=multiProcess)

    # check that csv file has been created
    assert os.path.isfile(stats_file), ('STATS csv file not created. should be here:' + stats_file)

    # check that csv file has been properly built
    stats = pd.read_hdf(stats_file, 'ParticleStats', 'stats')
    numline = stats.shape[0]
    assert numline > 1, 'stats empty'

    # check the columns
    path, filename = os.path.split(MODEL_PATH)
    header = pd.read_csv(os.path.join(path, 'header.tfl.txt'))
    class_labels = header.columns

    # construct expected column string
    column_string = ['major_axis_length',
                     'minor_axis_length',
                     'equivalent_diameter',
                     'solidity',
                     'minr',
                     'minc',
                     'maxr',
                     'maxc',
                     'export name',
                     'timestamp',
                     'saturation']
    for c in class_labels:
        column_string.append('probability_' + c)

    matching_elements = list(
            set(column_string) &
            set(stats.columns.tolist()))

    # check that output STATS file contains expected columns
    assert len(matching_elements) ==\
           len(stats.columns.tolist()) ==\
           len(column_string), 'output STATS file contains unexpected columns'

    # check the correct number of images have been processed
    settings = PySilcamSettings(conf_file_out)
    background_images = settings.Background.num_images
    number_processed = count_images_in_stats(stats)

    files = glob.glob(os.path.join(data_file, '*.bmp'))
    expected_processed = len(files) - background_images
    assert (number_processed == expected_processed), (str(number_processed) + ' images were processed. ' +
                                                      'Expected ' + str(expected_processed))

    # check that hdf file has been created
    assert os.path.isfile(hdf_file), ('hdf file not created. should be here:' + hdf_file)

    from pysilcam.postprocess import show_h5_meta
    show_h5_meta(hdf_file)
    from pysilcam.config import settings_from_h5
    Settings = settings_from_h5(hdf_file)
    # test a an appropriate setting after reading it back from the hdf5 file
    assert (Settings.ExportParticles.export_images is True), 'unexpected setting read from metadata in hdf5 file'

    # if report figure already exists, it has to be deleted
    if (os.path.isfile(report_figure)):
        os.remove(report_figure)

    silcam_report(stats_file, conf_file_out, dpi=10)
    assert os.path.isfile(report_figure), 'report figure file not created'
