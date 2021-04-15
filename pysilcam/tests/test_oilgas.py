# -*- coding: utf-8 -*-
import os
import sys
import unittest
import tempfile
import pandas as pd
from pysilcam.__main__ import silcam_process
from pysilcam.config import load_config, PySilcamSettings
import pysilcam.oilgas as scog

# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

# pytest on windows can't deal with multiprocessing, so switch it off if windows patform detected
multiProcess = True
if sys.platform == "win32":
    multiProcess = False

print('ROOTPATH', ROOTPATH)
print('MODEL_PATH', MODEL_PATH)


@unittest.skipIf((ROOTPATH is None), "test path not accessible")
def test_output_files():
    '''
    Check that oil and gas ratios are correctly identified

    Note that there is additional logic to the ML classification
    so the test_z_classify.py is not sufficient.
    '''

    # Location of the oil gas test data
    database_path = os.path.join(ROOTPATH, 'silcam_oilgas_database')

    # Do this whole test using a temporary export directory
    with tempfile.TemporaryDirectory() as tempdir:
        print('tempdir:', tempdir, 'created')

        # Set up config files:
        conf_file = os.path.join(database_path, 'config_oilgas.ini')
        conf = load_config(conf_file)
        conf.set('General', 'loglevel', 'DEBUG')
        conf.set('ExportParticles', 'outputpath', tempdir)
        # window_size should cover the whole dataset
        conf.set('PostProcess', 'window_size', '60')

        if MODEL_PATH is not None:
            conf.set('NNClassify', 'model_path', MODEL_PATH)

        for i in ['oil', 'gas']:

            # Test data location
            data_file = os.path.join(database_path, i)

            # Generate config files
            conf_file_out = os.path.join(database_path, f'config_{i}_generated.ini')
            conf.set('General', 'datafile', os.path.join(data_file, 'proc'))
            conf.set('General', 'logfile', os.path.join(data_file, 'proc', 'log.log'))
            with open(conf_file_out, 'w') as conf_file_hand:
                conf.write(conf_file_hand)
            settings = PySilcamSettings(conf_file_out)

            # If STATS file already exists, it has to be deleted
            stats_file = os.path.join(data_file, 'proc', f'{i}-STATS.h5')
            if (os.path.isfile(stats_file)):
                os.remove(stats_file)

            # Call process function, load stats, check stats isn't empty
            silcam_process(conf_file_out, data_file, multiProcess=multiProcess)
            assert os.path.isfile(stats_file), ('STATS HDF file not created. should be here:' + stats_file)
            stats = pd.read_hdf(stats_file, 'ParticleStats/stats')
            assert stats.shape[0] > 1, 'stats empty'

            rts = scog.rt_stats(settings)
            rts.stats = stats
            rts.update()

            # In the assertions below, the GOR should be as given. The d50s and saturation
            # check that processing is consistent over time.
            if i == 'oil':
                assert rts.gor < 3, f"GOR for oil, {rts.gor}, too high."
                assert (445. < rts.oil_d50 < 485.), f"Oil d50, {rts.oil_d50}, outside of normal range."
                assert (12 < rts.saturation < 15), f"Saturation (oil), {rts.saturation}, outside of normal range."

            if i == 'gas':
                assert rts.gor > 97, f"GOR for gas, {rts.gor}, too low."
                assert (290. < rts.gas_d50 < 310.), f"Gas d50, {rts.gas_d50}, outside of normal range."
                assert (12.5 < rts.saturation < 15), f"Saturation (gas), {rts.saturation}, outside of normal range."
