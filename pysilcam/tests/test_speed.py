import os
import unittest
import pandas as pd
import tempfile
from pysilcam.__main__ import silcam_process
from pysilcam.config import load_config

# Get user-defined path to unittest data folder
ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)

# Get user-defined tensorflow model path from environment variable
MODEL_PATH = os.environ.get('SILCAM_MODEL_PATH', None)

print('ROOTPATH', ROOTPATH)
print('MODEL_PATH', MODEL_PATH)


@unittest.skipIf((ROOTPATH is None),
                 "test path not accessible")
def test_speed():
    '''Testing that the speed without multiprocess'''

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
        conf_file_hand = open(conf_file_out, 'w')
        conf.write(conf_file_hand)
        conf_file_hand.close()

        t1 = pd.Timestamp.now()
        # call process function
        silcam_process(conf_file_out, data_file, multiProcess=False, nbImages=5)
        t2 = pd.Timestamp.now()
        td = t2 - t1
        assert td < pd.to_timedelta('00:00:45'), 'Processing time too long.'









