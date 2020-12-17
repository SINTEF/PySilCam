import pysilcam.tracking.track as sctr
from pysilcam.config import load_config, PySilcamSettings
import tempfile
import os
import unittest


ROOTPATH = os.environ.get('UNITTEST_DATA_PATH', None)


@unittest.skipIf((ROOTPATH is None),
                 "test path not accessible")
def test_track_process():

    conf_file = os.path.join(ROOTPATH, 'tracking/config_tracking.ini')
    datapath = os.path.join(ROOTPATH, 'tracking/Raw')

    # do this whole test using a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        conf = load_config(conf_file)
        conf.set('General', 'datafile', tempdir)
        conf.set('General', 'logfile', os.path.join(tempdir, 'log.log'))

        conf_file_out = os.path.join(tempdir, 'config_tracking.ini')
        with open(conf_file_out, 'w') as fh:
            conf.write(fh)

        sctr.track_process(conf_file_out, datapath)
        print('track_process completed')

        print('check output data')
        tracksfile = os.path.join(tempdir, 'Raw-TRACKS.h5')
        settings = PySilcamSettings(conf_file_out)

        data, tracks = sctr.load_and_process(tracksfile, settings.PostProcess.pix_size,
                                             track_length_limit=settings.Tracking.track_length_limit)

        assert len(data) == 517, 'incorrect number of detected particles'
        assert len(tracks) == 67, 'incorrect number of extracted tracks'

        sctr.make_boxplot(tracksfile, 'outputfigure')

        assert os.path.isfile('outputfigure.png'), 'outputfigure.png boxplot not found'
