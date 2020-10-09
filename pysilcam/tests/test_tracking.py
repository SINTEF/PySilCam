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
        print('* track_process completed')

        tracksfile = os.path.join(tempdir, 'Raw-TRACKS.h5')

        print('* checking post-processing of ', tracksfile)

        settings = PySilcamSettings(conf_file_out)

        data, tracks = sctr.load_and_process(tracksfile, settings.PostProcess.pix_size,
                                             track_length_limit=settings.Tracking.track_length_limit)

        print('    data', len(data))
        print('    tracks', len(tracks))

        print('* testing boxplotting from tracked data')
        sctr.make_boxplot(tracksfile)
        print('  boxplotting done')

        print('* test_track_process complete')
