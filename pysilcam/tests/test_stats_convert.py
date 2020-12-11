from pysilcam.postprocess import statscsv_to_statshdf
import pandas as pd
import tempfile
import os
import numpy as np


def test_stats_convert():
    stats_csv_file = r'https://raw.githubusercontent.com/wiki/SINTEF/PySilCam/data/STN07-STATS.csv'
    original_stats = pd.read_csv(stats_csv_file, index_col=False)
    print(original_stats.columns)
    original_stats_shape = np.shape(original_stats)

    with tempfile.TemporaryDirectory() as tempdir:
        print('tempdir:', tempdir, 'created')
        local_csv_filename = 'STN07-STATS.csv'
        original_stats.to_csv(local_csv_filename, index=False)
        statscsv_to_statshdf(local_csv_filename)
        local_hdf_filename = local_csv_filename.replace('-STATS.csv', '-STATS.h5')
        print(local_hdf_filename)
        assert os.path.isfile(local_hdf_filename), ('STATS.h5 file not created.')

        new_stats = pd.read_hdf(local_hdf_filename, 'ParticleStats/stats')
        print(new_stats.columns)
        new_stats_shape = np.shape(new_stats)
        print('original_stats_shape', original_stats_shape)
        print('new_stats_shape', new_stats_shape)
        assert (original_stats_shape == new_stats_shape), ('mismatch in shape between csv and h5 stats')

        assert (original_stats.columns.all() == new_stats.columns.all()),\
            ('mismatch in columns between csv and h5 stats')
