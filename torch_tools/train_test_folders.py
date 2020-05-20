import os
import shutil
import math
import random

test_frac = 0.1

source = '/Users/odin/Sintef/SilCam/pysilcam-testdata-newtorch/unittest-data/silcam_classification_database'
train_dir = '/Users/odin/Sintef/SilCam/pysilcam-testdata-newtorch/unittest-data/train'
test_dir = '/Users/odin/Sintef/SilCam/pysilcam-testdata-newtorch/unittest-data/test'

file_type = 'tiff'
exclude_copy = True

cats = os.listdir(source)


def make_train_test_dirs(cat, train_dir, test_dir):
    if not os.path.exists(os.path.join(train_dir, cat)):
        os.makedirs(os.path.join(train_dir, cat))
    if not os.path.exists(os.path.join(test_dir, cat)):
        os.makedirs(os.path.join(test_dir, cat))


for cat in cats:
    if cat[0] != '.':
        if os.path.exists(os.path.join(train_dir, cat)) or os.path.exists(os.path.join(test_dir, cat)):
            print('Skipping {} as a folder exists.'.format(cat))
            continue
        make_train_test_dirs(cat, train_dir, test_dir)
        files = os.listdir(os.path.join(source, cat))
        # Filter out files without correct extension
        files = [f for f in files if f.endswith(file_type)]
        if exclude_copy:
            files = [f for f in files if not f.endswith(' (copy).tiff')]

        random.shuffle(files)
        split = math.ceil(test_frac * len(files))
        test_files = files[:split]
        train_files = files[split:]

        for f in train_files:
            shutil.copy(os.path.join(source, cat, f), os.path.join(train_dir, cat))
        for f in test_files:
            shutil.copy(os.path.join(source, cat, f), os.path.join(test_dir, cat))

