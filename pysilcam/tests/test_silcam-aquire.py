# -*- coding: utf-8 -*-
import numpy as np
from pysilcam.acquisition import Acquire


def test_acquire_five_frames():
    '''Testing frame acquisition using fakepymba'''
    aq = Acquire()
    aqgen = aq.get_generator()

    # Check that we can generate frames
    prev_img = None
    for i, (timestamp, img) in enumerate(aqgen):
        # Check that frames (images) have non-zero size
        assert (img.size > 0)

        # Check that images are 3D (m x n x 3)
        assert (len(img.shape) == 3)

        # Check that we only get three channels
        assert (img.shape[2] == 3)

        # Check that we got different image data
        if prev_img is not None:
            assert (np.abs(prev_img - img).sum() > 1e-10)
        prev_img = img

        # Try five frames, then break
        if i > 5:
            break
