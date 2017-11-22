# -*- coding: utf-8 -*-
import numpy as np
from pysilcam.acquisition import acquire
from pysilcam.background import backgrounder


def test_background_aquire():
    '''Testing background-corrected frame acquisition'''
    
    aqgen = acquire()

    bad_lighting_limit = 10000
    #Check that we can generate background-corredted frames. 
    #Use 5 frames for correction.
    prev_imgc = None
    for i, (timestamp, imgc) in  enumerate(backgrounder(5, aqgen, bad_lighting_limit)):
        #Check that frames (images) have non-zero size
        assert imgc.size > 0, 'Image size was not positive'

        #Check that images are 2D (m x n x 3)
        assert(len(imgc.shape) == 3)

        #Check that we only get one channel
        #Check that corrected image is uint8
        assert(imgc.dtype == np.uint8)

        #Check that we got different image data
        if prev_imgc is not None:
            assert(np.abs(prev_imgc - imgc).sum() > 1e-10)
        prev_imgc = imgc

        #Try five frames, then break
        if i>5:
            break
