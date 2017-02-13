# -*- coding: utf-8 -*-
import numpy as np
from subprocess import check_output
import pysilcam.acquisition


def test_echo():
    '''An example test.'''
    result = run_cmd("echo hello world")
    assert result == "hello world\n"


def run_cmd(cmd):
    '''Run a shell command `cmd` and return its output.'''
    return check_output(cmd, shell=True).decode('utf-8')


def test_acquire_five_frames():
    '''Testing frame acquisition'''

    #Check that we can generate frames
    prev_img = None
    for i, img in  enumerate(pysilcam.acquisition.acquire()):
        #Check that frames (images) have non-zero size
        assert(img.size > 0)

        #Check that images are 3D (m x n x 3)
        assert(len(img.shape) == 3)

        #Check that we only get three channels
        assert(img.shape[2] == 3)

        #Check that we got different image data
        if prev_img is not None:
            assert(np.abs(prev_img - img).sum() > 1e-10)
        prev_img = img

        #Try five frames, then break
        if i>5:
            break
