# -*- coding: utf-8 -*-
'''
Partial drop-in replacement for Pymba, for testing purposes.
'''
from __future__ import print_function
import os
import sys
import time
import numpy as np
import imageio
import logging

#Handle potential Python 2.7 and Python 3
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

#Fake aqusition frequency
FPS = 15

#Module-wide logger
#logger = logging.getLogger(__name__.replace('pymba', 'fakepymba'))
logger = logging.getLogger(__name__)
def print(*args, **kwargs):
    #__builtin__.print('FakePymba: ', *args, **kwargs)
    logger.debug(*args, **kwargs)


def query_start():
    print('Starting query')

class Camera:
    def openCamera(self):
        print('Opening camera')

    def startCapture(self):
        print('Starting capture')
        #time.sleep(1.0/FPS)

    def runFeatureCommand(self, cmd):
        print('Camera command: {0}'.format(cmd))

    def endCapture(self):
        print('Ending camera capture')

    def getFrame(self):
        #time.sleep(1.0/FPS)
        return Frame()

    def revokeAllFrames(self):
        print('\nCleaning up: revoking all frames')
        pass


class Frame:
    def __init__(self):
        #If the environment variable PYSILCAM_TESTDATA is defined, read images
        #from that location.
        if 'PYSILCAM_TESTDATA' in os.environ.keys():
            offset = int(os.environ.get('PYSILCAM_OFFSET', 0))
            path = os.environ['PYSILCAM_TESTDATA']
            self.files = [os.path.join(path, f) 
                          for f in sorted(os.listdir(path)) 
                          if f.endswith('.bmp')][offset:]
            self.img_idx = 0
            img0 = imageio.imread(self.files[0])
            if len(img0.shape) == 2:
                self.height, self.width = img0.shape
            else:
                self.height, self.width, _ = img0.shape
        else:
            self.files = None
            self.width = 800
            self.height = 600
        print('Frame acquired')

    def getBufferByteData(self):
        if self.files is not None:
            frame = imageio.imread(self.files[self.img_idx])
            if len(frame.shape) == 2:
                frame2 = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
                for i in range(3):
                    frame2[:, :, i] = frame[:, :]
                frame = frame2
            self.img_idx += 1
            print('Getting buffer byte data from file {0}, {1}/{2}'.format(frame.shape, 
                                                                           self.img_idx, len(self.files)))
        else:
            frame = np.random.random((self.height, self.width, 3))
            print('Getting buffer byte data, {0}'.format(frame.shape))
            time.sleep(1.0/FPS)
        return frame.tobytes()

    def announceFrame(self):
        print('Frame acquired')

    def announceFrame(self):
        print('Announcing frame')

    def queueFrameCapture(self):
        print('Queuing frame capture')

    def waitFrameCapture(self):
        print('Waiting for frame capture')

class System:
    def __init__(self):
        print('System initialize')
        self.GeVTLIsPresent = False

class Vimba:
    def getSystem(self):
        print('Getting Vimba system')
        return System()

    def getCameraIds(self):
        print('Getting camera IDs')
        return [0]

    def getCamera(self, camera_id):
        print('Getting camera: {0}'.format(camera_id))
        return Camera()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
