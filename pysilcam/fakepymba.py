# -*- coding: utf-8 -*-
'''
Partial drop-in replacement for Pymba, for testing purposes.
'''
from __future__ import print_function
import os
import sys
import time
from datetime import datetime
import numpy as np
import imageio
import logging
import pandas as pd

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

def get_time_stamp(frame):
    return frame.timestamp

def query_start():
    print('Starting query')
    logging.debug('Starting query')

class Camera:
    def openCamera(self):
        print('Opening camera')
        logging.debug('Opening camera')

    def startCapture(self):
        print('Starting capture')
        logging.debug('Starting capture')
        #time.sleep(1.0/FPS)

    def runFeatureCommand(self, cmd):
        print('Camera command: {0}'.format(cmd))
        logging.debug('Camera command: {0}'.format(cmd))

    def endCapture(self):
        print('Ending camera capture')
        logging.debug('Ending camera capture')

    def getFrame(self):
        #time.sleep(1.0/FPS)
        if 'PYSILCAM_REALTIME_DATA' in os.environ.keys():
            return RealtimeFrame()
        else:
            return Frame()

    def revokeAllFrames(self):
        print('\nCleaning up: revoking all frames')
        logging.debug('\nCleaning up: revoking all frames')
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
        logging.debug('Frame acquired')

    def getBufferByteData(self):
        if self.files is not None:
            frame = imageio.imread(self.files[self.img_idx])
            if len(frame.shape) == 2:
                frame2 = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
                for i in range(3):
                    frame2[:, :, i] = frame[:, :]
                frame = frame2
            fname = os.path.basename(self.files[self.img_idx])
            self.timestamp = pd.to_datetime(fname[1:-4])
            self.img_idx += 1
            print('Getting buffer byte data from file {0}, {1}/{2}'.format(frame.shape, self.img_idx, len(self.files)))
            logging.debug('Getting buffer byte data from file {0}, {1}/{2}'.format(frame.shape, self.img_idx, len(self.files)))
        else:
            self.timestamp = datetime.now()
            frame = np.random.random((self.height, self.width, 3))
            print('Getting buffer byte data, {0}'.format(frame.shape))
            logging.debug('Getting buffer byte data, {0}'.format(frame.shape))
            time.sleep(1.0/FPS)
        return frame.tobytes()

    def announceFrame(self):
        print('Frame acquired')
        logging.debug('Frame acquired')

    def announceFrame(self):
        print('Announcing frame')
        logging.debug('Announcing frame')

    def queueFrameCapture(self):
        print('Queuing frame capture')
        logging.debug('Queuing frame capture')

    def waitFrameCapture(self):
        print('Waiting for frame capture')
        logging.debug('Waiting for frame capture')


class RealtimeFrame(Frame):
    '''For faster real-time processing from disk'''

    def _list_images(self):
        self.files = [os.path.join(self.path, f) 
                      for f in sorted(os.listdir(self.path)) 
                      if f.endswith('.bmp')]
 
    def __init__(self): 
        #Read files from this location
        self.path = os.environ['PYSILCAM_REALTIME_DATA']
        self._list_images()
        self.img_idx = 0
        self.filename = self.files[0]
        img0 = imageio.imread(self.filename)
        self.height = img0.shape[0]
        self.width = img0.shape[1]
        print('Realtime frame acquired')
        logging.debug('Realtime frame acquired')

    def getBufferByteData(self):
        self._list_images()
        while self.files[-3] == self.filename:
            print('No new images ({0}), waiting 1s and then retrying'.format(len(self.files)))
            logging.debug('No new images ({0}), waiting 1s and then retrying'.format(len(self.files)))
            time.sleep(1)
            self._list_images()

        self.filename = self.files[-3]
        frame = imageio.imread(self.filename)
        if len(frame.shape) == 2:
            frame2 = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
            for i in range(3):
                frame2[:, :, i] = frame[:, :]
            frame = frame2
        fname = os.path.basename(self.filename)
        self.timestamp = pd.to_datetime(fname[1:-4])
        self.img_idx += 1
        print('Getting buffer byte data from file {0}, #{1}'.format(frame.shape, self.img_idx))
        logging.debug('Getting buffer byte data from file {0}, #{1}'.format(frame.shape, self.img_idx))
        return frame.tobytes()


class System:
    def __init__(self):
        print('System initialize')
        logging.debug('System initialize')
        self.GeVTLIsPresent = False

class Vimba:
    def getSystem(self):
        print('Getting Vimba system')
        logging.debug('Getting Vimba system')
        return System()

    def getCameraIds(self):
        print('Getting camera IDs')
        logging.debug('Getting camera IDs')
        return [0]

    def getCamera(self, camera_id):
        print('Getting camera: {0}'.format(camera_id))
        logging.debug('Getting camera: {0}'.format(camera_id))
        return Camera()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
