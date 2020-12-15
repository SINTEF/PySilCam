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
from glob import glob

#Handle potential Python 2.7 and Python 3
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__


def silcam_load(filename):
    #Load the raw image from disc depending on filetype
    if filename.endswith('.silc'):
        img0 = np.load(filename, allow_pickle=False)
    elif filename.endswith('.silc_mono'):
        # this is a quick fix to load mono 8 bit images
        img_mono = np.load(filename, allow_pickle=False)
        r, c = np.shape(img_mono)
        img0 = np.zeros([r, c, 3], dtype=np.uint8)
        img0[:, :, 0] = img_mono
        img0[:, :, 1] = img_mono
        img0[:, :, 2] = img_mono
    else:
        img0 = imageio.imread(filename)
    return img0

def silcam_name2time(fname):
    timestamp = pd.to_datetime(os.path.splitext(fname)[0][1:])
    return timestamp

#Fake aqusition frequency
FPS = 5

#Module-wide logger
logger = logging.getLogger(__name__)

def get_time_stamp(frame):
    return frame.timestamp

def query_start():
    logger.debug('Starting query')

class Camera:
    def openCamera(self):
        logger.debug('Opening camera')

    def startCapture(self):
        logger.debug('Starting capture')
        #time.sleep(1.0/FPS)

    def runFeatureCommand(self, cmd):
        logger.debug('Camera command: {0}'.format(cmd))

    def endCapture(self):
        logger.debug('Ending camera capture')

    def getFrame(self):
        #time.sleep(1.0/FPS)
        if 'REALTIME_DISC' in os.environ.keys():
            return Frame()
        elif 'PYSILCAM_REALTIME_DATA' in os.environ.keys():
            return RealtimeFrame()
        else:
            return Frame()

    def revokeAllFrames(self):
        logger.debug('\nCleaning up: revoking all frames')
        pass


class Frame:
    def __init__(self):
        #If the environment variable PYSILCAM_TESTDATA is defined, read images
        #from that location.
        if 'PYSILCAM_TESTDATA' in os.environ.keys():
            # offset = int(os.environ.get('PYSILCAM_OFFSET', 0))
            offset = getattr(Frame, 'PYSILCAM_OFFSET', 0)
            path = os.environ['PYSILCAM_TESTDATA']
            path = path.replace('\ ',' ') # handle spaces (not sure on windows behaviour)
            self.path = path
            self.timestamp = None
            if 'REALTIME_DISC' in os.environ.keys():
                print('get initial file list')
                self.files = sorted(glob(os.path.join(self.path, '*.silc')),
                                    reverse=True)
                print(len(self.files), 'files.')
                while len(self.files)==0:
                    print('waiting for data')
                    time.sleep(1)
                    self.files = sorted(glob(os.path.join(self.path, '*.silc')),
                                        reverse=True)
                    print(self.files)
            else:
                self.files = [os.path.join(path, f)
                              for f in sorted(os.listdir(path))
                              if f.endswith('.silc')][offset:]

                if len(self.files)==0:
                    self.files = [os.path.join(path, f)
                                  for f in sorted(os.listdir(path))
                                  if f.startswith('D') and (f.endswith('.bmp'))][offset:]

            self.img_idx = 0

            img0 = silcam_load(self.files[0])

            if len(img0.shape) == 2:
                self.height, self.width = img0.shape
            else:
                self.height, self.width, _ = img0.shape

        else:
            self.files = None
            self.width = 2448
            self.height = 2050

        logger.debug('Frame acquired')

    def getBufferByteData(self):
        if 'REALTIME_DISC' in os.environ.keys():
            num_old_files = len(self.files)
            while len(self.files) == num_old_files:
                print('checking for new data')
                self.files = sorted(glob(os.path.join(self.path, '*.silc')),
                                    reverse=True)
                time.sleep(1)
            print(' new data found')

            self.img_idx = 0

        if self.files is not None:
            frame = silcam_load(self.files[self.img_idx])
            if len(frame.shape) == 2:
                frame2 = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
                for i in range(3):
                    frame2[:, :, i] = frame[:, :]
                frame = frame2
            fname = os.path.basename(self.files[self.img_idx])
            self.timestamp = silcam_name2time(fname)
            self.img_idx += 1
            logger.debug('Getting buffer byte data from file {0}, {1}/{2}'.format(frame.shape, self.img_idx, len(self.files)))
        else:
            self.timestamp = datetime.now()
            frame = np.zeros((self.height, self.width, 3),
                             dtype=np.uint8()) + np.random.randint(0,255)
            #logger.debug('Getting buffer byte data, {0}'.format(frame.shape))
            #time.sleep(1.0/FPS)
        return frame.tobytes()

    def announceFrame(self):
        logger.debug('Frame acquired')

    def announceFrame(self):
        logger.debug('Announcing frame')

    def queueFrameCapture(self):
        logger.debug('Queuing frame capture')

    def waitFrameCapture(self):
        logger.debug('Waiting for frame capture')


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
        img0 = silcam_load(self.filename)
        self.height = img0.shape[0]
        self.width = img0.shape[1]
        logger.debug('Realtime frame acquired')

    def getBufferByteData(self):
        self._list_images()
        while self.files[-3] == self.filename:
            logger.debug('No new images ({0}), waiting 1s and then retrying'.format(len(self.files)))
            time.sleep(1)
            self._list_images()

        self.filename = self.files[-3]
        frame = silcam_load(self.filename)
        if len(frame.shape) == 2:
            frame2 = np.zeros((self.height, self.width, 3), dtype=frame.dtype)
            for i in range(3):
                frame2[:, :, i] = frame[:, :]
            frame = frame2
        fname = os.path.basename(self.filename)
        self.timestamp = silcam_name2time(fname)
        self.img_idx += 1
        logger.debug('Getting buffer byte data from file {0}, #{1}'.format(frame.shape, self.img_idx))
        return frame.tobytes()


class System:
    def __init__(self):
        logger.debug('System initialize')
        self.GeVTLIsPresent = False

class Vimba:
    def getSystem(self):
        logger.debug('Getting Vimba system')
        return System()

    def getCameraIds(self):
        logger.debug('Getting camera IDs')
        return [0]

    def getCamera(self, camera_id):
        logger.debug('Getting camera: {0}'.format(camera_id))
        return Camera()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
