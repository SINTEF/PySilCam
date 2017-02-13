# -*- coding: utf-8 -*-
'''
Partial drop-in replacement for Pymba, for testing purposes.
'''
import numpy as np
import time

#Handle potential Python 2.7 and Python 3
try:
    import __builtin__
except ImportError:
    import builtins as __builtin__

#Fake aqusition frequency
FPS = 15

def print(*args, **kwargs):
        __builtin__.print('FakePymba: ', *args, **kwargs)

def query_start():
    print('Starting query')

class Camera:
    def openCamera(self):
        print('Opening camera')

    def startCapture(self):
        print('Starting capture')
        time.sleep(1.0/FPS)

    def runFeatureCommand(self, cmd):
        print('Camera command: {0}'.format(cmd))

    def endCapture(self):
        print('Ending camera capture')

    def getFrame(self):
        time.sleep(1.0/FPS)
        return Frame()

    def revokeAllFrames(self):
        print('\nCleaning up: revoking all frames')
        pass

class Frame:
    def __init__(self):
        self.frame = np.zeros((5, 4))
        self.height = 5
        self.width = 4
        print('Frame aquired')

    def getBufferByteData(self):
        return self.frame.tobytes()

    def announceFrame(self):
        print('Frame aquired')

    def getBufferByteData(self):
        return self.frame.tobytes()

    def announceFrame(self):
        print('Announcing frame')

    def queueFrameCapture(self):
        print('Queuing frame capture')

    def waitFrameCapture(self):
        print('Waiting for frame capture')

class System:
    def __init__(self):
        self.GeVTLIsPresent = False

class Vimba:
    def getSystem(self):
        return System()

    def getCameraIds(self):
        return [0]

    def getCamera(self, camera_id):
        return Camera()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
