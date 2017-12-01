# -*- coding: utf-8 -*-
import os
import warnings
import time
import numpy as np
import pandas as pd
import logging
import pysilcam.fakepymba as fakepymba
import sys

logger = logging.getLogger(__name__)

try:
    import pymba
except:
    logger.debug('Pymba not available. Cannot use camera')


def _init_camera(vimba):
    '''Initialize the camera system from vimba object'''

    # get system object
    system = vimba.getSystem()

    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)
    cameraIds = vimba.getCameraIds()
    for cameraId in cameraIds:
        logger.debug('Camera ID: {0}'.format(cameraId))

    #Check that we found a camera, if not, raise an error
    if len(cameraIds) == 0:
        logger.debug('No cameras detected')
        raise RuntimeError('No cameras detected!')
        camera = None
    else:
        # get and open a camera
        camera = vimba.getCamera(cameraIds[0])
        camera.openCamera()

    return camera


def _configure_camera(camera, config=dict()):
    '''Configure the camera.

    Config is an optioinal dictionary of parameter-value pairs.
    '''

    #Default settings
    camera.AcquisitionFrameRateAbs = 1
    camera.TriggerSource = 'FixedRate'
    camera.AcquisitionMode = 'SingleFrame'
    camera.ExposureTimeAbs = 30000
    #camera.PixelFormat = 'BayerRG8'
    camera.PixelFormat = 'RGB8Packed'
    camera.StrobeDuration = 600
    camera.StrobeDelay = 0
    camera.StrobeDurationMode = 'Controlled'
    camera.StrobeSource = 'FrameTriggerReady'
    camera.SyncOutPolarity = 'Normal'
    camera.SyncOutSelector = 'SyncOut1'
    camera.SyncOutSource = 'Strobe1'

    #camera.GVSPPacketSize = 9194
    camera.GVSPPacketSize = 1500


    #If a config is specified, override those values
    for k, v in config.items():
        setattr(camera, k, v)

    return camera


def print_camera_config(camera):
    '''Print the camera configuration'''
    config_info_map = {
        'AquisitionFrameRateAbs': 'Frame rate',
        'ExposureTimeAbs': 'Exposure time',
        'PixelFormat': 'PixelFormat',
        'StrobeDuration': 'StrobeDuration',
        'StrobeDelay': 'StrobeDelay',
        'StrobeDurationMode': 'StrobeDurationMode',
        'StrobeSource': 'StrobeSource',
        'SyncOutPolarity': 'SyncOutPolarity',
        'SyncOutSelector': 'SyncOutSelector',
        'SyncOutSource': 'SyncOutSource',
    }

    config_info = '\n'.join(['{0}: {1}'.format(a, camera.getattr(a))
                             for a, b in config_info_map])

    print(config_info)
    logger.debug(config_info)


class Acquire():
    def __init__(self, USE_PYMBA=False):
        if USE_PYMBA:
            self.pymba = pymba
            self.pymba.get_time_stamp = lambda x: pd.Timestamp.now()
            print('Pymba imported')
            self.get_generator = self.get_generator_camera
        else:
            self.pymba = fakepymba
            print('using fakepymba')
            self.get_generator = self.get_generator_disc

    def get_generator_disc(self, datapath=None, writeToDisk=False):
        '''
        Aquire images from disc
        
        Args:
            datapath: path from where the images are acquired.
            writeToDisk: this boolean is not used in this function, but the signature has 
                to be the same as get_generator_camera.

        Yields:
            timestamp: timestamp of the acquired image
            img: acquired image
        '''
        if datapath != None:
            os.environ['PYSILCAM_TESTDATA'] = datapath

        self.wait_for_camera()

        with self.pymba.Vimba() as vimba:
            camera = _init_camera(vimba)

            #Configure camera
            camera = _configure_camera(camera)

            #Prepare for image acquisition and create a frame
            frame0 = camera.getFrame()
            frame0.announceFrame()

            #Aquire raw images and yield to calling context
            while True:
                try:
                    timestamp, img = self._acquire_frame(camera, frame0)
                    yield timestamp, img
                except Exception:
                    frame0.img_idx += 1
                    if frame0.img_idx > len(frame0.files):
                        print('  END OF FILE LIST.')
                        logger.debug('  END OF FILE LIST.')
                        break


    def get_generator_camera(self, datapath=None, writeToDisk=False):
        '''
        Aquire images from Silcam
        
        Args:
            datapath: path from where the images are acquired.
            writeToDisk: boolean indicating wether the acquired 
                images have to be written to disc.

        Yields:
            timestamp: timestamp of the acquired image.
            img: acquired image.
        '''
        if datapath != None:
            os.environ['PYSILCAM_TESTDATA'] = datapath

        while True:
            try:
                #Wait until camera wakes up
                self.wait_for_camera()

                with self.pymba.Vimba() as vimba:
                    camera = _init_camera(vimba)

                    #Configure camera
                    camera = _configure_camera(camera)

                    #Prepare for image acquisition and create a frame
                    frame0 = camera.getFrame()
                    frame0.announceFrame()

                    #Aquire raw images and yield to calling context
                    while True:
                        timestamp, img = self._acquire_frame(camera, frame0)
                        if writeToDisk:
                            filename = os.path.join(datapath, timestamp.strftime('D%Y%m%dT%H%M%S.%f.silc'))
                            with open(filename, 'wb') as fh:
                                np.save(fh, img, allow_pickle=False)
                                fh.flush()
                                os.fsync(fh.fileno())
                                print('Written', filename)
                        yield timestamp, img
            except pymba.vimbaexception.VimbaException:
                print('Camera error. Restarting')
            except KeyboardInterrupt:
                print('User interrupt with ctrl+c, terminating PySilCam.')
                sys.exit(0)
           

    def _acquire_frame(self, camera, frame0):
        '''Aquire a single frame'''

        #Aquire single fram from camera
        camera.startCapture()
        frame0.queueFrameCapture()
        camera.runFeatureCommand('AcquisitionStart')
        camera.runFeatureCommand('AcquisitionStop')
        frame0.waitFrameCapture()

        #Copy frame data to numpy array (Bayer format)
        #bayer_img = np.ndarray(buffer = frame0.getBufferByteData(),
        #                       dtype = np.uint8,
        #                       shape = (frame0.height, frame0.width, 3))
        img = np.ndarray(buffer = frame0.getBufferByteData(),
                        dtype = np.uint8,
                        shape = (frame0.height, frame0.width, 3))

        timestamp = self.pymba.get_time_stamp(frame0)

        camera.endCapture()

        return timestamp, img.copy()

    def wait_for_camera(self):
        camera = None
        while not camera:
            with self.pymba.Vimba() as vimba:
                try:
                    camera = _init_camera(vimba)
                except RuntimeError:
                    msg = 'Could not connect to camera, sleeping five seconds and then retrying'
                    print(msg)
                    logger.warning(msg, exc_info=True)
                    time.sleep(5)

