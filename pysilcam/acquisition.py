# -*- coding: utf-8 -*-
import os
import psutil
import time
import numpy as np
import pandas as pd
import logging
from pysilcam.config import load_camera_config
import pysilcam.fakepymba as fakepymba
import sys
from queue import LifoQueue
from cv2 import cvtColor, COLOR_BAYER_BG2RGB, imwrite
from multiprocessing.managers import BaseManager


logger = logging.getLogger(__name__)

# setup the lifo queue for camera stream
# class MyManager(BaseManager):
#     '''
#     Customized manager class used to register LifoQueues
#     '''
#     pass
# manager = MyManager()
# manager.register('LifoQueue', LifoQueue)
# manager.start()
# imQueue = manager.LifoQueue(100) # make this large but not infinite (we set a limit when it is used later)
imQueue = LifoQueue(100)

isBayer = True


try:
    import pymba
except:
    logger.debug('Pymba not available. Cannot use camera')

def _init_camera(vimba):
    '''Initialize the camera system from vimba object
    Args:
        vimba (vimba object)  :  for example pymba.Vimba()
        
    Returns:
        camera       (Camera) : The camera without settings from the config
    '''

    # get system object
    system = vimba.getSystem()

    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)
    cameraIds = vimba.getCameraIds()
    for cameraId in cameraIds:
        logger.debug('Camera ID: {0}'.format(cameraId))

    # Check that we found a camera, if not, raise an error
    if len(cameraIds) == 0:
        logger.debug('No cameras detected')
        raise RuntimeError('No cameras detected!')
        camera = None
    else:
        # get and open a camera
        camera = vimba.getCamera(cameraIds[0])
        camera.openCamera()
    return camera


def _configure_camera(camera, config_file=None):
    '''Configure the camera.

    Args:
        camera       (Camera) : The camera with settings from the config
        config_file (str)     : Configuration file

    Returns:
        camera       (Camera) : The camera with settings from the config

    '''
    global isBayer

    # Read the configuration values from default config file
    defaultpath = os.path.dirname(os.path.abspath(__file__))
    defaultfile = os.path.join(defaultpath,'camera_config_RGBPacked.ini')
    config = load_camera_config(defaultfile)

    # Read the configuration values from users config file
    # The values found in this file, overrides those fro the default file
    # The rest keep the values from the defaults file
    config = load_camera_config(config_file, config)

    if config['PixelFormat'].upper().find('BAYERRG8') == 0:
        isBayer = True
    else:
        isBayer = False

    # If a config is specified, override those values
    for k, v in config.items():
        logger.info('{0} = {1}'.format(k,v))
        setattr(camera, k, v)

    return camera

def _frame_done_callback(frame):
    frame.queueFrameCapture(frameCallback=_frame_done_callback)

    if isBayer:
        img = np.ndarray(buffer=frame.getBufferByteData(), dtype=np.uint8, shape=(frame.height, frame.width))
    else:
        img = np.ndarray(buffer=frame.getBufferByteData(), dtype=np.uint8, shape=(frame.height, frame.width, 3))
    timestamp = pymba.get_time_stamp(frame)
    try:
        if frame.writeToDisk:
            filename = os.path.join(frame.datapath, timestamp.strftime('D%Y%m%dT%H%M%S.%f.bmp'))
            logger.info('Writing:' + filename)
            # write images to disc here before anything else gets in the way
            imwrite(filename, img)
        if imQueue.qsize()<2: # now we limit this queue so it is very small, and just keep acquiring
            # this queue must never reach the queue size, otherwise it will block!
            imQueue.put_nowait([timestamp, img])
    except:
        logger.warning("dropping frame!")


def _start_acqusition(camera, datapath, writeToDisk):
    # acquiring images is the most imporant job for this computer
    try:
        pid = psutil.Process(os.getpid())
        if (sys.platform == 'linux'):
            pid.nice(20)
        else:
            pid.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except:
        logger.warning('Could not prioritise acquisition process!')

    camera.startCapture()

    frame = camera.getFrame()
    frame.announceFrame()
    # add some info about where to save things to disc within the frame class
    # this is used by _frame_done_callback when it writes to disc
    frame.datapath = datapath
    frame.writeToDisk = writeToDisk
    frame.queueFrameCapture(frameCallback=_frame_done_callback)

    camera.runFeatureCommand('AcquisitionStart')

# def _stopAcqusition(camera):
    # TODO implement this!
    # We should do a more gracefull shutdown of the camera when terminating pySilcam.

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

    logger.debug(config_info)


class Acquire():
    '''
    Class used to acquire images from camera or disc
    '''
    def __init__(self, USE_PYMBA=False):
        if USE_PYMBA:
            self.pymba = pymba
            self.pymba.get_time_stamp = lambda x: pd.Timestamp.now()
            logger.info('Pymba imported')
            self.get_generator = self.get_generator_camera
        else:
            self.pymba = fakepymba
            logger.info('using fakepymba')
            self.get_generator = self.get_generator_disc

    def get_generator_disc(self, datapath=None, writeToDisk=False, camera_config_file=None):
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

            # Configure camera
            camera = _configure_camera(camera, config_file=camera_config_file)

            # Prepare for image acquisition and create a frame
            frame0 = camera.getFrame()
            frame0.announceFrame()

            # Aquire raw images and yield to calling context
            while True:
                try:
                    timestamp, img = self._acquire_frame(camera, frame0)
                    yield timestamp, img
                except Exception:
                    frame0.img_idx += 1
                    if frame0.img_idx > len(frame0.files):
                        logger.info('  END OF FILE LIST.')
                        break


    def get_generator_camera(self, datapath=None, writeToDisk=False, camera_config_file=None):
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
                # Wait until camera wakes up
                self.wait_for_camera()

                with self.pymba.Vimba() as vimba:
                    camera = _init_camera(vimba)

                    # Configure camera
                    camera = _configure_camera(camera, camera_config_file)

                    # Start stream
                    _start_acqusition(camera, datapath, writeToDisk)

                    # Acquire raw images and yield to calling context
                    while True:
                        timestamp, img = imQueue.get()
                        logger.info('%s acquired', timestamp.strftime('D%Y%m%dT%H%M%S.%f.bmp'))
                        if isBayer:
                            img = cvtColor(img, COLOR_BAYER_BG2RGB)
                        yield timestamp, img
            except pymba.vimbaexception.VimbaException as e:
                logger.warning('Camera error: %s, restarting...', e.message)
            except IOError as e:
                logger.error('I/O Error: %s', e.message)
                sys.exit(0)
            except KeyboardInterrupt:
                logger.info('User interrupt with ctrl+c, terminating PySilCam.')
                sys.exit(0)
           

    def _acquire_frame(self, camera, frame0):
        '''Aquire a single frame
        Args:
            camera (Camera)         : The camera with settings from the config
                                      obtained from _configure_camera()
            frame0  (frame)         : camera frame obtained from camera.getFrame()

        Returns:
            timestamp (timestamp)   : timestamp of image acquisition
            output (uint8)          : raw image acquired
        '''

        # Acquire single frame from camera
        camera.startCapture()
        frame0.queueFrameCapture()
        camera.runFeatureCommand('AcquisitionStart')
        camera.runFeatureCommand('AcquisitionStop')
        frame0.waitFrameCapture()

        img = np.ndarray(buffer=frame0.getBufferByteData(),
                         dtype=np.uint8,
                         shape=(frame0.height, frame0.width, 3))

        timestamp = self.pymba.get_time_stamp(frame0)

        camera.endCapture()

        output = img.copy()

        return timestamp, output

    def wait_for_camera(self):
        '''
        Waiting function that will continue forever until a camera becomes connected
        '''
        camera = None
        while not camera:
            with self.pymba.Vimba() as vimba:
                try:
                    camera = _init_camera(vimba)
                except RuntimeError:
                    logger.warning('Could not connect to camera, sleeping five seconds and then retrying',
                                   exc_info=True)
                    time.sleep(5)

if __name__ == "__main__":
    pass