# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import pandas as pd
import logging
from pysilcam.config import load_camera_config
import pysilcam.fakepymba as fakepymba
import sys

logger = logging.getLogger(__name__)

try:
    from vimba import Vimba, FrameStatus
except:
    logger.debug('VimbaPython not available. Cannot use camera')


def _init_camera(vimba):
    '''Initialize the camera system from vimba object
    Args:
        vimba (vimba object)  :  for example Vimba.get_instance()
        
    Returns:
        camera       (Camera) : The camera without settings from the config
    '''

    # find available cameras
    cameraIds = vimba.get_all_cameras()

    for cameraId in cameraIds:
        logger.debug('Camera ID: {0}'.format(cameraId))

    #Check that we found a camera, if not, raise an error
    if not cameraIds:
        logger.debug('No cameras detected')
        raise RuntimeError('No cameras detected!')
        camera = None
    else:
        # get and open the first camera on the list (multiple cameras should not be connected!)
        camera = cameraIds[0]

    return camera


def _configure_camera(camera, config_file=None):
    '''Configure the camera.

    Args:
        camera       (Camera) : The camera with settings from the config
        config_file (str)     : Configuration file

    Returns:
        camera       (Camera) : The camera with settings from the config

    '''

    # Read the configiration values from default config file
    defaultpath = os.path.dirname(os.path.abspath(__file__))
    defaultfile = os.path.join(defaultpath,'camera_config_defaults.ini')
    config = load_camera_config(defaultfile)

    # Read the configiration values from users config file
    # The values found in this file, overrides those fro the default file
    # The rest keep the values from the defaults file
    config = load_camera_config(config_file, config)

    #If a config is specified, override those values
    with camera:
        for k, v in config.items():
            print(k, v)
            logger.info('{0} = {1}'.format(k,v))
            # try to write settings to the camera
            try:
                getattr(camera, k).set(v)
            except AttributeError: # if there is an element of the camera config that is not compatible, then continue to the next
                continue

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

    logger.info(config_info) # TODO: Should this be printed?
    logger.debug(config_info)


class Acquire():
    '''
    Class used to acquire images from camera or disc
    '''
    def __init__(self, USE_PYMBA=False, datapath=None, writeToDisk=False,
                 FAKE_PYMBA_OFFSET=0, gui=None):
        if USE_PYMBA:
            self.vimba = Vimba
            logger.info('Vimba imported')
            self.gui = gui

            if datapath != None:
                os.environ['PYSILCAM_TESTDATA'] = datapath

            self.datapath = datapath
            self.writeToDisk = writeToDisk
        else:
            fakepymba.Frame.PYSILCAM_OFFSET = FAKE_PYMBA_OFFSET
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

            #Configure camera
            camera = _configure_camera(camera, config_file=camera_config_file)

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
                        #print('  END OF FILE LIST.')
                        logger.info('  END OF FILE LIST.')
                        break

    def image_handler(self, camera, frame):
        if frame.get_status() == FrameStatus.Complete:
            # get image
            img = frame.as_numpy_ndarray()

            timestamp = pd.Timestamp.now()
            filename = os.path.join(self.datapath, timestamp.strftime('D%Y%m%dT%H%M%S.%f.silc'))

            # if write to disc
            # write to disc
            if self.writeToDisk:
                with open(filename, 'wb') as fh:
                    np.save(fh, img, allow_pickle=False)
                    fh.flush()
                    os.fsync(fh.filenp())


            # previously we calculated acquisition frequency here

            # gui data handling
            # if self.gui is not None:
            #    while (self.gui.qsize() > 0):
            #        try:
            #            self.gui.get_nowait()
            #            time.sleep(0.001)
            #        except:
            #            continue
            #    # try:
            #    rtdict = dict()
            #    rtdict = {'dias': 0,
            #             'vd_oil': 0,
            #             'vd_gas': 0,
            #             'gor': np.nan,
            #             'oil_d50': 0,
            #             'gas_d50': 0,
            #             'saturation': 0}
            #   self.gui.put_nowait((timestamp, imraw, imraw, rtdict))

        camera.queue_frame(frame)  # ask the camera for the next frame, which would evtentually call image_handle again


    def stream_from_camera(self, camera_config_file=None):
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



        while True:
            try:
                # Wait until camera wakes up
                self.wait_for_camera()

                with self.vimba.get_instance() as vimba:
                    camera = _init_camera(vimba)

                    # Configure camera
                    camera = _configure_camera(camera, camera_config_file)

                    camera.start_streaming(handler=self.image_handler, buffer_count=10)

            except KeyboardInterrupt:
                logger.info('User interrupt with ctrl+c, terminating PySilCam.')
                camera.stop_streaming()
                sys.exit(0)
            except:
                logger.info('Camera error. Restarting')
                camera.stop_streaming() # restart setup here - don't stop!
                continue
                
    def _acquire_frame(self, camera, frame):
        '''Aquire a single frame while streaming
        
        requires camera.start_streaming(handler=_acquire_frame, buffer_count=10)
        
        '''
        print('_acquire_frame')
        if frame.get_status() == FrameStatus.Complete:
            self.timestamp = pd.Timestamp.now()
            self.image = frame.as_numpy_ndarray()
            cam.queue_frame(frame)


    def wait_for_camera(self):
        '''
        Waiting function that will continue forever until a camera becomes connected
        '''
        camera = None
        while not camera:
            with Vimba.get_instance() as vimba:
                try:
                    camera = _init_camera(vimba)
                except RuntimeError:
                    msg = 'Could not connect to camera, sleeping five seconds and then retrying'
                    print(msg) # TODO: WHy is there a print here? warning should write to sys.stderr anyway
                    logger.warning(msg, exc_info=True)
                    time.sleep(5)

