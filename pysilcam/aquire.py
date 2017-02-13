# -*- coding: utf-8 -*-
import warnings
import time
import numpy as np

#Try import pymba, if not available, revert to in-package mockup
try:
    import pymba
except:
    warnings.warn('Pymba not available, using mocked version', ImportWarning)
    import pysilcam.pymba as pymba


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
        print('Camera ID:', cameraId)
    
    # get and open a camera
    camera = vimba.getCamera(cameraIds[0])
    camera.openCamera()

    return camera


def _configure_camera(camera, config=dict()):
    '''Configure the camera.
    
    Config is an optioinal dictionary of parameter-value pairs.
    '''

    #Default settings
    camera.AcquisitionFrameRateAbs = 15
    camera.ExposureTimeAbs = 300
    camera.PixelFormat = 'BayerRG8'
    camera.StrobeDuration = 600
    camera.StrobeDelay = 0
    camera.StrobeDurationMode = 'Controlled'
    camera.StrobeSource = 'Exposing'
    camera.SyncOutPolarity = 'Normal'
    camera.SyncOutSelector = 'SyncOut1'
    camera.SyncOutSource = 'Strobe1'

    #If a config is specified, override those values
    for k, v in config.items():
        setattr(camera, k, v)

    return camera


def _aquire_frame(camera, frame0):
    '''Aquire a single frame in Bayer format'''

    #Aquire single fram from camera
    camera.startCapture()
    frame0.queueFrameCapture()
    camera.runFeatureCommand('AcquisitionStart')
    camera.runFeatureCommand('AcquisitionStop')
    frame0.waitFrameCapture()
    
    #Copy frame data to numpy array (Bayer format)
    bayer_img = np.ndarray(buffer = frame0.getBufferByteData(),
                           dtype = np.uint8,
                           shape = (frame0.height, frame0.width, 1))
    camera.endCapture()

    return bayer_img


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


def aquire():
    '''Aquire images from SilCam'''

    with pymba.Vimba() as vimba:
        #Initialize the camera interface
        camera = _init_camera(vimba)

        #Configure camera
        camera = _configure_camera(camera)

        #Prepare for image aquisition and create a frame
        pymba.query_start()
        frame0 = camera.getFrame()
        frame0.announceFrame()

        #Aquire raw images and yield to calling context
        try:
            while True:
                img = _aquire_frame(camera, frame0)
                yield img
        finally:
            #Clean up after capture
            camera.revokeAllFrames()
    
            # close camera


def aquire_disk():
    '''Aquire images from SilCam and write them to disk.'''
    for count, img in enumerate(aquire()):
        filename = 'data/foo{0}.bmp'.format(count)
        imageio.imwrite(filename, img)
        print("Stored image image {0} to file {1}.".format(count, filename))
