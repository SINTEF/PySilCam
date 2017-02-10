# -*- coding: utf-8 -*-
import pymba
import time


def _init_camera(vimba):
    '''Initialize the camera system'''

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
    camera0 = vimba.getCamera(cameraIds[0])
    camera0.openCamera()

    return camera0


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
    '''Aquire a single frame'''

    #Aquire single fram from camera
    camera.startCapture()
    frame0.queueFrameCapture()
    camera0.runFeatureCommand('AcquisitionStart')
    camera0.runFeatureCommand('AcquisitionStop')
    frame0.waitFrameCapture()
    
    #Copy frame data to numpy array
    moreUsefulImgData = np.ndarray(buffer = frame0.getBufferByteData(),
                                   dtype = np.uint8,
                                   shape = (frame0.height,
                                            frame0.width,
                                            1))
    cmaera.endCapture()

    return moreUsefulImgData


def print_camera_config(camera):
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
        camera = _configure_camera(cmaera)

        #Prepare for image aquisition and create a frame
        pymba.query_start()
        frame0 = camera.getFrame()
        frame0.announceFrame()

        count = 0
        while count < 20:
            img = _aquire(camera, frame0)
