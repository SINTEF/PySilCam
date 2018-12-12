from pymba import *
import time
import numpy as np
from pysilcam.config import load_camera_config
import skimage
import os
import cv2
import threading


def saveImg(img, i):
    path = '/home/bjarne/data/silcam_files/temp/'
    #cv2.imwrite(path + str(i) + 'bayer.bmp', img)
    #print('bayer: Sum: ', img.sum(), 'shape: ', img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
    print(str(i), ' Sum: ', img.sum(), 'shape: ', img.shape)
    cv2.imwrite(path + str(i) + 'bgr.bmp', img)

# start Vimba
with Vimba() as vimba:

    # INTI

    # get system object
    system = vimba.getSystem()

    # list available cameras (after enabling discovery for GigE cameras)
    if system.GeVTLIsPresent:
        system.runFeatureCommand("GeVDiscoveryAllOnce")
        time.sleep(0.2)
    cameraIds = vimba.getCameraIds()
    for cameraId in cameraIds:
        print(cameraId)

    # get and open a camera
    camera = vimba.getCamera(cameraIds[0])
    camera.openCamera()

    #INIT Done
    #config

    # Read the configiration values from default config file
    configPath = 'camera_config_defaults.ini'
    config = load_camera_config(configPath)

    # Read the configiration values from users config file
    # The values found in this file, overrides those fro the default file
    # The rest keep the values from the defaults file
    config = load_camera_config('', config)

    #If a config is specified, override those values
    for k, v in config.items():
        print(k,'=',v)
        setattr(camera, k, v)

    # list camera features
    #cameraFeatureNames = camera0.getFeatureNames()
    #for name in cameraFeatureNames:
    #    print(name)

    # get the value of a feature
    #print(camera0.AcquisitionMode)

    # set the value of a feature
    #camera0.AcquisitionMode = 'SingleFrame'

    start = time.time()
    samples = 500
    imgArr = np.array([])

    threadingSave = True

    for i in range(samples):
        print('grabbing frame {}'.format(i))
        frame0 = camera.getFrame()
        frame0.announceFrame()
        #frame0.waitFrameCapture()


        camera.startCapture()
        frame0.queueFrameCapture()
        camera.runFeatureCommand('AcquisitionStart')
        camera.runFeatureCommand('AcquisitionStop')
        frame0.waitFrameCapture()

        img = np.ndarray(buffer=frame0.getBufferByteData(), dtype=np.uint8, shape=(frame0.height, frame0.width))
        #img = np.ndarray(buffer=frame0.getBufferByteData(), dtype=np.uint8, shape=(frame0.height, frame0.width, 3))
        #img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)
        #imgArr = np.append(imgArr, img.copy())


        if threadingSave:
            t = threading.Thread(target=saveImg, args=(img.copy(),i))
            t.daemon = True
            t.start()
        else:
            saveImg(img.copy(), i)

        # clean up after capture
        camera.endCapture()
        camera.revokeAllFrames()

    print('acquire frequency: ',samples/(time.time() - start))

    #start = time.time()
    #for i in range(samples):


    # close camera
