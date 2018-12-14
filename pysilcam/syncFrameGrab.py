from pymba import *
import time
import numpy as np
from pysilcam.config import load_camera_config
import skimage
import os
from cv2 import cvtColor, COLOR_BAYER_BG2RGB, imwrite
import cv2
import threading

def saveImg(img, i):
    path = '/home/bjarne/data/silcam_files/temp/'
    #e1 = cv2.getTickCount()
    img = cvtColor(img, COLOR_BAYER_BG2RGB)
    #e2 = cv2.getTickCount()

    #t = (e2 - e1)/cv2.getTickFrequency()
    #print(t)
    #print(str(i), ' Sum: ', img.sum(), 'shape: ', img.shape)
    np.save(path + str(i), img, allow_pickle=False)


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
        #print(k,'=',v)
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
    samples = 100
    imgArr = np.array([])

    frame0 = camera.getFrame()
    frame0.announceFrame()
    camera.startCapture()


    for i in range(samples):
        print('grabbing frame {}'.format(i))

        frame0.queueFrameCapture()
        camera.runFeatureCommand('AcquisitionStart')
        err = frame0.waitFrameCapture(timeout=500)
        if err == 0:
            img = np.ndarray(buffer=frame0.getBufferByteData(), dtype=np.uint8, shape=(frame0.height, frame0.width))
            #saveImg(img, i)
        else:
            print(err)
            camera.revokeAllFrames()
            frame0 = camera.getFrame()
            frame0.announceFrame()

        #t = threading.Thread(target=saveImg, args=(img.copy(), i))
        #t.daemon = True
        #t.start()


        #camera.revokeAllFrames()



    print('acquire frequency: ',samples/(time.time() - start))
    # clean up after capture
    camera.endCapture()


    #start = time.time()
    #for i in range(samples):


    # close camera
