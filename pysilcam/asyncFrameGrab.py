from pymba import *
import time
import queue
import numpy as np
from pysilcam.config import load_camera_config
from cv2 import cvtColor, COLOR_BAYER_BG2RGB

def saveImg(img, i):
    path = '/home/bjarne/data/silcam_files/temp/'
    img = cvtColor(img, COLOR_BAYER_BG2RGB)
    np.save(path + str(1), img, allow_pickle=False)

    print('Frequency: ', 1/(time.time() - saveImg.t))
    saveImg.t = time.time()
saveImg.t = time.time()

def frameDone_callback(frame):
    frame.queueFrameCapture(frameCallback=frameDone_callback)
    img = np.ndarray(buffer=frame.getBufferByteData(), dtype=np.uint8, shape=(frame.height, frame.width))
    imQueue.put(img)

imQueue = queue.LifoQueue(10)

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
    config = load_camera_config('', config)

    #If a config is specified, override those values
    for k, v in config.items():
        #print(k,'=',v)
        setattr(camera, k, v)

    camera.startCapture()

    frame0 = camera.getFrame()
    frame0.announceFrame()

    frame0.queueFrameCapture(frameCallback=frameDone_callback)
    camera.runFeatureCommand('AcquisitionStart')


    while(1):
        img = imQueue.get()
        print(imQueue.qsize())
        saveImg(img, 1)
