import sys
import time
import datetime
import logging
from docopt import docopt
import numpy as np
from pysilcam import __version__
from pysilcam.acquisition import Acquire
from pysilcam.background import backgrounder
from pysilcam.process import processImage, statextract
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings, updatePathLength
import os
import pysilcam.silcam_classify as sccl
import multiprocessing
from multiprocessing.managers import BaseManager
from queue import LifoQueue
import psutil
from shutil import copyfile
import warnings
import pandas as pd
import time
from pysilcam.__main__ import *
import pygame
from skimage.io import imsave as imwrite

logger = logging.getLogger(__name__)


def get_image(datapath, config_filename):
    aqgen = liveview_acquire(datapath, config_filename, writeToDisk=False)
    while True:
        timestamp, im = next(aqgen)
        yield timestamp, im


def convert_image(im, size):
        im = pygame.surfarray.make_surface(np.uint8(im))
        im = pygame.transform.flip(im, False, True)
        im = pygame.transform.rotate(im, -90)
        im = pygame.transform.scale(im, size)
        return im


def liveview_acquire(datapath, config_filename, writeToDisk=False):
    '''Aquire images from the SilCam

    Args:
       datapath              (str)          :  Path to the image storage
       config_filename=None  (str)          :  Camera config file
       writeToDisk=True      (Bool)         :  True will enable writing of raw data to disc
                                               False will disable writing of raw data to disc
       gui=None          (Class object)     :  Queue used to pass information between process thread and GUI
                                               initialised in ProcThread within guicals.py
    '''

    #Load the configuration, create settings object
    settings = PySilcamSettings(config_filename)

    #Print configuration to screen
    print('---- CONFIGURATION ----\n')
    settings.config.write(sys.stdout)
    print('-----------------------\n')

    if (writeToDisk):
        # Copy config file
        configFile2Copy = datetime.datetime.now().strftime('D%Y%m%dT%H%M%S.%f') + os.path.basename(config_filename)
        copyfile(config_filename, os.path.join(datapath, configFile2Copy))

    configure_logger(settings.General)

    # update path_length
    updatePathLength(settings, logger)

    acq = Acquire(USE_PYMBA=True) # ini class
    t1 = time.time()

    aqgen = acq.get_generator(datapath, camera_config_file=config_filename, writeToDisk=writeToDisk)

    for i, (timestamp, imraw) in enumerate(aqgen):
        t2 = time.time()
        aq_freq = np.round(1.0/(t2 - t1), 1)
        requested_freq = settings.Camera.acquisitionframerateabs
        rest_time = (1 / requested_freq) - (1 / aq_freq)
        rest_time = np.max([rest_time, 0.])
        actual_aq_freq = 1/(1/aq_freq + rest_time)
        logger.info('Image {0} acquired at frequency {1:.1f} Hz'.format(i, actual_aq_freq))
        t1 = time.time()

        yield timestamp, imraw


def write_image(datapath, timestamp, imraw):
    filename = os.path.join(datapath, timestamp.strftime('D%Y%m%dT%H%M%S.%f.bmp'))
    imwrite(filename, np.uint8(imraw))
    logger.info('Image written')


def get_image_size(im):
    ims = np.shape(im)
    return ims


def zoomer(zoom):
    zoom += 1
    if zoom>2:
        zoom = 0
    return zoom


def liveview(datapath = '/mnt/DATA/emlynd/DATA/', config_filename = 'config_hardware_test.ini'):
    try:
        import pymba
    except:
        logger.info('Pymba not available. Cannot use camera')
        return

    aqgen = get_image(datapath, config_filename)
    timestamp, imraw = next(aqgen)
    ims = get_image_size(imraw)

    pygame.init()
    info = pygame.display.Info()
    size = (int(info.current_h / (ims[0] / ims[1])) - 50, info.current_h - 50)
    screen = pygame.display.set_mode(size)
    font = pygame.font.SysFont("monospace", 20)
    font_colour = (0, 127, 127)
    zoom = 0
    pause = False
    pygame.event.set_blocked(pygame.MOUSEMOTION)
    exit = False
    c = pygame.time.Clock()
    while not exit:
        if pause:
            event = pygame.event.wait()
            if event.type == 12:
                exit = True
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    zoom = zoomer(zoom)
                if event.key == pygame.K_SPACE:
                    write_image(datapath, timestamp, imraw)
                if event.key == pygame.K_p:
                    pause = np.invert(pause)
                if event.key == pygame.K_ESCAPE:
                    exit = True
                    continue
                else:
                    continue
                pygame.time.wait(100)

        timestamp, imraw = next(aqgen)

        if zoom>0:
            label = font.render('ZOOM [F]: ' + str(zoom), 1, font_colour)
            if zoom==1:
                imcrop = imraw[int(ims[0] / 4):-int(ims[0] / 4),
                               int(ims[1] / 4):-int(ims[1] / 4), :]
            else:
                imcrop = imraw[int(ims[0] / 2.5):-int(ims[0] / 2.5),
                               int(ims[1] / 2.5):-int(ims[1] / 2.5), :]
            im = convert_image(imcrop, size)
        else:
            im = convert_image(imraw, size)
            label = font.render('ZOOM [F]: OFF', 1, font_colour)

        screen.blit(im, (0, 0))
        screen.blit(label, (0, size[1] - 20))

        label = font.render('pause[p] write[space] exit[Esc]', 1, font_colour)
        screen.blit(label, (0, size[1] - 40))

        pygame.display.set_caption('Image display')
        label = font.render(str(timestamp) + '    Disp. FPS: ' +
                            str(c.get_fps()), 20, font_colour)
        screen.blit(label, (0, 0))
        label = font.render('Esc to exit',
                            1, font_colour)
        screen.blit(label, (0, 20))

        for event in pygame.event.get():
            if event.type == 12:
                pygame.quit()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    zoom = zoomer(zoom)
                if event.key == pygame.K_SPACE:
                    write_image(datapath, timestamp, imraw)
                if event.key == pygame.K_p:
                    pause = np.invert(pause)
                if event.key == pygame.K_ESCAPE:
                    exit = True
                    continue
        pygame.display.flip()
    pygame.quit()
