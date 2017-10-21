from pysilcam.acquisition import acquire
import numpy as np
import pygame
import subprocess
import os
import pysilcam.plotting as scplt
import pysilcam.postprocess as sc_pp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class liveview:

    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.fig.canvas.draw()

        blank = np.zeros((2448, 2048), dtype=np.uint8())
        #self.h = self.ax.plot([1,10])[0]
        self.h = self.ax.imshow(blank)

        #self.ax.set_xlim(10,10000)
        #self.ax.set_xscale('log')

        self.axbackground = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        self.oil_stats=[]

        plt.pause(0.000001)

    def update(self, im, settings):
        start_time = time.clock()
        dias, vd = sc_pp.vd_from_stats(self.gas_stats,
                settings.PostProcess)
        self.h.set_data(im)
        #self.ax.set_ylim(0,np.max(vd))

        self.ax.draw_artist(self.h)

        self.fig.canvas.blit(self.ax.bbox)
        plt.pause(0.00001)
        logger.info('Display took {0:.2f}s'.format(time.clock()-start_time))
