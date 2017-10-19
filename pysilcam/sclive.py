from pysilcam.acquisition import acquire
import matplotlib.pyplot as plt
import numpy as np
import pygame
import subprocess
import os
import pysilcam.plotting as scplt
import pysilcam.postprocess as sc_pp
import matplotlib
matplotlib.use('Agg')
import matplotlib.backends.backend_agg as agg
import pylab
import pandas as pd


class liveview:

    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        wh = info.current_h-100
        wh = 600
        self.size = (int(wh / (2048/2448)), int(wh))
        icon = pygame.Surface((1, 1))
        icon.set_alpha(0)
        pygame.display.set_icon(icon)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('SilCam display')
        self.c = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20)
        self.record = False
        self.help = False
        self.disp = 1
        self.stats = []
        self.oil_stats = []
        self.gas_stats = []


    def overlay(self):
        r, g, b = 0, 0, 255


        # top
        txt = str(self.timestamp)
        label = self.font.render(txt, 1, (r, g, b))
        self.screen.blit(label, (0,0))

        txt = 'FPS:' + str(np.round(self.c.get_fps(), decimals=2))
        label = self.font.render(txt, 1, (r, g, b))
        self.screen.blit(label, (0,20))

        if self.help:
            txt = 'HELP: [h]  |  DISPLAY: [d]  |  RECORD[r]'
            label = self.font.render(txt, 1, (r, g, b))
            self.screen.blit(label, (0, self.size[1]/2))


        # bottom
        montxt = "df -h | grep DATA | awk '{{print $5}}'"
        prc = subprocess.Popen([montxt], shell=True, stdout=subprocess.PIPE)
        pcentfull = prc.stdout.read().decode('ascii').strip()
        label = self.font.render(str(pcentfull), 1, (r, g, b))

        txt = os.getcwd() + ' ' + str(pcentfull)
        label = self.font.render(txt, 1, (r, g, b))
        self.screen.blit(label, (0,self.size[1]-40))

        if self.record:
            txt = 'REC.[r]: ON'
        else:
            txt = 'REC.[r]: OFF'
        label = self.font.render(txt, 1, (r, g*np.invert(self.record), b))
        self.screen.blit(label, (0,self.size[1]-20))

        return self


    def dispkey(self):
        self.disp += 1
        if self.disp>2:
            self.disp=0

    def helpkey(self):
        self.help = np.invert(self.help)
        if self.help:
            self.disp = 2

    def update(self, img, timestamp, settings=None):
        self.c.tick()
        if self.disp==0:
            im = pygame.surfarray.make_surface(np.uint8(img))
            im = pygame.transform.flip(im, False, True)
            im = pygame.transform.rotate(im, -90)
            im = pygame.transform.scale(im, self.size)
            self.screen.blit(im, (0,0))
        elif np.logical_and((self.disp==1), (len(self.stats)>0)):
            fig = pylab.figure(figsize=[8, 8], dpi=100)
            ax = fig.gca()
            
            dias, vd = sc_pp.vd_from_stats(self.oil_stats, settings.PostProcess)
            plt.plot(dias, vd,'r')
            dias, vd = sc_pp.vd_from_stats(self.gas_stats, settings.PostProcess)
            plt.plot(dias, vd,'b')
             
            ax.set_xlim(10, 10000)
            ax.set_xscale('log')
            ax.set_xlabel('Equiv. diam (um)')
            ax.set_ylabel('Volume concentration (uL/L)')
            
            plt.title(str(np.min(self.stats['timestamp'])) + ' - ' + 
                    str(np.max(self.stats['timestamp'])))
            
            canvas = agg.FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            surf = pygame.image.fromstring(raw_data, size, "RGB")
            surf = pygame.transform.scale(surf, self.size)
            self.screen.blit(surf, (0,0))
            plt.close('all')
        elif self.disp==2:
            self.screen.fill((255, 255, 255))

        self.timestamp = timestamp
        
        for event in pygame.event.get():
            if event.type == 12:
                pygame.quit()
                subprocess.call('killall silcam', shell=True)
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.record = np.invert(self.record)
                if event.key == pygame.K_d:
                    self.dispkey()
                if event.key == pygame.K_h:
                    self.helpkey()

        self.overlay()

        pygame.display.flip()

        return self
