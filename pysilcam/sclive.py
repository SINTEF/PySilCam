from pysilcam.acquisition import acquire
import matplotlib.pyplot as plt
import numpy as np
import pygame
import subprocess
import os

class liveview:

    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        wh = info.current_h-100
        wh = 600
        self.size = (int(wh / (2048/2448)), int(wh))
        print(self.size)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Raw image display')
        self.c = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20)
        self.record = False
        pass


    def overlay(self):
        # top
        txt = str(self.timestamp)
        label = self.font.render(txt, 1, (255, 255, 0))
        self.screen.blit(label, (0,0))

        txt = 'FPS:' + str(np.round(self.c.get_fps(), decimals=2))
        label = self.font.render(txt, 1, (255, 255, 0))
        self.screen.blit(label, (0,20))

        # bottom
        montxt = "df -h | grep DATA | awk '{{print $5}}'"
        prc = subprocess.Popen([montxt], shell=True, stdout=subprocess.PIPE)
        pcentfull = prc.stdout.read().decode('ascii').strip()
        label = self.font.render(str(pcentfull), 1, (255, 255,0))

        txt = os.getcwd() + ' ' + str(pcentfull)
        label = self.font.render(txt, 1, (255, 255,0))
        self.screen.blit(label, (0,self.size[1]-40))

        if self.record:
            txt = 'REC.[r]: ON'
        else:
            txt = 'REC.[r]: OFF'
        label = self.font.render(txt, 1, (255, 255*np.invert(self.record), 0))
        self.screen.blit(label, (0,self.size[1]-20))

        return self


    def update(self, img, timestamp):
        self.c.tick()
        im = pygame.surfarray.make_surface(np.uint8(img))
        im = pygame.transform.flip(im, False, True)
        im = pygame.transform.rotate(im, -90)
        im = pygame.transform.scale(im, self.size)
        self.screen.blit(im, (0,0))

        self.timestamp = timestamp
        
        for event in pygame.event.get():
            if event.type == 12:
                pygame.quit()
                subprocess.call('killall silcam', shell=True)
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.record = np.invert(self.record)

        self.overlay()

        pygame.display.flip()

        return self
