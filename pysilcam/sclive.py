from pysilcam.acquisition import acquire
import matplotlib.pyplot as plt
import numpy as np
import pygame
import subprocess

class liveview:

    def __init__(self):
        pygame.init()
        info = pygame.display.Info()
        self.size = (int(info.current_h / (2048/2448))-100, info.current_h-100)
        self.size = (int(2448/3), int(2048/3))
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption('Raw image display')
        self.c = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 20)
        self.record = False
        pass

    def update(self, img, timestamp):
        self.c.tick()
        im = pygame.surfarray.make_surface(np.uint8(img))
        im = pygame.transform.flip(im, False, True)
        im = pygame.transform.rotate(im, -90)
        im = pygame.transform.scale(im, self.size)
        self.screen.blit(im, (0,0))
        
        txt = str(timestamp)
        label = self.font.render(txt, 1, (255, 255, 0))
        self.screen.blit(label, (0,0))

        txt = 'Display FPS:' + str(self.c.get_fps())
        label = self.font.render(txt, 1, (255, 255, 0))
        self.screen.blit(label, (0,20))

        for event in pygame.event.get():
            if event.type == 12:
                pygame.quit()
                subprocess.call('killall silcam', shell=True)
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self.record = np.invert(self.record)

        if self.record:
            txt = 'RECORD [r]: ON'
        else:
            txt = 'RECORD [r]: OFF'
        label = self.font.render(txt, 1, (255, 255*np.invert(self.record), 0))
        self.screen.blit(label, (0,40))

        pygame.display.flip()

        return self
