import pysilcam.__main__ as psc
from multiprocessing import Process, Queue
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
import os
from pysilcam.config import load_config, PySilcamSettings
import pysilcam.oilgas as scog


class ProcThread(Process):

    def __init__(self, datadir):
        super(ProcThread, self).__init__()
        self.q = Queue()
        self.info = 'ini done'
        self.datadir = datadir
        self.configfile = ''
        self.settings = ''



    def run(self):
        psc.silcam_process(self.configfile, self.datadir, gui=self.q)
        #psc.silcam_sim(self.datadir, self.q)


    def go(self):
        self.rts = scog.rt_stats(self.settings)
        self.start()
        self.info = 'go sent'


    def stop_silcam(self):
        if self.is_alive():
            self.terminate()
            self.info = 'termination sent'
        else:
            self.info = 'nothing to terminate'


    def plot(self):
        infostr = 'waiting to plot'
        if self.is_alive():
            try:
                stats = self.q.get(timeout=0.1)

                try:
                    rts.stats = rts.stats().append(stats_all)
                except:
                    rts.stats = rts.stats.append(stats_all)
                rts.update()


                #stats = pd.DataFrame.from_dict(data)
                #print(data.head())

                #dias, vd = sc_pp.vd_from_stats(stats,
                #    settings.PostProcess)

                imc = data['imc']
                timestamp = data['timestamp']
                dias = data['dias']
                vd_oil = data['vd_oil']
                vd_gas = data['vd_gas']

                ttlstr = ('image time: ' +
                    str(timestamp))
                infostr = data['infostr']
            except:
                infostr = 'failed to get data from process'
                return

            plt.clf()
            plt.cla()

            plt.subplot(1,2,1)
            plt.cla()
            plt.imshow(imc)
            plt.axis('off')
            plt.title(ttlstr)

            plt.subplot(1,2,2)
            plt.cla()
            plt.plot(dias, vd_oil ,'r')
            plt.plot(dias, vd_gas ,'b')
            plt.xscale('log')
            plt.xlim((10, 10000))
            plt.ylabel('Volume Concentration [uL/L]')
            plt.xlabel('Diamter [um]')

            plt.tight_layout()

        while not self.q.empty():
            self.q.get_nowait()

        self.info = infostr



    def silcview(self):
        self.raw()
        files = [os.path.join(self.datadir, f) for f in
                sorted(os.listdir(self.datadir))
                if f.endswith('.bmp')]
        import pygame
        import time
        pygame.init()
        info = pygame.display.Info()
        size = (int(info.current_h / (2048/2448))-100, info.current_h-100)
        screen = pygame.display.set_mode(size)
        font = pygame.font.SysFont("monospace", 20)
        c = pygame.time.Clock()
        zoom = False
        counter = -1
        direction = 1 # 1=forward 2=backward
        pause = False
        pygame.event.set_blocked(pygame.MOUSEMOTION)
        while True:
            if pause:
                event = pygame.event.wait()
                if event.type == 12:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        zoom = np.invert(zoom)
                    if event.key == pygame.K_LEFT:
                        direction = -1
                    if event.key == pygame.K_RIGHT:
                        direction = 1
                    if event.key == pygame.K_p:
                        pause = np.invert(pause)
                    else:
                        continue
                    pygame.time.wait(100)

            counter += direction
            counter = np.max([counter, 0])
            counter = np.min([len(files)-1, counter])
            c.tick(15) # restrict to 15Hz
            f = files[counter]

            if not (counter == 0 | counter==len(files)-1):
                im = pygame.image.load(os.path.join(self.datadir, f)).convert()

            if zoom:
                label = font.render('ZOOM [F]: ON', 1, (255, 255, 0))
                im = pygame.transform.scale2x(im)
                screen.blit(im,(-size[0]/2,-size[1]/2))
            else:
               im = pygame.transform.scale(im, size)
               screen.blit(im,(0,0))
               label = font.render('ZOOM [F]: OFF', 1, (255, 255, 0))
            screen.blit(label,(0, size[1]-20))

            if direction==1:
                dirtxt = '>>'
            elif direction==-1:
                dirtxt = '<<'
            if pause:
                dirtxt = 'PAUSED ' + dirtxt
            label = font.render('DIRECTION [<-|->|p]: ' + dirtxt, 1, (255,255,0))
            screen.blit(label, (0, size[1]-40))

            if counter == 0:
                label = font.render('FIRST IMAGE', 1, (255,255,0))
                screen.blit(label, (0, size[1]-60))
            elif counter == len(files)-1:
                label = font.render('LAST IMAGE', 1, (255,255,0))
                screen.blit(label, (0, size[1]-60))


            timestamp = pd.to_datetime(
                    os.path.splitext(os.path.split(f)[-1])[0][1:])


            pygame.display.set_caption('raw image replay:' + os.path.split(f)[0])#, icontitle=None)
            label = font.render(str(timestamp), 20, (255, 255, 0))
            screen.blit(label,(0,0))
            label = font.render('Display FPS:' + str(c.get_fps()),
                    1, (255, 255, 0))
            screen.blit(label,(0,20))

            for event in pygame.event.get():
                if event.type == 12:
                    pygame.quit()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        zoom = np.invert(zoom)
                    if event.key == pygame.K_LEFT:
                        direction = -1
                    if event.key == pygame.K_RIGHT:
                        direction = 1
                    if event.key == pygame.K_p:
                        pause = np.invert(pause)
                        direction = 0



            pygame.display.flip()

        pygame.quit()

        print(self.datadir)


    def load_settings(self, configfile):
        conf = load_config(configfile)
        self.settings = PySilcamSettings(conf)

