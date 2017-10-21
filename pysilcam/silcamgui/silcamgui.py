import sys
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import skimage.io
from pysilcam.silcamgui.SilCam import Ui_SilCam
from pysilcam.silcamgui.SilCamController import Ui_SilCamController
import seaborn as sns
import pysilcam.postprocess as scpp
import pysilcam.plotting as scplt
import pysilcam.datalogger as scdl
import pysilcam.oilgas as scog
from pysilcam.config import load_config, PySilcamSettings
sns.set_style('ticks')
sns.set_context(font_scale=2)
import cmocean
import subprocess


DATADIR = os.getcwd()
#DATADIR = '/mnt/DATA/'

def names_to_times(names):
    times = []
    for n in names:
        name = os.path.split(n)[1]
        times.append(pd.to_datetime(name[1:-4]))
    return times


def times_to_hz(times):
    hz = []
    for i in range(len(times)-1):
        dt = times[i+1] - times[i]
        dt = dt / np.timedelta64(1, 's')
        hz.append(1/dt)
    return hz

class controller(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_SilCamController()
        self.ui.setupUi(self)

def main():
    app = QApplication(sys.argv)

    class StartQT5(QMainWindow):
        def __init__(self, parent=None):
            QMainWindow.__init__(self, parent)
            self.ui = Ui_SilCam()
            self.ui.setupUi(self)

            # --- some default states
            self.datadir = DATADIR
            self.settings = ''
            self.stats = []
            self.lv_raw_toggle = False
            self.monitor_toggle = False
            self.lvwaitseconds = 1

            # ---- figure in middle
            f = plt.figure()
            self.canvas = FigureCanvas(f)
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            self.ui.centralwidget.setLayout(layout)
            path_here = os.path.realpath(__file__)
            imfile = os.path.join(os.path.split(path_here)[0], 'ojleutslipp.jpg')
            im = skimage.io.imread(imfile)
            plt.imshow(im)
            plt.axis('off')
            self.canvas.draw()
            # ----

            # ---- define some callbacks
            self.ui.actionExit.triggered.connect(self.exit)
            self.ui.actionRaw.triggered.connect(self.raw)
            self.ui.actionSilc_viewer.triggered.connect(self.silcview)
            self.ui.actionLoadProcessed.triggered.connect(self.load_processed)
            self.ui.actionVD_Time_series.triggered.connect(self.time_series_vd)
            self.ui.actionExport_Time_series.triggered.connect(self.export_series_vd)
            self.ui.actionController.triggered.connect(self.acquire_controller)
            self.ui.actionSave_Figure.triggered.connect(self.save_figure)

            self.infolabel = QtWidgets.QTextEdit(self.ui.dockWidgetContents)
            self.infolabel.setReadOnly(True)
            self.infolabel.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
            ilayout = QVBoxLayout()
            ilayout.addWidget(self.infolabel)
            self.ui.dockWidgetContents.setLayout(ilayout)

            self.infstr = 'Hei!!\n'
            self.infolabel.setText(self.infstr)

            self.layout = layout
            app.processEvents()

            #self.acquire_controller()


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


        def save_figure(self):
            savename = QFileDialog.getSaveFileName(self,
                caption = 'Save plots',
                directory = os.path.join(self.datadir, '../'),
                filter = (('*.png'))
                )
            if savename == '':
                return

            savename = savename[0]
            print(savename)
            self.status_update('saving to ' + savename)
            app.processEvents()
            plt.savefig(savename, bbox_inches='tight', dpi=600)
            self.status_update('  done.')


        def acquire_controller(self):
            self.ctrl = controller(self)
            self.ctrl.ui.pb_live_raw.clicked.connect(self.lv_raw_switch)
            self.ctrl.ui.pb_DriveMonitor.clicked.connect(self.monitor_switch)
            self.ctrl.ui.pb_start.clicked.connect(self.record)
            self.ctrl.ui.pb_stop.clicked.connect(self.stop_record)
            self.status_update('opening acquisition controller')
            self.lv_raw_check()
            self.monitor_check()
            self.ctrl.show()
            self.monitor_switch()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))


        def monitor_switch(self):
            self.monitor_toggle = np.invert(self.monitor_toggle)
            if self.monitor_toggle:
                self.status_update('monitoring ' + self.datadir)
                self.status_update(' ' + self.datadir)
                self.monitor()
            else:
                self.status_update(' drive monitor disabled')

        def lv_raw_switch(self):
            self.lv_raw_toggle = np.invert(self.lv_raw_toggle)
            if self.lv_raw_toggle:
                self.status_update(' Live view enabled')
                self.lv_raw()
            else:
                self.status_update(' Live view disabled')


        def monitor(self):
            self.monitor_check()

            if not self.monitor_toggle:
                return

            files = [os.path.join(self.datadir, f)
                    for f in sorted(os.listdir(self.datadir)) if
                    f.endswith('.bmp')]
            nimages = str(len(files))
            if len(files) > 3:

                name1 = os.path.split(files[-3])[1]
                ts1 = pd.to_datetime(name1[1:-4])
                name2 = os.path.split(files[-2])[1]
                ts2 = pd.to_datetime(name2[1:-4])
                td = ts2 - ts1
                td = td / np.timedelta64(1, 's')
                hz = 1 / td
                hz = str(np.around(hz, decimals=2))

                last_image = pd.to_datetime(pd.datetime.now()) - ts2
                last_image = last_image / np.timedelta64(1, 's')
                last_image = str(np.around(last_image, decimals=1))

            else:
                hz = 'waiting for data'
                last_image = ' waiting for data'


            #montxt = "df -h | grep " + self.datadir + " | awk '{{print $5}}'"
            montxt = "df -h | grep DATA | awk '{{print $5}}'"
            prc = subprocess.Popen([montxt], shell=True, stdout=subprocess.PIPE)
            pcentfull = prc.stdout.read().decode('ascii').strip()


            ttlstr = (self.datadir + ' :  ' + nimages + ' images; ' +
                pcentfull + ' full; ' + hz + 'Hz; ' +
                last_image + ' sec. since prev.')

            self.status_update(ttlstr, grow=False)
            QtCore.QTimer.singleShot(1*1000, self.monitor)


        def monitor_check(self):
            if self.monitor_toggle:
                self.ctrl.ui.pb_DriveMonitor.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(0,150,0) }'))
            else:
                self.ctrl.ui.pb_DriveMonitor.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(150,150,255) }'))


        def lv_raw(self):
            self.lv_raw_check()
            if not self.lv_raw_toggle:
                return

            try:
                files = [os.path.join(self.datadir, f) for f in
                        sorted(os.listdir(self.datadir)) if f.endswith('.bmp') ]
                imfile = files[-2]

                name = os.path.split(imfile)[1]
                timestamp = pd.to_datetime(name[1:-4])

                #im = skimage.io.imread(imfile)
                with open(imfile, 'rb') as fh:
                    im = np.load(imfile, allow_pickle=False)
                plt.clf()
                plt.cla()
                plt.imshow(im)
                plt.title('Time now: ' + str(pd.datetime.now()) + '\n'
                        'Time acquired: ' + str(timestamp)
                        )
                plt.axis('off')
                plt.tight_layout()
                self.canvas.draw()
            except ValueError:
                plt.title('error loading image')
                self.canvas.draw()

            QtCore.QTimer.singleShot(self.lvwaitseconds*1000, self.lv_raw)


        def lv_raw_check(self):
            if self.lv_raw_toggle:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(0,150,0) }'))
            else:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(150,150,255) }'))


        def status_update(self, string, grow=True):
            if not grow:
                self.infstr = self.infstr[:self.infstr.rfind('\n')]
                self.infstr = self.infstr[:self.infstr.rfind('\n')]
                self.infstr += '\n'

            self.infstr += string + '\n'
            self.infolabel.setText(self.infstr)
            sb = self.infolabel.verticalScrollBar()
            sb.setValue(sb.maximum())
            app.processEvents()


        def load_sc_config(self):
            self.configfile = QFileDialog.getOpenFileName(self,
                    caption = 'Load config ini file',
                    directory = self.datadir,
                    filter = (('*.ini'))
                    )[0]
            if self.configfile == '':
                return
            conf = load_config(self.configfile)
            self.settings = PySilcamSettings(conf)


        def export_series_vd(self):
            if len(self.stats) == 0 :
                self.load_processed()

            stats = self.stats
            settings = self.settings

            savename = QFileDialog.getSaveFileName(self,
                caption = 'Save VD Time-Series CSV',
                directory = os.path.join(self.datadir, '../'),
                filter = (('.csv'))
                )
            if savename == '':
                return

            savename = savename[0] + savename[1]
            datafile = scdl.DataLogger(savename, scog.ogdataheader())

            u = stats['timestamp'].unique()

            self.status_update('exporting to' + savename)
            self.status_update(' ')
            for i, s in enumerate(u):
                substats = stats[stats['timestamp']==s]

                ts = pd.to_datetime(substats['timestamp'])
                ts = ts.iloc[0]
                data_all = scog.cat_data(ts, substats,
                    settings)
                datafile.append_data(data_all)
                string = '  {:0.01f} %'.format(i/len(u)*100)
                self.status_update(string , grow=False)
            self.status_update(' done.')


        def load_processed(self):
            self.filename = QFileDialog.getOpenFileName(self,
                    caption = 'Load stats csv file',
                    directory = self.datadir,
                    filter = (('*-STATS.csv'))
                    )[0]
            if self.filename == '':
                return

            self.status_update('loading ' + self.filename)
            app.processEvents()

            stats = pd.read_csv(self.filename)
            self.stats = stats

            nims = scpp.count_images_in_stats(stats)

            self.status_update(str(nims) + ' images in dataset')
            self.status_update(str(len(stats)) + ' particles in dataset')

            if self.settings == '':
                self.status_update('config file not found. please load one.')
                self.load_sc_config()
                if self.settings == '':
                    return

            dias, vd = scpp.vd_from_stats(stats, self.settings.PostProcess)

            plt.subplot(1,2,1)
            plt.cla()
            vdm = vd / sum(vd) * 100
            plt.plot(dias,vdm,'k')
            plt.xscale('log')
            plt.xlabel('Equiv. diam [um]')
            plt.ylabel('Volume concentration [%/size class]')
            plt.title('Average volume distribution')

            plt.subplot(1,2,2)
            plt.cla()
            scplt.nd_scaled(stats,self.settings.PostProcess, plt.gca())
            plt.title('Average number distribution')

            self.canvas.draw()

        def time_series_vd(self):
            if len(self.stats) == 0 :
                self.load_processed()

            stats = self.stats

            self.status_update('converting particles to distributions')
            self.status_update(' please wait....')
            app.processEvents()

            u = stats['timestamp'].unique()
            vd = []
            for i, s in enumerate(u):
                substats = stats[stats['timestamp']==s]
                dias, vd_ = scpp.vd_from_stats(substats, self.settings.PostProcess)
                vd.append(vd_)
                string = '  {:0.01f} %'.format(i/len(u)*100)
                self.status_update(string , grow=False)

            self.status_update(' done.')

            plt.clf()
            vd = np.transpose(vd)
            plt.pcolormesh(pd.to_datetime(u),dias,vd,
                    norm=matplotlib.colors.LogNorm(), cmap=cmocean.cm.matter)
            plt.ylabel('Equiv. diam [um]')
            plt.yscale('log')

            self.canvas.draw()



        def raw(self):
            self.datadir=QFileDialog.getExistingDirectory(self,'open',self.datadir,QFileDialog.ShowDirsOnly)
            if self.datadir == '':
                self.datadir = DATADIR
                return
            files = [os.path.join(self.datadir, f) for f in
                    sorted(os.listdir(self.datadir))
                    if f.endswith('.bmp')]
            self.nims = len(files)
            if self.nims < 2:
                self.status_update('Less than 2 images found')
                return
            times = names_to_times(files)


            plt.subplot(1,2,1)
            plt.cla()
            plt.plot(times,np.arange(0,self.nims),'k.',markersize=2)
            plt.title(self.datadir + ': ' + str(self.nims) + ' images')
            myFmt = matplotlib.dates.DateFormatter('%D-%H:%M:%S')
            plt.gca().xaxis.set_major_formatter(myFmt)
            plt.gcf().autofmt_xdate()
            plt.ylabel('Image number')

            hz = times_to_hz(times)
            plt.subplot(1,2,2)
            plt.cla()
            plt.hist(hz,bins=np.arange(0,7,0.1),color='k')
            plt.xlabel('Sampling Frequency [Hz]')
            plt.ylabel('Images')

            self.canvas.draw()


        def record(self):
            self.status_update('  ----  ')
            self.status_update('STARTING SILCAM-ACQUIRE!')
            self.status_update('  recording to: ' + self.datadir)
            self.status_update('  ')
            self.status_update('  use Drive monitor to check recording status')
            self.status_update('  use Live raw to see subset of raw images')
            self.status_update('  ')
            self.status_update('  ---- WARNING: Acquisition will continue ' +
                    'even if these windows are closed ----')
            self.status_update('  ----  ')
            self.status_update('  ')
            app.processEvents()
            #self.process=subprocess.Popen(['./logsilcam.sh'])
            app.processEvents()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(0,150,0) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(False)
            app.processEvents()


        def stop_record(self):
            self.status_update('  ----  ')
            self.status_update('  ----  ')
            self.status_update('KILLALL SILCAM-ACQUIRE PROCESSES!')
            self.status_update('  ----  ')
            self.status_update('  ----  ')
            self.status_update('  ')
            #subprocess.call('killall silcam-acquire', shell=True)
            app.processEvents()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(True)
            app.processEvents()


        def exit(self):
            app.quit()

    myapp = StartQT5()
    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
