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
from pysilcam.silcamgui.ServerDLG import Ui_Server
import seaborn as sns
import pysilcam.postprocess as scpp
import pysilcam.plotting as scplt
import pysilcam.datalogger as scdl
import pysilcam.oilgas as scog
sns.set_style('ticks')
sns.set_context(font_scale=2)
import cmocean
import subprocess
import datetime
import pysilcam.silcamgui.guicalcs as gc

DATADIR = os.getcwd()
#DATADIR = '/mnt/DATA/'
IP = '192.168.1.2'

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


class server_dlg(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_Server()
        self.ui.setupUi(self)
        self.ui.IPText.appendPlainText(IP)

    def go(self):
        ip = self.ui.IPText.toPlainText()
        self.server = scog.ServerThread(ip)


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
            self.settings = ''
            self.stats = []
            self.lv_raw_toggle = False
            self.monitor_toggle = False
            self.lvwaitseconds = 1
            self.process = gc.ProcThread(DATADIR)

            # ---- figure in middle
            f = plt.figure()
            self.canvas = FigureCanvas(f)
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            self.ui.fig_widget.setLayout(layout)
            path_here = os.path.realpath(__file__)
            imfile = os.path.join(os.path.split(path_here)[0], 'ojleutslipp.jpg')
            im = skimage.io.imread(imfile)
            plt.imshow(im)
            plt.axis('off')
            self.canvas.draw()
            # ----

            # ---- define some callbacks
            self.ui.actionExit.triggered.connect(self.exit)
            self.ui.actionSilc_viewer.triggered.connect(self.process.silcview)
            #self.ui.actionServer.triggered.connect(scog.ServerThread)
            self.ui.actionServer.triggered.connect(self.server)
            self.ui.actionController.triggered.connect(self.acquire_controller)
            self.ui.pb_ChangeDirectory.clicked.connect(self.change_directory)

            self.layout = layout

            self.status_update('')

            app.processEvents()

            self.acquire_controller()


        def server(self):
            print('opening serverdlg')
            self.serverdlg = server_dlg(self)
            self.serverdlg.ui.Start.clicked.connect(self.serverdlg.go)
            self.serverdlg.show()


        def acquire_controller(self):
            self.ctrl = controller(self)
            self.ctrl.ui.pb_live_raw.clicked.connect(self.lv_raw_switch)
            self.ctrl.ui.pb_start.clicked.connect(self.record)
            self.ctrl.ui.pb_stop.clicked.connect(self.stop_record)
            self.status_update('opening acquisition controller')
            self.lv_raw_check()
            self.ctrl.show()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))


        def monitor_switch(self):
            self.monitor_toggle = np.invert(self.monitor_toggle)
            if self.monitor_toggle:
                self.status_update('monitoring ' + self.process.datadir)
                self.status_update(' ' + self.process.datadir)
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


        def lv_raw(self):
            self.lv_raw_check()
            if not self.lv_raw_toggle:
                return

            self.status_update('asking for plot')
            self.process.plot()
            self.status_update(self.process.info)
            self.canvas.draw()

            QtCore.QTimer.singleShot(self.lvwaitseconds*1000, self.lv_raw)


        def lv_raw_check(self):
            if self.lv_raw_toggle:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(0,150,0) }'))
            else:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' +
                        'background-color: rgb(150,150,255) }'))


        def status_update(self, string):
            string = string + '  |  Directory: ' + self.process.datadir
            self.ui.statusBar.setText(string)
            app.processEvents()


        def change_directory(self):
            inidir = self.process.datadir
            self.process.datadir=QFileDialog.getExistingDirectory(self,'open',
                    self.process.datadir,QFileDialog.ShowDirsOnly)
            if self.process.datadir == '':
                self.process.datadir = inidir
            else:
                self.status_update('(new directory)')
            app.processEvents()


        def record(self):
            if self.process.settings == '':
                self.status_update('config file not found. please load one.')
                self.load_sc_config()
                if self.process.settings == '':
                    return

            self.status_update('STARTING SILCAM!')
            self.process.go()

            app.processEvents()

            #psc.silcam_process('config.ini', self.datadir)
            #self.process=subprocess.Popen(['./logsilcam.sh'])
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(0,150,0) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(False)
            app.processEvents()


        def stop_record(self):
            self.process.stop_silcam()

            self.status_update('  ----  ')
            self.status_update('KILLING SILCAM PROCESS')
            self.status_update('  ----  ')
                #subprocess.call('killall silcam-acquire', shell=True)
            self.process = gc.ProcThread(self.process.datadir)
            app.processEvents()

            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(True)
            app.processEvents()


        def load_sc_config(self):
            self.process.configfile = QFileDialog.getOpenFileName(self,
                    caption = 'Load config ini file',
                    directory = self.process.datadir,
                    filter = (('*.ini'))
                    )[0]
            if self.process.configfile == '':
                return
            self.process.load_settings(self.process.configfile)


        def closeEvent(self, event):
            self.stop_record()
            try:
                self.serverdlg.server.terminate()
            except:
                pass

        def exit(self):
            app.quit()


    myapp = StartQT5()
    myapp.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
