import sys
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget, QDialog,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import skimage.io
from pysilcam.silcamgui.SilCam import Ui_SilCam
from pysilcam.silcamgui.SilCamController import Ui_SilCamController
from pysilcam.silcamgui.ServerDLG import Ui_Server
from pysilcam.silcamgui.configEditor import Ui_Editconfig
import seaborn as sns
import pysilcam.postprocess as scpp
import pysilcam.plotting as scplt
import pysilcam.datalogger as scdl
import pysilcam.oilgas as scog
import cmocean
import subprocess
import datetime
import pysilcam.silcamgui.guicalcs as gc
from pysilcam.silcamgui.guicalcs import process_mode
from pysilcam.config import PySilcamSettings
from enum import Enum

sns.set_style('ticks')
sns.set_context(font_scale=2)
DATADIR = os.getcwd()
IP = '192.168.1.2'
DEFAULT_CONFIG = "../config_example.ini"

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

    def toggle_browse(self, disable):
        self.ui.le_path_to_data.setDisabled(disable)
        self.ui.pb_browse.setDisabled(disable)

    def toggle_write_to_disc(self, disable):
        self.ui.cb_store_to_disc.setDisabled(disable)

    def update_dir_path(self, dir_path):
        self.ui.le_path_to_data.setText(dir_path)

    def closeEvent(self, event):
        print('closing acquisition dlg is not allowed')
        event.ignore()

class ConfigEditor(QDialog):

    def __init__(self, configfile, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_Editconfig()
        self.ui.setupUi(self)
        self.configfileToModify = configfile
        self.fillInConfigEditor(configfile)
        self.ui.defaultPushButton.clicked.connect(lambda: self.fillInConfigEditor(DEFAULT_CONFIG))
        self.ui.browseDataPathPB.clicked.connect(self.browseDataPath)
        self.ui.browseLogFilePB.clicked.connect(self.browseLogFile)
        self.ui.browseOutputPathPB.clicked.connect(self.browseOutputPath)

    def fillInConfigEditor(self, inputFile):
        self.ui.configPathLabel.setText(self.configfileToModify)
        self.settings = PySilcamSettings(inputFile)
        self.ui.datafileEdit.setText(self.settings.General.datafile)
        idx = self.ui.loglevelEdit.findText(self.settings.General.loglevel, QtCore.Qt.MatchFixedString)
        if (idx == -1):
            idx = 0
        self.ui.loglevelEdit.setCurrentIndex(idx)
        self.ui.logfileEdit.setText(self.settings.General.logfile)
        if (self.settings.Process.real_time_stats == True):
            self.ui.real_time_statsEdit.setCurrentIndex(1)
        else:
            self.ui.real_time_statsEdit.setCurrentIndex(0)
        self.ui.path_lengthEdit.setText(str(self.settings.PostProcess.path_length))
        self.ui.window_sizeEdit.setText(str(self.settings.PostProcess.window_size))

        if (self.settings.ExportParticles.export_images == True):
            self.ui.export_imagesEdit.setCurrentIndex(1)
        else:
            self.ui.export_imagesEdit.setCurrentIndex(0)
        self.ui.outputpathEdit.setText(self.settings.ExportParticles.outputpath)
        self.ui.min_lengthEdit.setText(str(self.settings.ExportParticles.min_length))
        self.ui.num_imagesEdit.setText(str(self.settings.Background.num_images))
        self.ui.thresholdEdit.setText(str(self.settings.Process.threshold))
        self.ui.max_particlesEdit.setText(str(self.settings.Process.max_particles))


    def browseDataPath(self):
        dataPath = QFileDialog.getExistingDirectory(self,'Select output data folder',
                    DATADIR,QFileDialog.ShowDirsOnly)
        if dataPath == '':
            return

        self.ui.datafileEdit.setText(dataPath)

    def browseLogFile(self):
        dialog = QFileDialog(self)
        #logFile = dialog.getSaveFileName(self, "Select log file",
        #                                        DATADIR, "log file (*.log)")
        dialog.setLabelText(QFileDialog.Accept, "Select")
        dialog.setWindowTitle("Select path and enter name for log file")

        if dialog.exec():
            logFile = dialog.selectedFiles()
            logFileFinal = logFile[0]
            if logFile == '':
                return
            if ("." not in logFile[0]):
                logFileFinal = logFile[0] + ".log"
        else:
            return

        self.ui.logfileEdit.setText(logFileFinal)

    def browseOutputPath(self):
        outputPath = QFileDialog.getExistingDirectory(self,'Select output folder for export',
                    DATADIR,QFileDialog.ShowDirsOnly)
        if outputPath == '':
            return

        self.ui.outputpathEdit.setText(outputPath)

    def saveModif(self):
        self.settings.config.set("General", "datafile", self.ui.datafileEdit.text())
        self.settings.config.set("General", "loglevel", self.ui.loglevelEdit.currentText())
        self.settings.config.set("General", "logfile", self.ui.logfileEdit.text())
        self.settings.config.set("Process", "real_time_stats", self.ui.real_time_statsEdit.currentText())
        self.settings.config.set("PostProcess", "path_length", self.ui.path_lengthEdit.text())
        self.settings.config.set("PostProcess", "window_size", self.ui.window_sizeEdit.text())
        self.settings.config.set("ExportParticles", "export_images", self.ui.export_imagesEdit.currentText())
        self.settings.config.set("ExportParticles", "outputpath", self.ui.outputpathEdit.text())
        self.settings.config.set("ExportParticles", "min_length", self.ui.min_lengthEdit.text())
        self.settings.config.set("Background", "num_images", self.ui.num_imagesEdit.text())
        self.settings.config.set("Process", "threshold", self.ui.thresholdEdit.text())
        self.settings.config.set("Process", "max_particles", self.ui.max_particlesEdit.text())

        with open(self.configfileToModify, 'w') as configfile:
            self.settings.config.write(configfile)

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
            self.disc_write = False
            self.run_type = process_mode.process
            self.datadir = DATADIR
            self.process = gc.ProcThread(self.datadir, self.disc_write, self.run_type)

            # ---- figure in middle
            self.fig_main = plt.figure()
            self.canvas = FigureCanvas(self.fig_main)
            layout = QVBoxLayout()
            layout.addWidget(self.canvas)
            self.ui.fig_widget.setLayout(layout)
            path_here = os.path.realpath(__file__)
            imfile = os.path.join(os.path.split(path_here)[0], 'startimg.png')
            im = skimage.io.imread(imfile)
            plt.imshow(im)
            plt.axis('off')
            self.canvas.draw()

            # ---- define some callbacks
            self.ui.actionExit.triggered.connect(self.exit)
            self.ui.actionSilc_viewer.triggered.connect(self.process.silcview)
            #self.ui.actionServer.triggered.connect(scog.ServerThread)
            self.ui.actionServer.triggered.connect(self.server)
            self.ui.actionController.triggered.connect(self.acquire_controller)
            self.ui.actionConvert_silc_to_bmp.triggered.connect(self.convert_silc)
            self.ui.actionExport_summary_data.triggered.connect(self.export_summary_data)
            self.ui.actionExport_summary_figure.triggered.connect(self.export_summary_figure)
            self.ui.actionSilc_file_player.triggered.connect(self.silc_player)
            self.ui.actionEditConfig.triggered.connect(self.editConfig)

            self.layout = layout

            self.status_update('')

            app.processEvents()

            self.acquire_controller()


        def convert_silc(self):
            self.status_update('converting data to bmp....')
            scpp.silc_to_bmp(self.datadir)
            self.status_update('converting finished.')


        def export_summary_figure(self):

            self.status_update('Asking user for config file')
            self.load_sc_config()
            if self.process.configfile == '':
                self.status_update('Did not get config file')
                return

            self.stats_filename = ''
            self.status_update('Asking user for *-STATS.csv file')
            self.load_stats_filename()
            if self.stats_filename == '':
                self.status_update('Did not get STATS file')
                return

            self.status_update('Creating summary figure (all)....')

            plt.figure(figsize=(20,12))
            scplt.summarise_fancy_stats(self.stats_filename,
                    self.process.configfile, monitor=False)
            self.status_update('Saving summary figure (all)....')
            plt.savefig(self.stats_filename.strip('-STATS.csv') + '-Summary.png',
                    dpi=600, bbox_inches='tight')

            plt.figure(figsize=(20,12))
            self.status_update('Creating summary figure (oil)....')
            scplt.summarise_fancy_stats(self.stats_filename,
                    self.process.configfile, monitor=False, oilgas=scpp.outputPartType.oil)
            self.status_update('Saving summary figure (oil)....')
            plt.savefig(self.stats_filename.strip('-STATS.csv') + '-Summary_oil.png',
                    dpi=600, bbox_inches='tight')

            plt.figure(figsize=(20,12))
            self.status_update('Creating summary figure (gas)....')
            scplt.summarise_fancy_stats(self.stats_filename,
                    self.process.configfile, monitor=False, oilgas=scpp.outputPartType.gas)
            self.status_update('Saving summary figure (gas)....')
            plt.savefig(self.stats_filename.strip('-STATS.csv') + '-Summary_gas.png',
                    dpi=600, bbox_inches='tight')

            self.status_update('Summary figure done.')

            plt.figure(self.fig_main.number)


        def export_summary_data(self):

            self.status_update('Asking user for config file')
            self.load_sc_config()
            if self.process.configfile == '':
                self.status_update('Did not get config file')
                return

            self.stats_filename = ''
            self.status_update('Asking user for *-STATS.csv file')
            self.load_stats_filename()
            if self.stats_filename == '':
                self.status_update('Did not get STATS file')
                return

            self.status_update('Exporting all data....')
            df = scpp.stats_to_xls_png(self.process.configfile,
                    self.stats_filename)
            plt.figure(figsize=(20,12))
            plt.plot(df['Time'], df['D50'],'k.')
            plt.ylabel('d50 [um]')
            plt.savefig(self.stats_filename.strip('-STATS.csv') +
                    '-d50_TimeSeries.png', dpi=600, bbox_inches='tight')

            self.status_update('Exporting oil data....')
            df = scpp.stats_to_xls_png(self.process.configfile,
                    self.stats_filename, oilgas=scpp.outputPartType.oil)
            plt.figure(figsize=(20,12))
            plt.plot(df['Time'], df['D50'],'k.')
            plt.ylabel('d50 [um]')
            plt.savefig(self.stats_filename.strip('-STATS.csv') +
                    '-d50_TimeSeries_oil.png', dpi=600, bbox_inches='tight')

            self.status_update('Exporting gas data....')
            df = scpp.stats_to_xls_png(self.process.configfile,
                    self.stats_filename, oilgas=scpp.outputPartType.gas)
            plt.figure(figsize=(20,12))
            plt.plot(df['Time'], df['D50'],'k.')
            plt.ylabel('d50 [um]')
            plt.savefig(self.stats_filename.strip('-STATS.csv') +
                    '-d50_TimeSeries_gas.png', dpi=600, bbox_inches='tight')

            self.status_update('Export finished.')

            plt.figure(self.fig_main.number)

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
            self.ctrl.ui.pb_browse.clicked.connect(self.change_directory)

            self.status_update('opening acquisition controller')

            self.ctrl.ui.rb_to_disc.toggled.connect(lambda:
                    self.ctrl.toggle_browse(disable=False))
            self.ctrl.ui.rb_to_disc.toggled.connect(lambda:
                    self.ctrl.toggle_write_to_disc(disable=False))
            self.ctrl.ui.rb_to_disc.toggled.connect(lambda checked:
                    self.ctrl.ui.cb_store_to_disc.setChecked(False))
            self.ctrl.ui.rb_to_disc.toggled.connect(lambda: self.setProcessMode(process_mode.aquire))

            self.ctrl.ui.rb_process_historical.toggled.connect(lambda: self.ctrl.toggle_browse(disable=False))
            self.ctrl.ui.rb_process_historical.toggled.connect(lambda: self.ctrl.toggle_write_to_disc(disable=True))
            self.ctrl.ui.rb_process_historical.toggled.connect(
                lambda checked: self.ctrl.ui.cb_store_to_disc.setChecked(False))
            self.ctrl.ui.rb_process_historical.toggled.connect(lambda: self.setProcessMode(process_mode.process))

            self.ctrl.ui.rb_real_time.toggled.connect(lambda: self.ctrl.toggle_browse(disable=False))
            self.ctrl.ui.rb_real_time.toggled.connect(lambda: self.ctrl.toggle_write_to_disc(disable=False))
            self.ctrl.ui.rb_real_time.toggled.connect(lambda: self.setProcessMode(process_mode.real_time))
            self.ctrl.ui.cb_store_to_disc.toggled.connect(lambda checked: self.setStoreToDisc(checked))

            self.reset_acquire_dlg()

        def reset_acquire_dlg(self):
            if(self.run_type == process_mode.process):
                self.ctrl.ui.rb_process_historical.setChecked(True)
                self.ctrl.ui.cb_store_to_disc.setEnabled(False)
            elif(self.run_type == process_mode.aquire):
                self.ctrl.ui.rb_to_disc.setChecked(True)
                self.ctrl.ui.cb_store_to_disc.setEnabled(True)
            elif(self.run_type == process_mode.real_time):
                self.ctrl.ui.rb_real_time.setChecked(True)
                self.ctrl.ui.cb_store_to_disc.setEnabled(True)

            self.lv_raw_check()
            self.ctrl.show()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' + 'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' + 'background-color: rgb(150,150,255) }'))



        def setProcessMode(self, mode):
            self.run_type = mode
            app.processEvents()


        @pyqtSlot(bool, name='checked')
        def setStoreToDisc(self, checked):
            self.disc_write = checked


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


        def lv_raw(self):
            self.lv_raw_check()
            if not self.lv_raw_toggle:
                return

            self.process.plot()
            self.status_update('', uselog=True)
            self.canvas.draw()

            QtCore.QTimer.singleShot(self.lvwaitseconds*1000, self.lv_raw)


        def lv_raw_check(self):
            if self.lv_raw_toggle:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' + 'background-color: rgb(0,150,0) }'))
            else:
                self.ctrl.ui.pb_live_raw.setStyleSheet(('QPushButton {' + 'background-color: rgb(150,150,255) }'))


        def status_update(self, string, uselog=False):
            if uselog:
                try:
                    settings = PySilcamSettings(self.process.configfile)
                    with open(settings.General.logfile, 'r') as a:
                        lines = a.readlines()
                        string = str(lines[-1])
                except:
                    pass
            string = string + '  |  Directory: ' + self.datadir
            self.ui.statusBar.setText(string)
            app.processEvents()


        def count_data(self):
            silc, bmp = gc.count_data(self.datadir)
            self.status_update(('New directory contains ' +
                    str(silc) + ' silc files and ' +
                    str(bmp) + ' bmp files'))


        def change_directory(self):
            inidir = self.datadir
            self.datadir=QFileDialog.getExistingDirectory(self,'open',
                    self.datadir,QFileDialog.ShowDirsOnly)
            if self.datadir == '':
                self.datadir = inidir

            self.count_data()
            self.ctrl.update_dir_path(self.datadir)
            self.process = gc.ProcThread(self.datadir, self.disc_write, self.run_type)
            app.processEvents()


        def silc_player(self):
            self.process.silcview()


        def record(self):
            self.process = gc.ProcThread(self.datadir, self.disc_write, self.run_type)
            if self.process.settings == '':
                self.status_update('config file not found. please load one.')
                self.load_sc_config()
                if self.process.configfile == '':
                    return

            self.status_update('STARTING SILCAM!')
            self.process.go()
            app.processEvents()
            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' + 'background-color: rgb(0,150,0) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' + 'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(False)
            self.ctrl.ui.cb_store_to_disc.setEnabled(False)
            app.processEvents()


        def stop_record(self):
            self.status_update('Asking silcam to stop')
            self.process.stop_silcam()
            self.status_update('Asking silcam to stop.. OK')
            self.reset_acquire_dlg()
            app.processEvents()

            self.ctrl.ui.pb_start.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_stop.setStyleSheet(('QPushButton {' +
                'background-color: rgb(150,150,255) }'))
            self.ctrl.ui.pb_start.setEnabled(True)
            app.processEvents()


        def load_stats_filename(self):
            self.stats_filename = QFileDialog.getOpenFileName(self,
                    caption = 'Load a *-STATS.csv file',
                    directory = self.datadir,
                    filter = (('*-STATS.csv'))
                    )[0]
            if self.stats_filename == '':
                return


        def load_sc_config(self):
            self.process.configfile = QFileDialog.getOpenFileName(self,
                    caption = 'Load config ini file',
                    directory = self.datadir,
                    filter = (('*.ini'))
                    )[0]
            if self.process.configfile == '':
                return

            # move current directory to the config file folder in order to
            # handle relative paths fened from the config file
            os.chdir(os.path.split(self.process.configfile)[0])

        def editConfig(self):
            configfile = QFileDialog.getOpenFileName(self,
                    caption = 'Select config ini file',
                    directory = self.datadir,
                    filter = (('*.ini'))
                    )[0]
            if configfile == '':
                return

            configEditor = ConfigEditor(configfile)
            if (configEditor.exec() == QDialog.Accepted):
                configEditor.saveModif()

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
