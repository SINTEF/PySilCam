import pandas as pd
import pysilcam.postprocess as scpp
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
from tqdm import tqdm
import numpy as np
import cmocean
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget, QDialog,
QAction, QTabWidget,QVBoxLayout, QFileDialog, QMessageBox)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pysilcam.silcamgui.SummaryExplorer import Ui_SummaryExplorer
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas



class Plotter(QMainWindow):
    def __init__(self, config_file, stats_csv_file, parent=None):
        QMainWindow.__init__(self, parent)
        self.ui = Ui_SummaryExplorer()
        self.ui.setupUi(self)

        self.PLTwidget = plt.figure()
        self.canvas = FigureCanvas(self.PLTwidget)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.PLTwidget.setLayout(layout)
        # self.canvas.updateGeometry()
        self.canvas.draw()
        self.figure = plt.gcf()

        print('showing')
        self.showMaximized()

        # self.figure, self.a = plt.subplots(1, 2, figsize=(15, 6))
        # self.canvas = FigureCanvas(self.figure)
        # layout = QVBoxLayout()
        # layout.addWidget(self.canvas)
        # self.ui.PLTwidget.setLayout(layout)
        # self.canvas.draw()
        return
        # self.figure = self.ui.PLTwidget
        # config_file = "/mnt/PDrive/PJ/MiniTowerSilCamConfig.ini"
        # stats_csv_file = "/mnt/PDrive/PJ/Oseberg2017OilOnly0.25mmNozzle2-STATS.csv"

        settings = PySilcamSettings(config_file)
        print('Loading stats')
        stats = pd.read_csv(stats_csv_file, nrows=10000, parse_dates=['timestamp'])
        print('  OK.')

        u = stats['timestamp'].unique()
        sample_volume = scpp.get_sample_volume(settings.PostProcess.pix_size, path_length=settings.PostProcess.path_length)

        dias, bin_lims = scpp.get_size_bins()
        vd_oil = np.zeros((len(u), len(dias)))
        vd_gas = np.zeros_like(vd_oil)
        vd_total = np.zeros_like(vd_oil)
        d50_gas = np.zeros(len(u))
        d50_oil = np.zeros_like(d50_gas)
        d50_total = np.zeros_like(d50_gas)
        #@todo make this number of particle per image, and sum according to index later
        nparticles_all = 0
        nparticles_total = 0
        nparticles_oil = 0
        nparticles_gas = 0

        print('Analysing time-series')
        for i, s in enumerate(tqdm(u)):
            substats = stats[stats['timestamp'] == s]
            nparticles_all += len(substats)

            nims = scpp.count_images_in_stats(substats)
            sv = sample_volume * nims

            oil = scog.extract_oil(substats)
            nparticles_oil += len(oil)
            dias, vd_oil_ = scpp.vd_from_stats(oil, settings.PostProcess)
            vd_oil_ /= sv
            vd_oil[i, :] = vd_oil_

            gas = scog.extract_gas(substats)
            nparticles_gas += len(gas)
            dias, vd_gas_ = scpp.vd_from_stats(gas, settings.PostProcess)
            vd_gas_ /= sv
            vd_gas[i, :] = vd_gas_
            d50_gas[i] = scpp.d50_from_vd(vd_gas_, dias)

            nparticles_total += len(oil) + len(gas)
            vd_total_ = vd_oil_ + vd_gas_
            d50_total[i] = scpp.d50_from_vd(vd_total_, dias)
            vd_total[i, :] = vd_total_

        # f, self.a = plt.subplots(1, 2, figsize=(15, 6))
        plt.sca(self.a[0])

        plt.pcolormesh(u, dias, np.log(vd_total.T), cmap=cmocean.cm.amp)
        plt.plot(u, d50_total, 'kx', markersize=5, alpha=0.25)
        plt.plot(u, d50_gas, 'bx', markersize=5, alpha=0.25)
        plt.yscale('log')

        start_time = min(u)
        end_time = max(u)
        self.line1 = plt.vlines(start_time, 1, 12000, 'r')
        self.line2 = plt.vlines(end_time, 1, 12000, 'r')

        self.u = u
        self.vd_total = vd_total
        self.vd_oil = vd_oil
        self.vd_gas = vd_gas
        self.dias = dias
        self.figure.canvas.callbacks.connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not None:
            mid_time = pd.to_datetime(matplotlib.dates.num2date(event.xdata))
            mid_time.tz_convert(None)
            print(mid_time)

            # mid_time = pd.to_datetime('2018-11-21 11:10:00')

            av_window = pd.to_timedelta('00:5:00')

            start_time = mid_time - av_window / 2
            end_time = mid_time + av_window / 2
            u = pd.to_datetime(self.u)
            print(u[0], start_time, end_time)
            timeind = np.argwhere((u > start_time.tz_convert(None)) & (u < end_time.tz_convert(None)))

            psd_total = np.mean(self.vd_total[timeind, :], axis=0)[0]
            psd_oil = np.mean(self.vd_oil[timeind, :], axis=0)[0]
            psd_gas = np.mean(self.vd_gas[timeind, :], axis=0)[0]

            psd_vc_total = np.sum(psd_total)
            psd_vc_oil = np.sum(psd_oil)
            psd_vc_gas = np.sum(psd_gas)

            psd_d50_total = scpp.d50_from_vd(psd_total, self.dias)
            psd_d50_oil = scpp.d50_from_vd(psd_oil, self.dias)
            psd_d50_gas = scpp.d50_from_vd(psd_gas, self.dias)

            psd_gor = sum(psd_gas) / (sum(psd_oil) + sum(psd_gas)) * 100

            psd_nims = len(timeind)
            psd_start = min(u[timeind])
            psd_end = max(u[timeind])
            psd_mid = psd_start + (psd_end - psd_start) / 2


            plt.sca(self.a[1])
            plt.cla()
            plt.plot(self.dias, psd_total, 'k', linewidth=5)
            plt.plot(self.dias, psd_oil, color=[0.7, 0.4, 0])
            plt.plot(self.dias, psd_gas, 'b')
            plt.xscale('log')

            print('GOR [%]:', psd_gor)
            print('d50 total [um]:', psd_d50_total)
            print('d50 oil [um]:', psd_d50_oil)
            print('d50 gas [um]:', psd_d50_gas)
            print('VC total [uL/L]:', psd_vc_total)
            print('VC oil [uL/L]:', psd_vc_oil)
            print('VC gas [uL/L]:', psd_vc_gas)
            print('Number of images:', psd_nims)
            print('start:', pd.to_datetime(psd_start[0]))
            print('end:', pd.to_datetime(psd_end[0]))
            print('mid(ish):', pd.to_datetime(psd_mid[0]))
            print('specified mid time:', mid_time)

            plt.sca(self.a[0])
            self.line1.remove()
            self.line2.remove()
            self.line1 = plt.vlines(start_time, 1, 12000, 'r')
            self.line2 = plt.vlines(end_time, 1, 12000, 'r')
            self.canvas.draw()
        else:
            print('Clicked ouside axes bounds but inside plot window')

    def closeEvent(self, event):
        pass


if __name__ == '__main__':
    pass