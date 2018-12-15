import pandas as pd
import pysilcam.postprocess as scpp
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
from tqdm import tqdm
import numpy as np
import cmocean
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtWidgets import (QMainWindow, QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import sys


class InteractivePlotter(QMainWindow):
    def __init__(self, parent=None):
        super(InteractivePlotter, self).__init__(parent)
        self.showMaximized()
        self.setWindowTitle("SummaryExplorer")
        QApplication.processEvents()
        self.fft_frame = FftFrame(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.fft_frame)
        self.setLayout(self.layout)
        self.setCentralWidget(self.fft_frame)
        self.showMaximized()


        self.setWindowTitle('Loading: ' + self.fft_frame.graph_view.stats_filename)
        QApplication.processEvents()

        self.fft_frame.graph_view.setup_figure(self.fft_frame.graph_view.configfile,
                                               self.fft_frame.graph_view.stats_filename)

        self.setWindowTitle(self.fft_frame.graph_view.stats_filename)
        QApplication.processEvents()

        self.fft_frame.graph_view.update_plot(self.fft_frame.graph_view.mid_time)
        QApplication.processEvents()


class FftFrame(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super(FftFrame, self).__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.parent = parent
        self.graph_view = GraphView(self)

    def resizeEvent(self, event):
        self.graph_view.setGeometry(self.rect())


class GraphView(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super(GraphView, self).__init__(parent)

        # self.fig, self.axes = plt.subplots(2,2)
        self.fig = plt.figure()
        self.axisconstant = plt.subplot(221)
        self.axispsd = plt.subplot(122)
        self.axistext = plt.subplot(223)
        plt.sca(self.axistext)
        plt.axis('off')
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.setStretchFactor(self.canvas, 1)
        self.setLayout(self.layout)

        self.configfile = "E:/PJ/MiniTowerSilCamConfig.ini"
        self.stats_filename = "E:/PJ/Oseberg2017OilOnly0.25mmNozzle2-STATS.csv"
        self.canvas.draw()

    def setup_figure(self, config_file, stats_csv_file):
        settings = PySilcamSettings(config_file)
        print('Loading stats')
        stats = pd.read_csv(stats_csv_file, nrows=10000, parse_dates=['timestamp'])
        print('  OK.')

        u = stats['timestamp'].unique()
        u = pd.to_datetime(u)
        sample_volume = scpp.get_sample_volume(settings.PostProcess.pix_size,
                                               path_length=settings.PostProcess.path_length)

        dias, bin_lims = scpp.get_size_bins()
        vd_oil = np.zeros((len(u), len(dias)))
        vd_gas = np.zeros_like(vd_oil)
        vd_total = np.zeros_like(vd_oil)
        d50_gas = np.zeros(len(u))
        d50_oil = np.zeros_like(d50_gas)
        d50_total = np.zeros_like(d50_gas)
        # @todo make this number of particle per image, and sum according to index later
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

        plt.sca(self.axisconstant)
        plt.pcolormesh(u, dias, np.log(vd_total.T), cmap=cmocean.cm.amp)
        plt.plot(u, d50_total, 'kx', markersize=5, alpha=0.25)
        plt.plot(u, d50_gas, 'bx', markersize=5, alpha=0.25)
        plt.yscale('log')
        plt.ylabel('ECD [um]')

        self.start_time = min(u)
        self.end_time = max(u)
        self.mid_time = min(u) + (max(u) - min(u)) / 2
        self.line1 = plt.vlines(self.start_time, 1, 12000, 'r')
        self.line2 = plt.vlines(self.end_time, 1, 12000, 'r')

        self.u = u
        self.vd_total = vd_total
        self.vd_oil = vd_oil
        self.vd_gas = vd_gas
        self.dias = dias
        self.fig.canvas.callbacks.connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes is not None:
            mid_time = pd.to_datetime(matplotlib.dates.num2date(event.xdata))
            # mid_time.tz_convert(None)
            self.update_plot(mid_time)
        else:
            print('Clicked ouside axes bounds but inside plot window')

    def update_plot(self, mid_time):


        print(mid_time)

        # mid_time = pd.to_datetime('2018-11-21 11:10:00')

        av_window = pd.to_timedelta('00:00:05')

        start_time = mid_time - av_window / 2
        end_time = mid_time + av_window / 2
        u = pd.to_datetime(self.u)
        print(u[0], start_time, end_time)
        timeind = np.argwhere((u > start_time) & (u < end_time))

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


        plt.sca(self.axispsd)
        plt.cla()
        plt.plot(self.dias, psd_total, 'k', linewidth=5, label='Total')
        plt.plot(self.dias, psd_oil, color=[0.7, 0.4, 0], label='Oil')
        plt.plot(self.dias, psd_gas, 'b', label='Gas')
        plt.xlabel('ECD [um]')
        plt.ylabel('VD [uL/L]')
        plt.xscale('log')

        plt.sca(self.axistext)

        string = ''
        string += 'GOR: {:0.01f}'.format(psd_gor)
        string += '\n d50 total [um]: {:0.0f}'.format(psd_d50_total)
        string += '\n d50 oil [um]: {:0.0f}'.format(psd_d50_oil)
        string += '\n d50 gas [um]: {:0.0f}'.format(psd_d50_gas)
        string += '\n VC total [uL/L]: {:0.0f}'.format(psd_vc_total)
        string += '\n VC oil [uL/L]: {:0.0f}'.format(psd_vc_oil)
        string += '\n VC gas [uL/L]: {:0.0f}'.format(psd_vc_gas)
        string += '\n Num images: {:0.0f}'.format(psd_nims)
        string += '\n Start: ' + str(pd.to_datetime(psd_start[0]))
        string += '\n Start: ' + str(pd.to_datetime(psd_start[0]))
        string += '\n End: ' + str(pd.to_datetime(psd_end[0]))
        string += '\n Window [sec.] {:0.0f}:'.format(pd.to_timedelta(psd_end[0]-psd_start[0]).seconds)

        plt.title(string, verticalalignment='top', horizontalalignment='right', loc='right')

        plt.sca(self.axisconstant)
        self.line1.remove()
        self.line2.remove()
        self.line1 = plt.vlines(start_time, 1, 12000, 'r')
        self.line2 = plt.vlines(end_time, 1, 12000, 'r')
        self.canvas.draw()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractivePlotter()
    window.show()
    app.exec_()