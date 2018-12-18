import pandas as pd
import pysilcam.postprocess as scpp
import pysilcam.oilgas as scog
from pysilcam.config import PySilcamSettings
from tqdm import tqdm
import numpy as np
import cmocean
import matplotlib.pyplot as plt
import matplotlib
from PyQt5.QtWidgets import QMainWindow, QApplication, QAction, QInputDialog, QMessageBox, QFileDialog
from PyQt5 import QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import sys
import os
from pysilcam.silcamgui.guicalcs import export_timeseries


class FigFrame(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super(FigFrame, self).__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.parent = parent
        self.graph_view = PlotView(self)

    def resizeEvent(self, event):
        self.graph_view.setGeometry(self.rect())


class InteractivePlotter(QMainWindow):
    def __init__(self, parent=None):
        super(InteractivePlotter, self).__init__(parent)
        self.showMaximized()
        self.setWindowTitle("SummaryExplorer")
        QApplication.processEvents()
        self.plot_fame = FigFrame(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.plot_fame)
        self.setLayout(self.layout)
        self.setCentralWidget(self.plot_fame)
        self.showMaximized()

        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')

        loadButton = QAction('Load', self)
        loadButton.setStatusTip('Load data')
        loadButton.setShortcut("Ctrl+o")
        loadButton.triggered.connect(self.plot_fame.graph_view.load_data)
        fileMenu.addAction(loadButton)

        saveButton = QAction('Save', self)
        saveButton.setStatusTip('Save PSD data to xls')
        saveButton.setShortcut("Ctrl+s")
        saveButton.triggered.connect(self.plot_fame.graph_view.save_data)
        fileMenu.addAction(saveButton)

        avwinButton = QAction('Average window', self)
        avwinButton.setStatusTip('Change the average window')
        avwinButton.triggered.connect(self.modify_av_wind)
        fileMenu.addAction(avwinButton)

        exitButton = QAction('Exit', self)
        exitButton.setStatusTip('Close')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)


        # self.setWindowTitle('Loading: ' + self.plot_fame.graph_view.stats_filename)
        # QApplication.processEvents()

        # self.plot_fame.graph_view.setup_figure(self.plot_fame.graph_view.configfile,
        #                                        self.plot_fame.graph_view.stats_filename)

        # self.setWindowTitle(self.plot_fame.graph_view.stats_filename)
        # QApplication.processEvents()
        #
        # self.plot_fame.graph_view.update_plot()
        # QApplication.processEvents()


    def keyPressEvent(self, event):
        pressedkey = event.key()
        if (pressedkey == QtCore.Qt.Key_Up) or (pressedkey == QtCore.Qt.Key_W):
            self.plot_fame.graph_view.av_window += pd.Timedelta(seconds=1)
            self.plot_fame.graph_view.update_plot()
            event.accept()
        elif (pressedkey == QtCore.Qt.Key_Right) or (pressedkey == QtCore.Qt.Key_D):
            self.plot_fame.graph_view.mid_time += pd.Timedelta(seconds=1)
            self.plot_fame.graph_view.update_plot()
            event.accept()
        elif (pressedkey == QtCore.Qt.Key_Down) or (pressedkey == QtCore.Qt.Key_S):
            self.plot_fame.graph_view.av_window -= pd.Timedelta(seconds=1)
            self.plot_fame.graph_view.av_window = max(pd.Timedelta(seconds=1), self.plot_fame.graph_view.av_window)
            self.plot_fame.graph_view.update_plot()
            event.accept()
        elif (pressedkey == QtCore.Qt.Key_Left) or (pressedkey == QtCore.Qt.Key_A):
            self.plot_fame.graph_view.mid_time -= pd.Timedelta(seconds=1)
            self.plot_fame.graph_view.av_window = max(pd.Timedelta(seconds=1), self.plot_fame.graph_view.av_window)
            self.plot_fame.graph_view.update_plot()
            event.accept()
        else:
            event.ignore()


    def modify_av_wind(self):
        window_seconds = self.plot_fame.graph_view.av_window.seconds
        input_value, okPressed = QInputDialog.getInt(self, "Get integer", "Average window:", window_seconds, 0, 60*60, 1)

        if okPressed:
            self.plot_fame.graph_view.av_window = pd.Timedelta(seconds=input_value)


class PlotView(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(PlotView, self).__init__(parent)

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

        self.configfile = ''
        self.stats_filename = ''
        self.av_window = pd.Timedelta(seconds=30)
        self.datadir = os.getcwd()
        # self.configfile = "E:/PJ/MiniTowerSilCamConfig.ini"
        # self.stats_filename = "E:/PJ/Oseberg2017OilOnly0.25mmNozzle2-STATS.csv"
        # self.stats_filename = "E:/PJ/Oseberg2017OilOnly0.25mmNozzle2sdghjsk-STATS.csv"

        # self.load_data()

        # self.load_from_stats()

        # self.load_from_timeseries()

        self.canvas.draw()


    def load_data(self):
        self.datadir = os.path.split(self.configfile)[0]

        self.stats_filename = ''
        self.stats_filename = QFileDialog.getOpenFileName(self,
                                                          caption='Load a *-STATS.csv file',
                                                          directory=self.datadir,
                                                          filter=(('*-STATS.csv'))
                                                          )[0]
        if self.stats_filename == '':
            return

        timeseriesgas_file = self.stats_filename.replace('-STATS.csv', '-TIMESERIESgas.xlsx')

        if os.path.isfile(timeseriesgas_file):
            self.load_from_timeseries()
        else:

            msgBox = QMessageBox()
            msgBox.setText('The STATS data appear not to have been exported to TIMSERIES.xlsx' +
                           '\n We can use the STATS file anyway (which might take a while)' +
                           '\n or we can convert the data to TIMSERIES.xls now,'
                           '\n which can be used quickly if you want to load these data another time.')
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setWindowTitle('What to do?')
            load_stats_button = msgBox.addButton('Load stats anyway',
                                                 QMessageBox.ActionRole)
            convert_stats_button = msgBox.addButton('Convert and save timeseries',
                                                    QMessageBox.ActionRole)
            msgBox.addButton(QMessageBox.Cancel)
            msgBox.exec_()
            if self.configfile == '':
                self.configfile = QFileDialog.getOpenFileName(self,
                                                                  caption='Load config ini file',
                                                                  directory=self.datadir,
                                                                  filter=(('*.ini'))
                                                                  )[0]
                if self.configfile == '':
                    return
            if (msgBox.clickedButton() == load_stats_button):
                self.settings = PySilcamSettings(self.configfile)
                self.av_window = pd.Timedelta(seconds=self.settings.PostProcess.window_size)
                self.load_from_stats()
            elif (msgBox.clickedButton() == convert_stats_button):
                export_timeseries(self.configfile, self.stats_filename)
                self.load_from_timeseries()
            else:
                print('cancel')
                return

        self.setup_figure()




    def load_from_timeseries(self):
        timeseriesgas_file = self.stats_filename.replace('-STATS.csv', '-TIMESERIESgas.xlsx')
        timeseriesoil_file = self.stats_filename.replace('-STATS.csv', '-TIMESERIESoil.xlsx')

        print(timeseriesgas_file)

        gas = pd.read_excel(timeseriesgas_file, parse_dates=['Time'])
        oil = pd.read_excel(timeseriesoil_file, parse_dates=['Time'])

        self.dias = np.array(oil.columns[0:52], dtype=float)
        self.vd_oil = oil.as_matrix(columns=oil.columns[0:52])
        self.vd_gas = gas.as_matrix(columns=gas.columns[0:52])
        self.vd_total = self.vd_oil + self.vd_gas
        self.u = pd.to_datetime(oil['Time'].values)
        self.d50_gas = gas['D50']
        self.d50_oil = oil['D50']

        # nc = scpp.vd_to_nc(vd_oil, dias)

        self.d50_total = np.zeros_like(self.d50_oil)
        for i, vd in enumerate(self.vd_total):
            self.d50_total[i] = scpp.d50_from_vd(vd, self.dias)


    def load_from_stats(self):
        stats = pd.read_csv(self.stats_filename, parse_dates=['timestamp'])

        u = stats['timestamp'].unique()
        u = pd.to_datetime(u)
        sample_volume = scpp.get_sample_volume(self.settings.PostProcess.pix_size,
                                               path_length=self.settings.PostProcess.path_length)

        dias, bin_lims = scpp.get_size_bins()
        vd_oil = np.zeros((len(u), len(dias)))
        vd_gas = np.zeros_like(vd_oil)
        vd_total = np.zeros_like(vd_oil)
        d50_gas = np.zeros(len(u))
        d50_oil = np.zeros_like(d50_gas)
        d50_total = np.zeros_like(d50_gas)
        # @todo make this number of particles per image, and sum according to index later
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
            dias, vd_oil_ = scpp.vd_from_stats(oil, self.settings.PostProcess)
            vd_oil_ /= sv
            vd_oil[i, :] = vd_oil_

            gas = scog.extract_gas(substats)
            nparticles_gas += len(gas)
            dias, vd_gas_ = scpp.vd_from_stats(gas, self.settings.PostProcess)
            vd_gas_ /= sv
            vd_gas[i, :] = vd_gas_
            d50_gas[i] = scpp.d50_from_vd(vd_gas_, dias)

            nparticles_total += len(oil) + len(gas)
            vd_total_ = vd_oil_ + vd_gas_
            d50_total[i] = scpp.d50_from_vd(vd_total_, dias)
            vd_total[i, :] = vd_total_

        self.vd_total = vd_total
        self.vd_gas = vd_gas
        self.vd_oil = vd_oil
        self.d50_total = d50_total
        self.d50_oil = d50_oil
        self.d50_gas = d50_gas
        self.u = u
        self.dias = dias


    def setup_figure(self):


        # f, self.a = plt.subplots(1, 2, figsize=(15, 6))

        plt.sca(self.axisconstant)
        plt.cla()
        plt.pcolormesh(self.u, self.dias, np.log(self.vd_total.T), cmap=cmocean.cm.matter)
        plt.plot(self.u, self.d50_total, 'kx', markersize=5, alpha=0.25)
        plt.plot(self.u, self.d50_gas, 'bx', markersize=5, alpha=0.25)
        plt.yscale('log')
        plt.ylabel('ECD [um]')
        plt.ylim(10, 12000)

        self.start_time = min(self.u)
        self.end_time = max(self.u)
        self.mid_time = min(self.u) + (max(self.u) - min(self.u)) / 2
        self.line1 = plt.vlines(self.start_time, 1, 12000, 'r')
        self.line2 = plt.vlines(self.end_time, 1, 12000, 'r')

        self.fig.canvas.callbacks.connect('button_press_event', self.on_click)

        self.update_plot()

    def on_click(self, event):
        if event.inaxes is not None:
            try:
                self.mid_time = pd.to_datetime(matplotlib.dates.num2date(event.xdata))
                # mid_time.tz_convert(None)
                self.update_plot()
            except:
                pass
        else:
            print('Clicked ouside axes bounds but inside plot window')

    def update_plot(self, save=False):

        # mid_time = pd.to_datetime('2018-11-21 11:10:00')

        # self.av_window = pd.to_timedelta('00:00:05')

        start_time = self.mid_time - self.av_window / 2
        end_time = self.mid_time + self.av_window / 2
        u = pd.to_datetime(self.u)
        timeind = np.argwhere((u > start_time) & (u < end_time))

        psd_nims = len(timeind)
        if psd_nims < 1:
            plt.sca(self.axispsd)
            plt.cla()

            plt.sca(self.axistext)

            string = ''
            string += '\n Num images: {:0.0f}'.format(psd_nims)
            string += '\n Start: ' + str(start_time)
            string += '\n End: ' + str(end_time)
            string += '\n Window [sec.] {:0.0f}:'.format((end_time - start_time).seconds)

            plt.title(string, verticalalignment='top', horizontalalignment='right', loc='right')

            plt.sca(self.axisconstant)
            self.line1.remove()
            self.line2.remove()
            self.line1 = plt.vlines(start_time, 1, 12000, 'r', linestyle='--')
            self.line2 = plt.vlines(end_time, 1, 12000, 'r', linestyle='--')
            self.canvas.draw()
            return

        psd_start = min(u[timeind])
        psd_end = max(u[timeind])

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

        plt.sca(self.axispsd)
        plt.cla()
        plt.plot(self.dias, psd_total, 'k', linewidth=5, label='Total')
        plt.plot(self.dias, psd_oil, color=[0.7, 0.4, 0], label='Oil')
        plt.plot(self.dias, psd_gas, 'b', label='Gas')
        plt.xlabel('ECD [um]')
        plt.ylabel('VD [uL/L]')
        plt.xscale('log')
        plt.xlim(10, 12000)

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
        string += '\n End: ' + str(pd.to_datetime(psd_end[0]))
        string += '\n Window [sec.] {:0.0f}:'.format(pd.to_timedelta(psd_end[0]-psd_start[0]).seconds)

        plt.title(string, verticalalignment='top', horizontalalignment='right', loc='right')

        plt.sca(self.axisconstant)
        self.line1.remove()
        self.line2.remove()
        self.line1 = plt.vlines(pd.to_datetime(psd_start[0]), 1, 12000, 'r')
        self.line2 = plt.vlines(pd.to_datetime(psd_end[0]), 1, 12000, 'r')
        self.canvas.draw()

        if save:
            timestring = pd.to_datetime(psd_start[0]).strftime('D%Y%m%dT%H%M%S')
            outputname = self.stats_filename.replace('-STATS.csv','-PSD-' + timestring)
            outputname = QFileDialog.getSaveFileName(self,
                                                   "Select file to Save", outputname,
                                                   ".xlsx")
            if outputname[1]=='':
                print('Did not recieve filename')
                return
            outputname = outputname[0] + outputname[1]
            print(outputname)

            from openpyxl import Workbook
            wb = Workbook()
            ws = wb.active
            ws['A1'] = 'Start:'
            ws['B1'] = min(u)
            ws['A2'] = 'Weighted average:'
            ws['B2'] = 'NOT IMPLEMENTED'
            ws['A3'] = 'End:'
            ws['B3'] = max(u)

            ws['A5'] = 'Number of images:'
            ws['B5'] = psd_nims

            ws['D5'] = 'd50(microns):'
            ws['E5'] = psd_d50_total
            ws['A6'] = 'Number of particles:'
            ws['B6'] = 'NOT IMPLEMENTED'
            ws['D6'] = 'peak || modal size class (microns):'
            ws['E6'] = 'NOT IMPLEMENTED'

            ws['D13'] = 'd50(microns):'
            ws['E13'] = psd_d50_oil
            ws['D14'] = 'peak || modal size class (microns):'
            ws['E14'] = 'NOT IMPLEMENTED'

            ws['D21'] = 'd50(microns):'
            ws['E21'] = psd_d50_gas
            ws['D22'] = 'peak || modal size class (microns):'
            ws['E22'] = 'NOT IMPLEMENTED'


            ws['A8'] = 'Bin mid-sizes (microns):'
            ws['A9'] = 'Vol. Conc. / bin (uL/L):'
            ws['A16'] = 'Vol. Conc. / bin (uL/L):'
            ws['A24'] = 'Vol. Conc. / bin (uL/L):'
            ws['A12'] = 'OIL Info'
            ws['A20'] = 'GAS Info'
            # d = ws.cells(row='8')
            for c in range(len(self.dias)):
                ws.cell(row=8, column=c + 2, value=self.dias[c])
                ws.cell(row=9, column=c + 2, value=psd_total[c])
                ws.cell(row=16, column=c + 2, value=psd_oil[c])
                ws.cell(row=24, column=c + 2, value=psd_gas[c])


            wb.save(outputname)
            print('Saved:', outputname)

    def save_data(self):
        self.update_plot(save=True)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractivePlotter()
    window.show()
    app.exec_()