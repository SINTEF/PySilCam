import sys
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import (QMainWindow, QApplication, QPushButton, QWidget,
QAction, QTabWidget,QVBoxLayout, QFileDialog)
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from test import Ui_SilCamInterTest

DATADIR = '/mnt/ARRAY/ENTICE/Data'


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


class StartQT5(QTabWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = Ui_SilCamInterTest()
        self.ui.setupUi(self)

        self.datadir = DATADIR
        self.nims = 0

        #GUI configuration
        self.figure, self.a = plt.subplots(1,2,figsize=(10,5))
        self.resize(800,480)
        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.ui.frame.setLayout(layout)

        self.ui.pb_datadir.clicked.connect(self.button_test)


    def button_test(self):
        print(self.nims)
        self.datadir=QFileDialog.getExistingDirectory(self,'open',self.datadir,QFileDialog.ShowDirsOnly)
        if self.datadir == '':
            self.datadir = DATADIR
            return
        files = [os.path.join(self.datadir, f) for f in
                sorted(os.listdir(self.datadir))
                if f.endswith('.bmp')]
        self.nims = len(files)
        times = names_to_times(files)

        plt.sca(self.a[0])
        plt.cla()
        plt.plot(times,np.arange(0,self.nims),'k.',markersize=2)
        plt.title(self.datadir + ': ' + str(self.nims) + ' images')

        hz = times_to_hz(times)
        plt.sca(self.a[1])
        plt.cla()
        plt.hist(hz,bins=np.arange(0,7,0.1),color='k')

        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myapp = StartQT5()
    myapp.show()
    sys.exit(app.exec_())
