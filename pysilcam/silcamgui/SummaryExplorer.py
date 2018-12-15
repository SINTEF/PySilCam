from PyQt5.QtWidgets import (QMainWindow, QApplication)
from PyQt5 import QtWidgets



class InterativePlotter(QMainWindow):
    def __init__(self, parent=None):
        super(InterativePlotter, self).__init__(parent)
        self.showMaximized()
        self.setWindowTitle("SummaryExplorer")
        QApplication.processEvents()
        self.fft_frame = FftFrame(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.fft_frame)
        self.setLayout(self.layout)
        self.setCentralWidget(self.fft_frame)
        self.showMaximized()


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

        self.fig, self.axes = plt.subplots(1,2)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.canvas)
        self.layout.setStretchFactor(self.canvas, 1)
        self.setLayout(self.layout)

        self.canvas.show()