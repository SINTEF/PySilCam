# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SilCam.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SilCam(object):
    def setupUi(self, SilCam):
        SilCam.setObjectName("SilCam")
        SilCam.resize(814, 703)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SilCam.sizePolicy().hasHeightForWidth())
        SilCam.setSizePolicy(sizePolicy)
        SilCam.setWindowTitle("SilCam")
        SilCam.setStatusTip("")
        self.centralwidget = QtWidgets.QWidget(SilCam)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.statusBar = QtWidgets.QLabel(self.centralwidget)
        self.statusBar.setObjectName("statusBar")
        self.gridLayout.addWidget(self.statusBar, 0, 1, 1, 1, QtCore.Qt.AlignTop)
        self.fig_widget = QtWidgets.QWidget(self.centralwidget)
        self.fig_widget.setMinimumSize(QtCore.QSize(790, 400))
        self.fig_widget.setObjectName("fig_widget")
        self.gridLayout.addWidget(self.fig_widget, 1, 0, 1, 2)
        SilCam.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SilCam)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 814, 27))
        self.menubar.setObjectName("menubar")
        self.menuProcessing = QtWidgets.QMenu(self.menubar)
        self.menuProcessing.setEnabled(True)
        self.menuProcessing.setObjectName("menuProcessing")
        self.menuExport = QtWidgets.QMenu(self.menuProcessing)
        self.menuExport.setObjectName("menuExport")
        SilCam.setMenuBar(self.menubar)
        self.actionServer = QtWidgets.QAction(SilCam)
        self.actionServer.setObjectName("actionServer")
        self.actionConvert_silc_to_bmp = QtWidgets.QAction(SilCam)
        self.actionConvert_silc_to_bmp.setObjectName("actionConvert_silc_to_bmp")
        self.actionExport_summary_figure = QtWidgets.QAction(SilCam)
        self.actionExport_summary_figure.setObjectName("actionExport_summary_figure")
        self.actionSilc_file_player = QtWidgets.QAction(SilCam)
        self.actionSilc_file_player.setObjectName("actionSilc_file_player")
        self.actionEditConfig = QtWidgets.QAction(SilCam)
        self.actionEditConfig.setObjectName("actionEditConfig")
        self.actionPath_length_adjuster = QtWidgets.QAction(SilCam)
        self.actionPath_length_adjuster.setObjectName("actionPath_length_adjuster")
        self.actionExport_summary_data = QtWidgets.QAction(SilCam)
        self.actionExport_summary_data.setObjectName("actionExport_summary_data")
        self.menuExport.addAction(self.actionExport_summary_data)
        self.menuExport.addAction(self.actionExport_summary_figure)
        self.menuExport.addAction(self.actionConvert_silc_to_bmp)
        self.menuProcessing.addAction(self.actionEditConfig)
        self.menuProcessing.addAction(self.menuExport.menuAction())
        self.menuProcessing.addAction(self.actionSilc_file_player)
        self.menuProcessing.addAction(self.actionServer)
        self.menuProcessing.addAction(self.actionPath_length_adjuster)
        self.menubar.addAction(self.menuProcessing.menuAction())

        self.retranslateUi(SilCam)
        QtCore.QMetaObject.connectSlotsByName(SilCam)

    def retranslateUi(self, SilCam):
        _translate = QtCore.QCoreApplication.translate
        self.statusBar.setText(_translate("SilCam", "STATUS"))
        self.menuProcessing.setTitle(_translate("SilCam", "Tools"))
        self.menuExport.setTitle(_translate("SilCam", "Export"))
        self.actionServer.setText(_translate("SilCam", "Realtime data server"))
        self.actionConvert_silc_to_bmp.setText(_translate("SilCam", "Raw data convert silc to bmp"))
        self.actionExport_summary_figure.setText(_translate("SilCam", "Summary figure (to png)"))
        self.actionSilc_file_player.setText(_translate("SilCam", "Silc file player"))
        self.actionEditConfig.setText(_translate("SilCam", "Edit config file"))
        self.actionPath_length_adjuster.setText(_translate("SilCam", "Path length adjuster"))
        self.actionExport_summary_data.setText(_translate("SilCam", "Summary time series (to xls, png)"))

