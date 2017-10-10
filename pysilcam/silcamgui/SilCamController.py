# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SilCamController.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SilCamController(object):
    def setupUi(self, SilCamController):
        SilCamController.setObjectName("SilCamController")
        SilCamController.resize(191, 205)
        SilCamController.setWindowOpacity(0.9)
        self.pb_start = QtWidgets.QPushButton(SilCamController)
        self.pb_start.setGeometry(QtCore.QRect(10, 10, 168, 41))
        self.pb_start.setAutoDefault(False)
        self.pb_start.setDefault(False)
        self.pb_start.setFlat(False)
        self.pb_start.setObjectName("pb_start")
        self.pb_stop = QtWidgets.QPushButton(SilCamController)
        self.pb_stop.setGeometry(QtCore.QRect(10, 60, 168, 41))
        self.pb_stop.setAutoDefault(False)
        self.pb_stop.setDefault(False)
        self.pb_stop.setFlat(False)
        self.pb_stop.setObjectName("pb_stop")
        self.pb_live_raw = QtWidgets.QPushButton(SilCamController)
        self.pb_live_raw.setGeometry(QtCore.QRect(10, 110, 168, 41))
        self.pb_live_raw.setAutoDefault(False)
        self.pb_live_raw.setDefault(False)
        self.pb_live_raw.setFlat(False)
        self.pb_live_raw.setObjectName("pb_live_raw")
        self.pb_DriveMonitor = QtWidgets.QPushButton(SilCamController)
        self.pb_DriveMonitor.setGeometry(QtCore.QRect(10, 160, 168, 41))
        self.pb_DriveMonitor.setAutoDefault(False)
        self.pb_DriveMonitor.setDefault(False)
        self.pb_DriveMonitor.setFlat(False)
        self.pb_DriveMonitor.setObjectName("pb_DriveMonitor")

        self.retranslateUi(SilCamController)
        QtCore.QMetaObject.connectSlotsByName(SilCamController)

    def retranslateUi(self, SilCamController):
        _translate = QtCore.QCoreApplication.translate
        SilCamController.setWindowTitle(_translate("SilCamController", "Acquisition control"))
        self.pb_start.setText(_translate("SilCamController", "START"))
        self.pb_stop.setText(_translate("SilCamController", "STOP"))
        self.pb_live_raw.setText(_translate("SilCamController", "Live raw"))
        self.pb_DriveMonitor.setText(_translate("SilCamController", "Drive monitor"))

