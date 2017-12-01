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
        SilCamController.resize(569, 204)
        SilCamController.setWindowOpacity(0.9)
        self.pb_start = QtWidgets.QPushButton(SilCamController)
        self.pb_start.setGeometry(QtCore.QRect(390, 10, 168, 41))
        self.pb_start.setAutoDefault(False)
        self.pb_start.setDefault(False)
        self.pb_start.setFlat(False)
        self.pb_start.setObjectName("pb_start")
        self.pb_stop = QtWidgets.QPushButton(SilCamController)
        self.pb_stop.setGeometry(QtCore.QRect(390, 60, 168, 41))
        self.pb_stop.setAutoDefault(False)
        self.pb_stop.setDefault(False)
        self.pb_stop.setFlat(False)
        self.pb_stop.setObjectName("pb_stop")
        self.pb_live_raw = QtWidgets.QPushButton(SilCamController)
        self.pb_live_raw.setGeometry(QtCore.QRect(390, 110, 168, 41))
        self.pb_live_raw.setAutoDefault(False)
        self.pb_live_raw.setDefault(False)
        self.pb_live_raw.setFlat(False)
        self.pb_live_raw.setObjectName("pb_live_raw")
        self.line = QtWidgets.QFrame(SilCamController)
        self.line.setGeometry(QtCore.QRect(360, 10, 20, 141))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.rb_to_disc = QtWidgets.QRadioButton(SilCamController)
        self.rb_to_disc.setGeometry(QtCore.QRect(20, 30, 171, 17))
        self.rb_to_disc.setObjectName("rb_to_disc")
        self.rb_process_historical = QtWidgets.QRadioButton(SilCamController)
        self.rb_process_historical.setGeometry(QtCore.QRect(20, 60, 191, 17))
        self.rb_process_historical.setObjectName("rb_process_historical")
        self.le_path_to_data = QtWidgets.QLineEdit(SilCamController)
        self.le_path_to_data.setGeometry(QtCore.QRect(20, 140, 331, 20))
        self.le_path_to_data.setObjectName("le_path_to_data")
        self.pb_browse = QtWidgets.QPushButton(SilCamController)
        self.pb_browse.setGeometry(QtCore.QRect(280, 170, 75, 23))
        self.pb_browse.setObjectName("pb_browse")
        self.rb_real_time = QtWidgets.QRadioButton(SilCamController)
        self.rb_real_time.setGeometry(QtCore.QRect(20, 90, 191, 17))
        self.rb_real_time.setObjectName("rb_real_time")
        self.cb_store_to_disc = QtWidgets.QCheckBox(SilCamController)
        self.cb_store_to_disc.setGeometry(QtCore.QRect(220, 90, 111, 17))
        self.cb_store_to_disc.setObjectName("cb_store_to_disc")

        self.retranslateUi(SilCamController)
        QtCore.QMetaObject.connectSlotsByName(SilCamController)

    def retranslateUi(self, SilCamController):
        _translate = QtCore.QCoreApplication.translate
        SilCamController.setWindowTitle(_translate("SilCamController", "Acquisition control"))
        self.pb_start.setText(_translate("SilCamController", "START"))
        self.pb_stop.setText(_translate("SilCamController", "STOP"))
        self.pb_live_raw.setText(_translate("SilCamController", "Live raw"))
        self.rb_to_disc.setText(_translate("SilCamController", "Aquire images to disk"))
        self.rb_process_historical.setText(_translate("SilCamController", "Process historical data"))
        self.pb_browse.setText(_translate("SilCamController", "Browse"))
        self.rb_real_time.setText(_translate("SilCamController", "Realtime processing"))
        self.cb_store_to_disc.setText(_translate("SilCamController", "Store to disk"))

