# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SilCamInterTest(object):
    def setupUi(self, SilCamInterTest):
        SilCamInterTest.setObjectName("SilCamInterTest")
        SilCamInterTest.resize(895, 716)
        self.Data = QtWidgets.QWidget()
        self.Data.setObjectName("Data")
        self.pb_datadir = QtWidgets.QPushButton(self.Data)
        self.pb_datadir.setGeometry(QtCore.QRect(20, 10, 111, 27))
        self.pb_datadir.setObjectName("pb_datadir")
        self.frame = QtWidgets.QFrame(self.Data)
        self.frame.setGeometry(QtCore.QRect(20, 50, 731, 521))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        SilCamInterTest.addTab(self.Data, "")
        self.Control = QtWidgets.QWidget()
        self.Control.setObjectName("Control")
        SilCamInterTest.addTab(self.Control, "")
        self.Config = QtWidgets.QWidget()
        self.Config.setObjectName("Config")
        SilCamInterTest.addTab(self.Config, "")

        self.retranslateUi(SilCamInterTest)
        SilCamInterTest.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(SilCamInterTest)

    def retranslateUi(self, SilCamInterTest):
        _translate = QtCore.QCoreApplication.translate
        SilCamInterTest.setWindowTitle(_translate("SilCamInterTest", "TabWidget"))
        self.pb_datadir.setText(_translate("SilCamInterTest", "Plot raw stats"))
        SilCamInterTest.setTabText(SilCamInterTest.indexOf(self.Data), _translate("SilCamInterTest", "Data"))
        SilCamInterTest.setTabText(SilCamInterTest.indexOf(self.Control), _translate("SilCamInterTest", "Control"))
        SilCamInterTest.setTabText(SilCamInterTest.indexOf(self.Config), _translate("SilCamInterTest", "Config"))

