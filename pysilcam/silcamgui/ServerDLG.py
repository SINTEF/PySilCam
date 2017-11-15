# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ServerDLG.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Server(object):
    def setupUi(self, Server):
        Server.setObjectName("Server")
        Server.resize(310, 201)
        self.Start = QtWidgets.QPushButton(Server)
        self.Start.setGeometry(QtCore.QRect(90, 120, 141, 71))
        self.Start.setObjectName("Start")
        self.IPText = QtWidgets.QPlainTextEdit(Server)
        self.IPText.setGeometry(QtCore.QRect(60, 50, 191, 31))
        self.IPText.setObjectName("IPText")

        self.retranslateUi(Server)
        QtCore.QMetaObject.connectSlotsByName(Server)

    def retranslateUi(self, Server):
        _translate = QtCore.QCoreApplication.translate
        Server.setWindowTitle(_translate("Server", "HTTP Server"))
        self.Start.setText(_translate("Server", "Start"))

