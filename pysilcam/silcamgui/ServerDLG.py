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
        Server.resize(310, 137)
        self.verticalLayout = QtWidgets.QVBoxLayout(Server)
        self.verticalLayout.setObjectName("verticalLayout")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Server)
        self.plainTextEdit.setMaximumSize(QtCore.QSize(256, 110))
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.plainTextEdit, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.Start = QtWidgets.QPushButton(Server)
        self.Start.setObjectName("Start")
        self.verticalLayout.addWidget(self.Start, 0, QtCore.Qt.AlignHCenter|QtCore.Qt.AlignBottom)

        self.retranslateUi(Server)
        QtCore.QMetaObject.connectSlotsByName(Server)

    def retranslateUi(self, Server):
        _translate = QtCore.QCoreApplication.translate
        Server.setWindowTitle(_translate("Server", "Dialog"))
        self.Start.setText(_translate("Server", "Start"))

