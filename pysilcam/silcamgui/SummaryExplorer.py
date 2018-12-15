# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SummaryExplorer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SummaryExplorer(object):
    def setupUi(self, SummaryExplorer):
        SummaryExplorer.setObjectName("SummaryExplorer")
        SummaryExplorer.resize(567, 453)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(SummaryExplorer.sizePolicy().hasHeightForWidth())
        SummaryExplorer.setSizePolicy(sizePolicy)
        SummaryExplorer.setMinimumSize(QtCore.QSize(10, 10))
        self.gridLayout = QtWidgets.QGridLayout(SummaryExplorer)
        self.gridLayout.setObjectName("gridLayout")
        self.PLTwidget = QtWidgets.QWidget(SummaryExplorer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PLTwidget.sizePolicy().hasHeightForWidth())
        self.PLTwidget.setSizePolicy(sizePolicy)
        self.PLTwidget.setMinimumSize(QtCore.QSize(411, 231))
        self.PLTwidget.setObjectName("PLTwidget")
        self.gridLayout.addWidget(self.PLTwidget, 0, 0, 1, 1)

        self.retranslateUi(SummaryExplorer)
        QtCore.QMetaObject.connectSlotsByName(SummaryExplorer)

    def retranslateUi(self, SummaryExplorer):
        _translate = QtCore.QCoreApplication.translate
        SummaryExplorer.setWindowTitle(_translate("SummaryExplorer", "Summary Explorer"))

