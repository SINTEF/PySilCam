# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'STATSTrimmer.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_STATSTimeEDT(object):
    def setupUi(self, STATSTimeEDT):
        STATSTimeEDT.setObjectName("STATSTimeEDT")
        STATSTimeEDT.resize(554, 589)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(STATSTimeEDT.sizePolicy().hasHeightForWidth())
        STATSTimeEDT.setSizePolicy(sizePolicy)
        self.dateTimeStart = QtWidgets.QDateTimeEdit(STATSTimeEDT)
        self.dateTimeStart.setGeometry(QtCore.QRect(9, 9, 131, 27))
        self.dateTimeStart.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dateTimeStart.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToPreviousValue)
        self.dateTimeStart.setCalendarPopup(False)
        self.dateTimeStart.setObjectName("dateTimeStart")
        self.label = QtWidgets.QLabel(STATSTimeEDT)
        self.label.setGeometry(QtCore.QRect(150, 9, 111, 17))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(STATSTimeEDT)
        self.label_2.setGeometry(QtCore.QRect(150, 42, 111, 17))
        self.label_2.setObjectName("label_2")
        self.PBSave = QtWidgets.QPushButton(STATSTimeEDT)
        self.PBSave.setGeometry(QtCore.QRect(260, 42, 85, 27))
        self.PBSave.setObjectName("PBSave")
        self.dateTimeEnd = QtWidgets.QDateTimeEdit(STATSTimeEDT)
        self.dateTimeEnd.setGeometry(QtCore.QRect(9, 42, 131, 27))
        self.dateTimeEnd.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.dateTimeEnd.setCorrectionMode(QtWidgets.QAbstractSpinBox.CorrectToPreviousValue)
        self.dateTimeEnd.setCalendarPopup(False)
        self.dateTimeEnd.setObjectName("dateTimeEnd")
        self.PBPlot = QtWidgets.QPushButton(STATSTimeEDT)
        self.PBPlot.setGeometry(QtCore.QRect(260, 9, 85, 27))
        self.PBPlot.setObjectName("PBPlot")
        self.PLTwidget = QtWidgets.QWidget(STATSTimeEDT)
        self.PLTwidget.setGeometry(QtCore.QRect(30, 80, 501, 481))
        self.PLTwidget.setObjectName("PLTwidget")

        self.retranslateUi(STATSTimeEDT)
        QtCore.QMetaObject.connectSlotsByName(STATSTimeEDT)

    def retranslateUi(self, STATSTimeEDT):
        _translate = QtCore.QCoreApplication.translate
        STATSTimeEDT.setWindowTitle(_translate("STATSTimeEDT", "STATS Trimmer"))
        self.dateTimeStart.setDisplayFormat(_translate("STATSTimeEDT", "dd.MM.yy HH:mm:ss"))
        self.label.setText(_translate("STATSTimeEDT", "Start time"))
        self.label_2.setText(_translate("STATSTimeEDT", "End time"))
        self.PBSave.setText(_translate("STATSTimeEDT", "Save"))
        self.dateTimeEnd.setDisplayFormat(_translate("STATSTimeEDT", "dd.MM.yy HH:mm:ss"))
        self.PBPlot.setText(_translate("STATSTimeEDT", "&Plot"))
        self.PBPlot.setShortcut(_translate("STATSTimeEDT", "P"))

