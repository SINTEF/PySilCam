# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PathLengthCTRL.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PLAdjust(object):
    def setupUi(self, PLAdjust):
        PLAdjust.setObjectName("PLAdjust")
        PLAdjust.resize(491, 145)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(PLAdjust.sizePolicy().hasHeightForWidth())
        PLAdjust.setSizePolicy(sizePolicy)
        PLAdjust.setMinimumSize(QtCore.QSize(491, 145))
        PLAdjust.setMaximumSize(QtCore.QSize(491, 145))
        self.SET_button = QtWidgets.QPushButton(PLAdjust)
        self.SET_button.setGeometry(QtCore.QRect(200, 110, 85, 27))
        self.SET_button.setObjectName("SET_button")
        self.label = QtWidgets.QLabel(PLAdjust)
        self.label.setGeometry(QtCore.QRect(10, 10, 221, 17))
        self.label.setObjectName("label")
        self.horizontalSlider = QtWidgets.QSlider(PLAdjust)
        self.horizontalSlider.setGeometry(QtCore.QRect(9, 40, 471, 31))
        self.horizontalSlider.setMinimum(5)
        self.horizontalSlider.setMaximum(40)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.horizontalSlider.setTickInterval(5)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.label_2 = QtWidgets.QLabel(PLAdjust)
        self.label_2.setGeometry(QtCore.QRect(0, 70, 31, 17))
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(PLAdjust)
        self.label_3.setGeometry(QtCore.QRect(70, 70, 31, 17))
        self.label_3.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(PLAdjust)
        self.label_4.setGeometry(QtCore.QRect(130, 70, 31, 17))
        self.label_4.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(PLAdjust)
        self.label_5.setGeometry(QtCore.QRect(200, 70, 31, 17))
        self.label_5.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(PLAdjust)
        self.label_6.setGeometry(QtCore.QRect(260, 70, 31, 17))
        self.label_6.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(PLAdjust)
        self.label_7.setGeometry(QtCore.QRect(330, 70, 31, 17))
        self.label_7.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(PLAdjust)
        self.label_8.setGeometry(QtCore.QRect(390, 70, 31, 17))
        self.label_8.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(PLAdjust)
        self.label_9.setGeometry(QtCore.QRect(460, 70, 31, 17))
        self.label_9.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(PLAdjust)
        self.label_10.setGeometry(QtCore.QRect(230, 80, 31, 17))
        self.label_10.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignTop)
        self.label_10.setObjectName("label_10")

        self.retranslateUi(PLAdjust)
        QtCore.QMetaObject.connectSlotsByName(PLAdjust)

    def retranslateUi(self, PLAdjust):
        _translate = QtCore.QCoreApplication.translate
        PLAdjust.setWindowTitle(_translate("PLAdjust", "Path Length Adjuster"))
        self.SET_button.setText(_translate("PLAdjust", "&SET"))
        self.SET_button.setShortcut(_translate("PLAdjust", "S"))
        self.label.setText(_translate("PLAdjust", "TextLabel"))
        self.label_2.setText(_translate("PLAdjust", "5"))
        self.label_3.setText(_translate("PLAdjust", "10   "))
        self.label_4.setText(_translate("PLAdjust", "15"))
        self.label_5.setText(_translate("PLAdjust", "20   "))
        self.label_6.setText(_translate("PLAdjust", "25"))
        self.label_7.setText(_translate("PLAdjust", "30  "))
        self.label_8.setText(_translate("PLAdjust", "35"))
        self.label_9.setText(_translate("PLAdjust", "40"))
        self.label_10.setText(_translate("PLAdjust", "[mm]"))

