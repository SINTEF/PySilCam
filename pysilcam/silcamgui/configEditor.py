# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'configEditor.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Editconfig(object):
    def setupUi(self, Editconfig):
        Editconfig.setObjectName("Editconfig")
        Editconfig.resize(767, 667)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Editconfig.sizePolicy().hasHeightForWidth())
        Editconfig.setSizePolicy(sizePolicy)
        Editconfig.setMinimumSize(QtCore.QSize(695, 618))
        Editconfig.setModal(True)
        self.layoutWidget = QtWidgets.QWidget(Editconfig)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 16, 741, 541))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_5.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.generalgroupBox = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.generalgroupBox.sizePolicy().hasHeightForWidth())
        self.generalgroupBox.setSizePolicy(sizePolicy)
        self.generalgroupBox.setMaximumSize(QtCore.QSize(1000, 500))
        self.generalgroupBox.setObjectName("generalgroupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.generalgroupBox)
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.datafileEdit = QtWidgets.QLineEdit(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.datafileEdit.sizePolicy().hasHeightForWidth())
        self.datafileEdit.setSizePolicy(sizePolicy)
        self.datafileEdit.setObjectName("datafileEdit")
        self.gridLayout.addWidget(self.datafileEdit, 0, 1, 1, 1)
        self.browseDataPathPB = QtWidgets.QPushButton(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browseDataPathPB.sizePolicy().hasHeightForWidth())
        self.browseDataPathPB.setSizePolicy(sizePolicy)
        self.browseDataPathPB.setMaximumSize(QtCore.QSize(20, 20))
        self.browseDataPathPB.setObjectName("browseDataPathPB")
        self.gridLayout.addWidget(self.browseDataPathPB, 0, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 3, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.generalgroupBox)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)
        self.loglevelEdit = QtWidgets.QComboBox(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.loglevelEdit.sizePolicy().hasHeightForWidth())
        self.loglevelEdit.setSizePolicy(sizePolicy)
        self.loglevelEdit.setObjectName("loglevelEdit")
        self.loglevelEdit.addItem("")
        self.loglevelEdit.addItem("")
        self.loglevelEdit.addItem("")
        self.loglevelEdit.addItem("")
        self.loglevelEdit.addItem("")
        self.loglevelEdit.addItem("")
        self.gridLayout.addWidget(self.loglevelEdit, 1, 1, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_19.sizePolicy().hasHeightForWidth())
        self.label_19.setSizePolicy(sizePolicy)
        self.label_19.setObjectName("label_19")
        self.gridLayout.addWidget(self.label_19, 1, 3, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.generalgroupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 2, 0, 1, 1)
        self.logfileEdit = QtWidgets.QLineEdit(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.logfileEdit.sizePolicy().hasHeightForWidth())
        self.logfileEdit.setSizePolicy(sizePolicy)
        self.logfileEdit.setObjectName("logfileEdit")
        self.gridLayout.addWidget(self.logfileEdit, 2, 1, 1, 1)
        self.browseLogFilePB = QtWidgets.QPushButton(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.browseLogFilePB.sizePolicy().hasHeightForWidth())
        self.browseLogFilePB.setSizePolicy(sizePolicy)
        self.browseLogFilePB.setMaximumSize(QtCore.QSize(20, 20))
        self.browseLogFilePB.setObjectName("browseLogFilePB")
        self.gridLayout.addWidget(self.browseLogFilePB, 2, 2, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.generalgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 2, 3, 1, 1)
        self.gridLayout.setColumnMinimumWidth(0, 120)
        self.gridLayout.setColumnMinimumWidth(1, 180)
        self.gridLayout.setColumnMinimumWidth(2, 25)
        self.gridLayout.setColumnMinimumWidth(3, 320)
        self.gridLayout_5.addWidget(self.generalgroupBox, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_4.setObjectName("gridLayout_4")
        spacerItem = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem, 0, 2, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setObjectName("label_17")
        self.gridLayout_4.addWidget(self.label_17, 0, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 1, 0, 1, 1)
        self.outputpathEdit = QtWidgets.QLineEdit(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.outputpathEdit.sizePolicy().hasHeightForWidth())
        self.outputpathEdit.setSizePolicy(sizePolicy)
        self.outputpathEdit.setObjectName("outputpathEdit")
        self.gridLayout_4.addWidget(self.outputpathEdit, 1, 1, 1, 1)
        self.browseOutputPathPB = QtWidgets.QPushButton(self.groupBox_2)
        self.browseOutputPathPB.setMaximumSize(QtCore.QSize(20, 20))
        self.browseOutputPathPB.setObjectName("browseOutputPathPB")
        self.gridLayout_4.addWidget(self.browseOutputPathPB, 1, 2, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_18.sizePolicy().hasHeightForWidth())
        self.label_18.setSizePolicy(sizePolicy)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 1, 3, 1, 1)
        self.export_imagesEdit = QtWidgets.QComboBox(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.export_imagesEdit.sizePolicy().hasHeightForWidth())
        self.export_imagesEdit.setSizePolicy(sizePolicy)
        self.export_imagesEdit.setObjectName("export_imagesEdit")
        self.export_imagesEdit.addItem("")
        self.export_imagesEdit.addItem("")
        self.gridLayout_4.addWidget(self.export_imagesEdit, 0, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setObjectName("label_7")
        self.gridLayout_4.addWidget(self.label_7, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 2, 0, 1, 1)
        self.min_lengthEdit = QtWidgets.QLineEdit(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.min_lengthEdit.sizePolicy().hasHeightForWidth())
        self.min_lengthEdit.setSizePolicy(sizePolicy)
        self.min_lengthEdit.setObjectName("min_lengthEdit")
        self.gridLayout_4.addWidget(self.min_lengthEdit, 2, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 2, 3, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_4.addItem(spacerItem1, 2, 2, 1, 1)
        self.gridLayout_4.setColumnMinimumWidth(0, 120)
        self.gridLayout_4.setColumnMinimumWidth(1, 180)
        self.gridLayout_4.setColumnMinimumWidth(2, 25)
        self.gridLayout_4.setColumnMinimumWidth(3, 320)
        self.gridLayout_5.addWidget(self.groupBox_2, 4, 0, 1, 1)
        self.processgroupBox = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.processgroupBox.sizePolicy().hasHeightForWidth())
        self.processgroupBox.setSizePolicy(sizePolicy)
        self.processgroupBox.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.processgroupBox.setObjectName("processgroupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.processgroupBox)
        self.gridLayout_2.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_4 = QtWidgets.QLabel(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setObjectName("label_4")
        self.gridLayout_2.addWidget(self.label_4, 1, 3, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.processgroupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 3, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 1, 2, 1, 1)
        self.real_time_statsEdit = QtWidgets.QComboBox(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.real_time_statsEdit.sizePolicy().hasHeightForWidth())
        self.real_time_statsEdit.setSizePolicy(sizePolicy)
        self.real_time_statsEdit.setObjectName("real_time_statsEdit")
        self.real_time_statsEdit.addItem("")
        self.real_time_statsEdit.addItem("")
        self.gridLayout_2.addWidget(self.real_time_statsEdit, 0, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.processgroupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 0, 0, 1, 1)
        self.window_sizeEdit = QtWidgets.QLineEdit(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_sizeEdit.sizePolicy().hasHeightForWidth())
        self.window_sizeEdit.setSizePolicy(sizePolicy)
        self.window_sizeEdit.setObjectName("window_sizeEdit")
        self.gridLayout_2.addWidget(self.window_sizeEdit, 3, 1, 1, 1)
        self.path_lengthEdit = QtWidgets.QLineEdit(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.path_lengthEdit.sizePolicy().hasHeightForWidth())
        self.path_lengthEdit.setSizePolicy(sizePolicy)
        self.path_lengthEdit.setObjectName("path_lengthEdit")
        self.gridLayout_2.addWidget(self.path_lengthEdit, 1, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 0, 3, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.processgroupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.processgroupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_15.sizePolicy().hasHeightForWidth())
        self.label_15.setSizePolicy(sizePolicy)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 3, 3, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 3, 2, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.processgroupBox)
        self.label_25.setObjectName("label_25")
        self.gridLayout_2.addWidget(self.label_25, 2, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem5, 2, 2, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.processgroupBox)
        self.label_26.setObjectName("label_26")
        self.gridLayout_2.addWidget(self.label_26, 2, 3, 1, 1)
        self.com_portEdit = QtWidgets.QComboBox(self.processgroupBox)
        self.com_portEdit.setObjectName("com_portEdit")
        self.gridLayout_2.addWidget(self.com_portEdit, 2, 1, 1, 1)
        self.gridLayout_2.setColumnMinimumWidth(0, 120)
        self.gridLayout_2.setColumnMinimumWidth(1, 180)
        self.gridLayout_2.setColumnMinimumWidth(2, 25)
        self.gridLayout_2.setColumnMinimumWidth(3, 320)
        self.gridLayout_5.addWidget(self.processgroupBox, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setMaximumSize(QtCore.QSize(1000, 16777215))
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_22 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_22.sizePolicy().hasHeightForWidth())
        self.label_22.setSizePolicy(sizePolicy)
        self.label_22.setObjectName("label_22")
        self.gridLayout_3.addWidget(self.label_22, 0, 3, 1, 1)
        self.num_imagesEdit = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.num_imagesEdit.sizePolicy().hasHeightForWidth())
        self.num_imagesEdit.setSizePolicy(sizePolicy)
        self.num_imagesEdit.setObjectName("num_imagesEdit")
        self.gridLayout_3.addWidget(self.num_imagesEdit, 0, 1, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 0, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_20.sizePolicy().hasHeightForWidth())
        self.label_20.setSizePolicy(sizePolicy)
        self.label_20.setObjectName("label_20")
        self.gridLayout_3.addWidget(self.label_20, 1, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.groupBox)
        self.label_21.setObjectName("label_21")
        self.gridLayout_3.addWidget(self.label_21, 2, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem6, 0, 2, 1, 1)
        self.thresholdEdit = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.thresholdEdit.sizePolicy().hasHeightForWidth())
        self.thresholdEdit.setSizePolicy(sizePolicy)
        self.thresholdEdit.setObjectName("thresholdEdit")
        self.gridLayout_3.addWidget(self.thresholdEdit, 1, 1, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem7, 1, 2, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_23.sizePolicy().hasHeightForWidth())
        self.label_23.setSizePolicy(sizePolicy)
        self.label_23.setObjectName("label_23")
        self.gridLayout_3.addWidget(self.label_23, 1, 3, 1, 1)
        self.max_particlesEdit = QtWidgets.QLineEdit(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.max_particlesEdit.sizePolicy().hasHeightForWidth())
        self.max_particlesEdit.setSizePolicy(sizePolicy)
        self.max_particlesEdit.setObjectName("max_particlesEdit")
        self.gridLayout_3.addWidget(self.max_particlesEdit, 2, 1, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_3.addItem(spacerItem8, 2, 2, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_24.sizePolicy().hasHeightForWidth())
        self.label_24.setSizePolicy(sizePolicy)
        self.label_24.setObjectName("label_24")
        self.gridLayout_3.addWidget(self.label_24, 2, 3, 1, 1)
        self.gridLayout_3.setColumnMinimumWidth(0, 120)
        self.gridLayout_3.setColumnMinimumWidth(1, 180)
        self.gridLayout_3.setColumnMinimumWidth(2, 25)
        self.gridLayout_3.setColumnMinimumWidth(3, 320)
        self.gridLayout_5.addWidget(self.groupBox, 5, 0, 1, 1)
        self.configPathLabel = QtWidgets.QLabel(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.configPathLabel.sizePolicy().hasHeightForWidth())
        self.configPathLabel.setSizePolicy(sizePolicy)
        self.configPathLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.configPathLabel.setObjectName("configPathLabel")
        self.gridLayout_5.addWidget(self.configPathLabel, 0, 0, 1, 1)
        self.layoutWidget1 = QtWidgets.QWidget(Editconfig)
        self.layoutWidget1.setGeometry(QtCore.QRect(550, 580, 201, 72))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.defaultPushButton = QtWidgets.QPushButton(self.layoutWidget1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.defaultPushButton.sizePolicy().hasHeightForWidth())
        self.defaultPushButton.setSizePolicy(sizePolicy)
        self.defaultPushButton.setMinimumSize(QtCore.QSize(130, 0))
        self.defaultPushButton.setObjectName("defaultPushButton")
        self.gridLayout_6.addWidget(self.defaultPushButton, 0, 1, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(self.layoutWidget1)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout_6.addWidget(self.buttonBox, 2, 0, 1, 2)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_6.addItem(spacerItem9, 0, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem10, 1, 0, 1, 1)

        self.retranslateUi(Editconfig)
        self.buttonBox.accepted.connect(Editconfig.accept)
        self.buttonBox.rejected.connect(Editconfig.reject)
        QtCore.QMetaObject.connectSlotsByName(Editconfig)

    def retranslateUi(self, Editconfig):
        _translate = QtCore.QCoreApplication.translate
        Editconfig.setWindowTitle(_translate("Editconfig", "Edit configuration file"))
        self.generalgroupBox.setTitle(_translate("Editconfig", "General"))
        self.label.setText(_translate("Editconfig", "Output data path:"))
        self.browseDataPathPB.setText(_translate("Editconfig", "..."))
        self.label_10.setText(_translate("Editconfig", "Location to store processed data."))
        self.label_8.setText(_translate("Editconfig", "Log level:"))
        self.loglevelEdit.setItemText(0, _translate("Editconfig", "INFO"))
        self.loglevelEdit.setItemText(1, _translate("Editconfig", "DEBUG"))
        self.loglevelEdit.setItemText(2, _translate("Editconfig", "WARNING"))
        self.loglevelEdit.setItemText(3, _translate("Editconfig", "ERROR"))
        self.loglevelEdit.setItemText(4, _translate("Editconfig", "CRITICAL"))
        self.loglevelEdit.setItemText(5, _translate("Editconfig", "NOTSET"))
        self.label_19.setText(_translate("Editconfig", "Level of detail to log."))
        self.label_6.setText(_translate("Editconfig", "Log file:"))
        self.browseLogFilePB.setText(_translate("Editconfig", "..."))
        self.label_16.setText(_translate("Editconfig", "Location of the log file."))
        self.groupBox_2.setTitle(_translate("Editconfig", "Export"))
        self.label_17.setText(_translate("Editconfig", "Save particles images to disk."))
        self.label_9.setText(_translate("Editconfig", "Output folder:"))
        self.browseOutputPathPB.setText(_translate("Editconfig", "..."))
        self.label_18.setText(_translate("Editconfig", "Folder to store the particles images."))
        self.export_imagesEdit.setItemText(0, _translate("Editconfig", "False"))
        self.export_imagesEdit.setItemText(1, _translate("Editconfig", "True"))
        self.label_7.setText(_translate("Editconfig", "Export images:"))
        self.label_11.setText(_translate("Editconfig", "Minimum length:"))
        self.label_12.setText(_translate("Editconfig", "[µm] Minimum length of particles to save to disk."))
        self.processgroupBox.setTitle(_translate("Editconfig", "Process"))
        self.label_4.setText(_translate("Editconfig", "[mm] Distance between camera and illumination."))
        self.label_5.setText(_translate("Editconfig", "Time average:"))
        self.real_time_statsEdit.setItemText(0, _translate("Editconfig", "False"))
        self.real_time_statsEdit.setItemText(1, _translate("Editconfig", "True"))
        self.label_3.setText(_translate("Editconfig", "Real time stats:"))
        self.label_13.setText(_translate("Editconfig", "Enable statistics to be displayed in  real-time."))
        self.label_2.setText(_translate("Editconfig", "Path length:"))
        self.label_15.setText(_translate("Editconfig", "[seconds] Number of seconds to average for real-time statistics."))
        self.label_25.setText(_translate("Editconfig", "COM port"))
        self.label_26.setText(_translate("Editconfig", "COM port used to transfer the information about the path length."))
        self.groupBox.setTitle(_translate("Editconfig", "Advanced"))
        self.label_22.setText(_translate("Editconfig", "Number of images to use for background correction."))
        self.label_14.setText(_translate("Editconfig", "Number of images:"))
        self.label_20.setText(_translate("Editconfig", "Threshold:"))
        self.label_21.setText(_translate("Editconfig", "Max particles:"))
        self.label_23.setText(_translate("Editconfig", "Detection sensitivity, between 0 and 1."))
        self.label_24.setText(_translate("Editconfig", "Skip analysis if more particles are detected."))
        self.configPathLabel.setText(_translate("Editconfig", "TextLabel"))
        self.defaultPushButton.setText(_translate("Editconfig", "Default configuration"))
