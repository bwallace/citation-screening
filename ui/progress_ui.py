# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'progress.ui'
#
# Created: Thu May 27 14:07:40 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_progress(object):
    def setupUi(self, progress):
        progress.setObjectName("progress")
        progress.resize(265, 300)
        progress.setMinimumSize(QtCore.QSize(265, 300))
        progress.setMaximumSize(QtCore.QSize(265, 300))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        progress.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/buttons/abstrackr.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        progress.setWindowIcon(icon)
        self.ok_button = QtGui.QPushButton(progress)
        self.ok_button.setGeometry(QtCore.QRect(200, 260, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.ok_button.setFont(font)
        self.ok_button.setObjectName("ok_button")
        self.groupBox = QtGui.QGroupBox(progress)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 241, 241))
        self.groupBox.setObjectName("groupBox")
        self.label = QtGui.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(20, 40, 101, 21))
        self.label.setObjectName("label")
        self.lbl_total = QtGui.QLabel(self.groupBox)
        self.lbl_total.setGeometry(QtCore.QRect(130, 40, 61, 21))
        self.lbl_total.setText("")
        self.lbl_total.setObjectName("lbl_total")
        self.label_2 = QtGui.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(20, 80, 101, 21))
        self.label_2.setObjectName("label_2")
        self.lbl_so_far = QtGui.QLabel(self.groupBox)
        self.lbl_so_far.setGeometry(QtCore.QRect(130, 80, 61, 21))
        self.lbl_so_far.setText("")
        self.lbl_so_far.setObjectName("lbl_so_far")
        self.lbl_o = QtGui.QLabel(self.groupBox)
        self.lbl_o.setGeometry(QtCore.QRect(20, 120, 101, 21))
        self.lbl_o.setObjectName("lbl_o")
        self.lbl_o_2 = QtGui.QLabel(self.groupBox)
        self.lbl_o_2.setGeometry(QtCore.QRect(20, 160, 101, 21))
        self.lbl_o_2.setObjectName("lbl_o_2")
        self.lbl_pos = QtGui.QLabel(self.groupBox)
        self.lbl_pos.setGeometry(QtCore.QRect(130, 120, 71, 21))
        self.lbl_pos.setText("")
        self.lbl_pos.setObjectName("lbl_pos")
        self.lbl_neg = QtGui.QLabel(self.groupBox)
        self.lbl_neg.setGeometry(QtCore.QRect(130, 160, 61, 21))
        self.lbl_neg.setText("")
        self.lbl_neg.setObjectName("lbl_neg")
        self.lbl_o_3 = QtGui.QLabel(self.groupBox)
        self.lbl_o_3.setGeometry(QtCore.QRect(20, 200, 101, 21))
        self.lbl_o_3.setObjectName("lbl_o_3")
        self.lbl_maybe = QtGui.QLabel(self.groupBox)
        self.lbl_maybe.setGeometry(QtCore.QRect(130, 200, 61, 21))
        self.lbl_maybe.setText("")
        self.lbl_maybe.setObjectName("lbl_maybe")

        self.retranslateUi(progress)
        QtCore.QMetaObject.connectSlotsByName(progress)

    def retranslateUi(self, progress):
        progress.setWindowTitle(QtGui.QApplication.translate("progress", "labeling progress", None, QtGui.QApplication.UnicodeUTF8))
        self.ok_button.setText(QtGui.QApplication.translate("progress", "ok", None, QtGui.QApplication.UnicodeUTF8))
        self.groupBox.setTitle(QtGui.QApplication.translate("progress", "progress", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("progress", "Total documents:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("progress", "Labeled so far:", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_o.setText(QtGui.QApplication.translate("progress", "Labeled positive:", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_o_2.setText(QtGui.QApplication.translate("progress", "Labeled negative:", None, QtGui.QApplication.UnicodeUTF8))
        self.lbl_o_3.setText(QtGui.QApplication.translate("progress", "Labeled maybe:", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
import icons_rc
