# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'term_label_editor.ui'
#
# Created: Fri May 21 17:01:00 2010
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_term_label_editor(object):
    def setupUi(self, term_label_editor):
        term_label_editor.setObjectName("term_label_editor")
        term_label_editor.resize(436, 312)
        term_label_editor.setMinimumSize(QtCore.QSize(436, 312))
        term_label_editor.setMaximumSize(QtCore.QSize(436, 312))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        term_label_editor.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/buttons/abstrackr.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        term_label_editor.setWindowIcon(icon)
        self.term_list = QtGui.QListWidget(term_label_editor)
        self.term_list.setGeometry(QtCore.QRect(10, 10, 241, 231))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.term_list.setFont(font)
        self.term_list.setObjectName("term_list")
        self.label = QtGui.QLabel(term_label_editor)
        self.label.setGeometry(QtCore.QRect(280, 10, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lbl_pic = QtGui.QLabel(term_label_editor)
        self.lbl_pic.setGeometry(QtCore.QRect(300, 40, 71, 81))
        self.lbl_pic.setText("")
        self.lbl_pic.setPixmap(QtGui.QPixmap(":/buttons/two_thumbs_up.png"))
        self.lbl_pic.setObjectName("lbl_pic")
        self.remove_term_button = QtGui.QPushButton(term_label_editor)
        self.remove_term_button.setGeometry(QtCore.QRect(10, 250, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(20)
        self.remove_term_button.setFont(font)
        self.remove_term_button.setIconSize(QtCore.QSize(64, 64))
        self.remove_term_button.setObjectName("remove_term_button")
        self.add_term_button = QtGui.QPushButton(term_label_editor)
        self.add_term_button.setGeometry(QtCore.QRect(50, 250, 31, 31))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(16)
        self.add_term_button.setFont(font)
        self.add_term_button.setIconSize(QtCore.QSize(64, 64))
        self.add_term_button.setObjectName("add_term_button")
        self.ok_button = QtGui.QPushButton(term_label_editor)
        self.ok_button.setGeometry(QtCore.QRect(370, 270, 51, 31))
        font = QtGui.QFont()
        font.setFamily("Verdana")
        font.setPointSize(9)
        self.ok_button.setFont(font)
        self.ok_button.setObjectName("ok_button")

        self.retranslateUi(term_label_editor)
        QtCore.QMetaObject.connectSlotsByName(term_label_editor)

    def retranslateUi(self, term_label_editor):
        term_label_editor.setWindowTitle(QtGui.QApplication.translate("term_label_editor", "edit labeled terms", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("term_label_editor", "current term label:", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_term_button.setToolTip(QtGui.QApplication.translate("term_label_editor", "delete term", None, QtGui.QApplication.UnicodeUTF8))
        self.remove_term_button.setText(QtGui.QApplication.translate("term_label_editor", "-", None, QtGui.QApplication.UnicodeUTF8))
        self.add_term_button.setToolTip(QtGui.QApplication.translate("term_label_editor", "add term", None, QtGui.QApplication.UnicodeUTF8))
        self.add_term_button.setText(QtGui.QApplication.translate("term_label_editor", "+", None, QtGui.QApplication.UnicodeUTF8))
        self.ok_button.setText(QtGui.QApplication.translate("term_label_editor", "ok", None, QtGui.QApplication.UnicodeUTF8))

import icons_rc
