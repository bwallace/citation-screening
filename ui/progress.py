from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui
import pdb
import ui_progress

class Progress(QDialog, ui_progress.Ui_progress):
        def __init__(self, labels, parent=None):
            super(Progress, self).__init__(parent)
            self.setupUi(self)
            self.labels = labels
            self.populate_data()
            QObject.connect(self.ok_button, SIGNAL("pressed()"),
                                            self.ok)
                      
        def ok(self):
            self.close()
            
        def populate_data(self):
            #pyqtRemoveInputHook()
            #pdb.set_trace()
            self.lbl_total.setText(str(len(self.labels)))
            have_labels = len([x for x in self.labels if x.label is not None])
            self.lbl_so_far.setText(str(have_labels))
            pos = len([x for x in self.labels if x.label == 1])
            self.lbl_pos.setText(str(pos))
            neg = len([x for x in self.labels if x.label == -1])
            self.lbl_neg.setText(str(neg))
            self.lbl_maybe.setText(str(have_labels - (pos+neg)))
            