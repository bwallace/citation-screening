from PyQt4.Qt import *
import pdb
import ui_term_label_editor

class TermLabelEditor(QDialog, ui_term_label_editor.Ui_term_label_editor):
        def __init__(self, parent=None):
            super(TermLabelEditor, self).__init__(parent)
            self.setupUi(self)