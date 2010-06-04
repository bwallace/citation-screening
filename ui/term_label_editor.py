from PyQt4.Qt import *
from PyQt4 import QtCore, QtGui
import pdb
import ui_term_label_editor
from abstrackr import Annotation

class TermLabelEditor(QDialog, ui_term_label_editor.Ui_term_label_editor):
        def __init__(self, annotations, session, parent=None):
            super(TermLabelEditor, self).__init__(parent)
            self.setupUi(self)
            self.session = session
            self.current_term = None
            self.annotations = annotations
            self.parent = parent
            self.populate_term_list()
            self.remove_term_button.enabled = False
            QObject.connect(self.term_list, SIGNAL("currentTextChanged(QString)"),
                                            self.term_selected)
            QObject.connect(self.remove_term_button, SIGNAL("pressed()"),
                                            self.remove_selected_term)
            QObject.connect(self.ok_button, SIGNAL("pressed()"),
                                            self.ok)
                      
        def ok(self):
            self.close()
        
        def populate_term_list(self):
            self.term_list.clear()
            for annotation in self.annotations:
                self.term_list.addItem(QString(annotation.text))
        
        def term_selected(self, term):
            if term == "" or term is None:
                pyqtRemoveInputHook()
                pdb.set_trace()
                self.term_list.setCurrentRow(0)
                self.term_selected(self.term_list.item(0).text())
            self.current_term = term
            print "CURRENT TERM: %s" % term
            self.remove_term_button.enabled = True
            ### TODO unicode (!)
            annotation = self._get_annotation_obj(str(term))
            self.update_label_pic(annotation)
            
        def update_label_pic(self, cur_annotation):
            img_strs = {
                2: "two_thumbs_up.png",
                1: "thumbs_up_48.png",
                -1: "thumbs_down_48.png",
                -2: "two_thumbs_down.png"
            }    


            self.lbl_pic.setPixmap(QtGui.QPixmap(":/buttons/%s" % img_strs[cur_annotation.label]))

        def _get_annotation_obj(self, term):
            for annotation in self.annotations:
                if annotation.text == term:
                    return annotation
         
        def remove_selected_term(self):
            annotation_obj = self._get_annotation_obj(self.current_term)
            self.session.delete(annotation_obj)
            self.session.flush()
            self.annotations = self.parent._get_annotations(self.session)
            self.populate_term_list()