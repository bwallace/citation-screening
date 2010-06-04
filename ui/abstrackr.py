import sys
import pdb
import datetime
import os.path
from datetime import datetime
import pickle
from PyQt4 import QtCore, QtGui, Qt
from PyQt4.Qt import *
from PyQt4.QtCore import pyqtRemoveInputHook
# by the way: here's how to set breaks with QT
#     pyqtRemoveInputHook()
#     pdb.set_trace()

# TODO write unit tests ( ... )
import nose # for unit tests
import sqlalchemy
from sqlalchemy import *
from sqlalchemy import orm
from sqlalchemy.orm import create_session
from sqlalchemy import or_
from sqlalchemy import and_
import re

import setuptools
import abstrackr_ui
import term_label_editor
import progress

current_index_path = "current_ref_index"
current_lbl_d_path = "lbl_d"


# ##
# these two classes are for the ORM
# mapper (sqlalchemy)
###
class Citation(object):
    pass

class Labeling(object):
    pass

class Annotation(object):
    pass

class AbstrackrForm(QtGui.QMainWindow, abstrackr_ui.Ui_abstrackr_window):

    def __init__(self, parent=None, db_path=None):
        #
        # We follow the advice given by Mark Summerfield in his Python QT book:
        # Namely, we use multiple inheritence to gain access to the ui. We take
        # this approach throughout OpenMeta.
        #
        super(AbstrackrForm, self).__init__(parent)
        self.setupUi(self)

        # @TODO this is lame; we look for a file that tells
        # us the current refman / pubmed id -- need to generalize in the future
        self.current_refman_index = 0
        if os.path.isfile(current_index_path):
            self.current_refman_index = eval(open(current_index_path, 'r').readline())


        # @TODO this is (also) lame; same as above, but for
        # the reviewer
        reviewer_name_path = "whoami.txt"
        self.reviewer_name = open(reviewer_name_path, 'r').readline()
        
        self.current_lbl_d = {}
        if os.path.isfile(current_lbl_d_path):
            self.current_lbl_d = eval(open(current_lbl_d_path, 'r').readline())


    
        if db_path is not None:
            self.db_path = db_path
        elif len(sys.argv)<=1:
            print "\nno database path provided! dying."
            print "try: \n >python abstrackr.py my_db.db3"
            return
        else:
            self.db_path = sys.argv[-1]

        print "database path: %s" % self.db_path
        self.setup_db()

        self.refman_ids = self.get_refman_ids()
        self.refman_ids.sort()
        self.current_start_time = None
        self.display_current_citation()
        self.setup_signals_and_slots()

    def setup_signals_and_slots(self):
        QObject.connect(self.next_study_button, SIGNAL("pressed()"), self.next_study)
        QObject.connect(self.last_study_button, SIGNAL("pressed()"), self.previous_study)
        QObject.connect(self.accept_button, SIGNAL("pressed()"), self.accept_study)
        QObject.connect(self.reject_button, SIGNAL("pressed()"), self.reject_study)
        QObject.connect(self.maybe_button, SIGNAL("pressed()"), self.maybe_study)
        QObject.connect(self.jump_button, SIGNAL("pressed()"), self.jump_to_study)
        QObject.connect(self.jump_txt, SIGNAL("returnPressed()"),self.jump_to_study)
        QObject.connect(self.abstract_text, SIGNAL("selectionChanged()"),
                                            self.text_highlighted)
        QObject.connect(self.pos_term_btn1, SIGNAL("pressed()"),
                                       lambda: self.pos_annotation(1))
        QObject.connect(self.pos_term_btn2, SIGNAL("pressed()"),
                                       lambda: self.pos_annotation(2))
        QObject.connect(self.neg_term_btn1, SIGNAL("pressed()"),
                                       lambda: self.neg_annotation(1))
        QObject.connect(self.neg_term_btn2, SIGNAL("pressed()"),
                                       lambda: self.neg_annotation(2))
                          
        QObject.connect(self.action_edit_terms, SIGNAL("triggered()"),\
                                       self.edit_labeled_terms)
        QObject.connect(self.action_progress, SIGNAL("triggered()"),\
                                       self.view_progress)       

    def edit_labeled_terms(self):
        session = create_session()
        annotations = self._get_annotations(session)            
        editor_window =  term_label_editor.TermLabelEditor(annotations, session, parent=self)
        editor_window.show()
        
    def view_progress(self):
        session = create_session()
        labels = self._get_labels(session)
        progress_window = progress.Progress(labels, parent=self)
        progress_window.show()
        
    def pos_annotation(self, degree):
        session = create_session()
        labeled_feature = self.build_annotation()
        labeled_feature.label = float(degree)
        session.save(labeled_feature)
        session.flush()
        self.redisplay_current_citation()

    def neg_annotation(self, degree):
        session = create_session()
        labeled_feature = self.build_annotation()
        labeled_feature.label = -1* float(degree)
        session.save(labeled_feature)
        session.flush()
        self.redisplay_current_citation()
        
    def build_annotation(self):
        labeled_text = self.abstract_text.textCursor().selectedText()
        labeled_feature = Annotation()
        labeled_feature.study_id = self.current_refman_id
        labeled_feature.reviewer = self.reviewer_name
        labeled_feature.text = str(labeled_text)
        labeled_feature.annotation_time = datetime.now()
        return labeled_feature

    def accept_study(self):
        session = create_session()
        labeling = self._get_current_labeling(session)
        self.update_labeling_time(labeling)
        labeling.label = 1
        self.current_lbl_d[self.current_refman_id] = 1
        session.flush()
        self.next_study()

    def maybe_study(self):
        session = create_session()
        labeling = self._get_current_labeling(session)
        self.update_labeling_time(labeling)
        ##
        # we encode 'maybe' as 0.
        labeling.label = 0
        self.current_lbl_d[self.current_refman_id] = 0
        session.flush()
        self.next_study()
        
    def reject_study(self):
        session = create_session()
        labeling = self._get_current_labeling(session)
        self.update_labeling_time(labeling)
        labeling.label = -1
        self.current_lbl_d[self.current_refman_id] = -1
        session.flush()
        self.next_study()

    def update_labeling_time(self, labeling):
        time_delta = (datetime.now() - self.current_start_time).seconds
        if labeling.label is None:
            # then this is the first time the study has been labeled
            labeling.labeling_time = time_delta
            labeling.first_labeled = datetime.now()
        else:
            labeling.labeling_time = labeling.labeling_time + time_delta
        labeling.label_last_updated = datetime.now()

    def jump_to_study(self):
        jump_to_refman_id = int(str(self.jump_txt.text()))
        # inefficient, but that probably doesn't matter (linear
        # in the number of citations)
        self.current_refman_index = self.refman_ids.index(jump_to_refman_id)
        self.display_current_citation()

    def next_study(self):
        # note that we keep an index into the list of refman ids,
        # which we assume are at this point sorted. the ids themselves
        # needn't be contiguous.
        self.current_refman_index += 1
        if self.current_refman_index == len(self.refman_ids):
            self.current_refman_index = 0
        self.display_current_citation()

    def previous_study(self):
        self.current_refman_index -= 1
        if self.current_refman_index < 0:
            self.current_refman_index = len(self.refman_ids)-1
        self.display_current_citation()

    def get_refman_ids(self):
        query = self.citations.select()
        return [study['refman_id'] for study in query.execute().fetchall()]

    def display_current_citation(self):
        self.current_refman_id = self.refman_ids[self.current_refman_index]
        query = self.citations.select(self.citations.c.refman_id == self.current_refman_id)
        self.current_citation = query.execute().fetchone()

        session = create_session()
        cur_labeling = self._get_current_labeling(session)
        if cur_labeling.label is not None:
            cur_lbl = str(int(cur_labeling.label))

            if cur_lbl == "0":
                self.lbl_lbl.setText("?")
            else:
                self.lbl_lbl.setText(cur_lbl)
        else:
            self.lbl_lbl.setText("None")

        self.redisplay_current_citation()

        self.refman_id_lbl.setText("%s" % self.current_refman_id)
        self.current_start_time = datetime.now()
        # finally, write out the current refman index to file
        # (this approach to saving state needs to be improved,
        # obviously)
        fout = open(current_index_path, 'w')
        fout.write(str(self.current_refman_index))
        fout.close()

        fout = open(current_lbl_d_path, 'w')
        fout.write(str(self.current_lbl_d))
        fout.close()

    def redisplay_current_citation(self):
        # we don't always have an abstract
        abstract = self.current_citation["abstract"]
        if abstract is None:
            abstract = "(no abstract for this citation.)"
        # we *could* separate out the styling elements (h1, etc.) but this would
        # be overkill as it's minimal.
        display_str = ["<h1>%s</h1>" % self.mark_up(self.current_citation["title"])]
        display_str.append(self.current_citation["authors"])
        display_str.append("<b>in: %s</b>" % self.current_citation["journal"])
        display_str.append("<p>"+ self.mark_up(abstract) + "</p>")
        self.abstract_text.setText("<br>".join(display_str))

    def mark_up(self, text):
        # this higlights the labeled terms.
        # probably shouldn't query every time; but, eh.
        # premature optimization is evil and that jazz.
        pos_terms = self.annotations.select(self.annotations.c.label > 0).execute().fetchall()
        pos_terms_d = dict(zip([p_t.text for p_t in pos_terms],\
                               [p_t.label for p_t in pos_terms]))
 
        neg_terms = self.annotations.select(self.annotations.c.label < 0).execute().fetchall()
        neg_terms_d = dict(zip([n_t.text for n_t in neg_terms],\
                               [n_t.label for n_t in neg_terms]))
        neg_terms = self.annotations.select(self.annotations.c.label > 0).execute().fetchall()
        marked_up_text = self.mark_terms(text, pos_terms_d.keys())
        marked_up_text = self.mark_terms(marked_up_text, neg_terms_d.keys(), color="red")
        return marked_up_text

    def mark_terms(self, text, terms, color="yellow"):
      for term in terms:
          # TODO fix
          #case_insensitive = re.compile(term, re.IGNORECASE)
          #text = case_insensitive.sub(\
          #     "<FONT style='BACKGROUND-COLOR: %s'>%s</FONT>" % \
          #     (color, term), text)
          text = text.replace(term, "<FONT style='BACKGROUND-COLOR: %s'>%s</FONT>" % \
                       (color, term))
      return text

    def text_highlighted(self):
        if self.abstract_text.textCursor().hasSelection():
           self.toggle_term_annotation(True)
        else:
           self.toggle_term_annotation(False)

    def toggle_term_annotation(self, val):
        self.pos_term_btn1.setEnabled(val)
        self.pos_term_btn2.setEnabled(val)
        self.neg_term_btn1.setEnabled(val)
        self.neg_term_btn2.setEnabled(val)

    def setup_db(self):
        db = create_engine('sqlite:///%s' % self.db_path)
        metadata = MetaData(db)
        self.citations = Table('citations', metadata, autoload=True)
        self.labelings = Table('labeling', metadata, autoload=True)
        self.annotations = Table('annotations', metadata, autoload=True)
        self.citation_mapper = orm.mapper(Citation, self.citations)
        self.labeling_mapper = orm.mapper(Labeling, self.labelings)
        self.annotation_mapper = orm.mapper(Annotation, self.annotations)

    def _get_current_labeling(self, session):
        return session.query(Labeling).filter(and_(\
            Labeling.study_id == self.current_refman_id, Labeling.reviewer == self.reviewer_name))\
            .first()

    def _get_labels(self, session):
        return session.query(Labeling).filter(Labeling.reviewer == self.reviewer_name).all()
        
    def _get_annotations(self, session):
        return session.query(Annotation).all()
        
if __name__ == "__main__":

    print "abstrackr; version .0002"
    app = QtGui.QApplication(sys.argv)
    #abstrackr = AbstrackrForm(db_path =  "proton_beam.db3")
    abstrackr = AbstrackrForm()
    abstrackr.show()
    sys.exit(app.exec_())