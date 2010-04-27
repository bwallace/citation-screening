
import pdb
import AA4
import sqlite3

def xml_to_sql(xml_path, db_path):
    ref_d = AA4.xmlToDict(xml_path)
    
    # now establish a db connection
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # setup the tables
    setup_tables(c)
    # now populate them
    to_sql(c, ref_d)
    
def to_sql(c, xml_d):
    pdb.set_trace()
    pass
    
def setup_tables(c):
    c.execute('''create table citations(
       refman_id INTEGER PRIMARY KEY,
       title TEXT,
       authors TEXT, 
       abstract TEXT,
       journal TEXT,
       keywords TEXT,
       pmid INTEGER,
       )''')
    
    c.execute('''create table labeling(
       id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
       study_id INTEGER NOT NULL,
       reviewer TEXT NOT NULL,
       label FLOAT DEFAULT "NULL" ,
       start_time DATETIME,
       end_time DATETIME
    )''')