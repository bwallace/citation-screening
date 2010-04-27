
import pdb
import sqlite3

import elementtree
from elementtree.ElementTree import ElementTree

def xml_to_sql(xml_path, db_path):
    print "building a dictionary from %s..." % xml_path
    ref_d = xml_to_dict(xml_path)
    print "ok."

    # now establish a db connection
    print "setting up a database..."
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # setup the tables
    setup_tables(c)
    print "ok. now populating the database..."
    # now populate them
    to_sql(c, ref_d)

    print "success."
    conn.commit()
    conn.close()

def to_sql(c, xml_d):
    for ref_id, citation_d in xml_d.items():
        insert_citation(c, ref_id, citation_d)
        insert_labeling_task(c, ref_id, "James")

def insert_citation(c, refman_id, citation_d):
    c.execute('''insert into citations
        (refman_id, title, authors, abstract, journal, keywords, pmid)
        VALUES (?, ?, ?, ?, ?, ?, ?);''',
        (refman_id, citation_d["title"], " and ".join(citation_d["authors"]), citation_d['abstract'],
         citation_d['journal'], ','.join(citation_d['keywords']), citation_d['pmid']))

def insert_labeling_task(c, refman_id, assign_to):
    c.execute('''insert into labeling
        (id, study_id, reviewer)
        VALUES (NULL, ?, ?);''',
        (refman_id, assign_to))

def setup_tables(c):
    c.execute('''create table citations(
       refman_id INTEGER PRIMARY KEY,
       title TEXT,
       authors TEXT,
       abstract TEXT,
       journal TEXT,
       keywords TEXT,
       pmid INTEGER
    )''')

    c.execute('''create table labeling(
       id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
       study_id INTEGER NOT NULL,
       reviewer TEXT NOT NULL,
       label FLOAT DEFAULT NULL ,
       labeling_time INTEGER,
       first_labeled DATETIME,
       label_last_updated DATETIME
    )''')

    c.execute('''create table annotations(
       id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
       study_id INTEGER NOT NULL,
       reviewer TEXT NOT NULL,
       label FLOAT DEFAULT NULL ,
       text TEXT NOT NULL,
       annotation_time DATETIME
    )''')

def xml_to_dict(fPath):
    '''
    Converts study data from (ref man generated) XML to a dictionary matching study IDs (keys) to
    title/abstract tuples (values). For example: dict[n] might map to a tuple [t_n, a_n] where t_n is the
    title of the nth paper and a_n is the abstract
    '''

    ref_ids_to_abs = {}
    num_no_abs = 0
    tree = ElementTree(file=fPath)

    for record in tree.findall('.//record'):
        pubmed_id = None
        refmanid = eval(record.findtext('.//rec-number'))

        # attempt to grab the pubmed id
        pubmed_id = ""
        try:
            pubmed = record.findtext('.//notes/style')
            pubmed = pubmed.split("-")
            for i in range(len(pubmed)):
                if "UI" in pubmed[i]:
                    pubmed_str = pubmed[i+1].strip()
                    pubmed_id = eval("".join([x for x in pubmed_str if x in string.digits]))
        except Exception, ex:
            print ex

        ab_text = record.findtext('.//abstract/style')
        if ab_text is None:
            num_no_abs += 1

        title_text = record.findtext('.//titles/title/style')

        # Also grab keywords
        keywords = [keyword.text.strip().lower() for keyword in record.findall(".//keywords/keyword/style")]

        # and authors
        authors = [author.text for author in record.findall(".//contributors/authors/author/style")]

        # journal
        journal = record.findtext(".//periodical/abbr-1/style")

        ref_ids_to_abs[refmanid] = {"title":title_text, "abstract":ab_text, "journal":journal,\
                    "keywords":keywords, "pmid":pubmed_id, "authors":authors}

    print "Finished. Returning %s title/abstract/keyword sets, %s of which have no abstracts." \
                    % (len(ref_ids_to_abs.keys()), num_no_abs)
    return ref_ids_to_abs

def get_authors(record):
    pass

