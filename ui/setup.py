from distutils.core import setup
import py2exe

#setup(console=['abstrackr.py'], eggs = ['sqlalchemy', 'setuptools'])
setup(windows=[{"script" : "abstrackr.py"}], eggs= ['sqlalchemy', 'setuptools'],
          options={"py2exe" : {"includes" : ["sip", "PyQt4.QtSql","sqlite3"],\
                                             "packages":["sqlalchemy.databases.sqlite", "setuptools"]}})