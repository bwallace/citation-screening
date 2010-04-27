import numpy
from numpy import array
from numpy import linalg

from numpy import *
import pdb
 
 
def pca(data):
    """ assume one sample per column """
    values, vecs = linalg.eigh(cov(data))
    perm = argsort(-values)  # sort in descending order
    return values[perm], vecs[:, perm]