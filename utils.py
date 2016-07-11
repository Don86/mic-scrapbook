
import numpy as np
#import scipy
import os
import sys
import time
import re


"""Various methods that are frequently used,
and aren't specific to any module
"""

def flatten_sparse_matrix(x, onehot = True):
    """Sets all nonzero entries of x to 1,
    then compresses to a 1D vector by taking the sum across rows
    i.e. squishes a rectangular matrix vertically.

    Params
    ------
    x: Array of float, shape (m, n)
    onehot: if true, converts any nonzero entry into 1

    Returns
    -------
    x_vec: 1D array of float, shape (n,)"""

    x_vec = x
    #Set all the nonzero elements of x to 1
    if onehot:
        x_vec[x_vec != 0] = 1
    x_vec = np.sum(x_vec, axis = 0)

    return x_vec


def sparse_vec_format(v):
    """Returns the indices and values of the nonzero elements
    of a sparse vector, v
    """

    C = []
    idx = []
    for i in range(len(v)):
        if v[i] != 0:
            C.append(v[i])
            idx.append(i)

    return C, idx
