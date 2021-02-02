# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

import numpy as np
from numpy.ctypeslib import ndpointer
from ctypes import cdll, c_void_p, c_bool, c_float, c_int
from os.path import abspath, join, pardir, dirname, realpath

import glob

libfile = glob.glob('build/*/utils*.so')[0]

libutils = cdll.LoadLibrary(libfile)

_find_peaks = libutils.find_peaks
_find_peaks.argtypes = [
    ndpointer(dtype=np.float32, ndim=2, flags='C,A,W'),   # out
    ndpointer(dtype=np.float32, ndim=2, flags='C,A'),     # img
    c_int,                                                # width
    c_int,                                                # height
    c_float,                                              # threshold
    c_bool,                                               # subpixel
]
_find_peaks.restype = c_int

def find_peaks(img, threshold, subpixel=False):
    assert img.ndim == 2
    h, w = img.shape
    size = (h//2+1) * (w//2+1)
    img = np.require(img, dtype=np.float32, requirements=('C', 'A'))
    out = np.require(np.empty((size, 2)), dtype=np.float32, requirements=('C', 'A', 'W'))
    count = _find_peaks(out, img, int(w), int(h), np.float32(threshold), bool(subpixel))
    return out[:count]
