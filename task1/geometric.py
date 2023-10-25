from PIL import Image
import numpy as np
import argparse
import sys


# G1 | Horizontal flip
def hflip(array):
    array[:] = array[:, ::-1]
    return array


# G2 | Vertical flip
def vflip(array):
    array[:] = array[::-1, :]
    return array


# G3 | Diagonal flip
def dflip(array):
    vflip(hflip(array))
    return array


# G4 | Image enlargement using scale
def enlarge(array0, scale):
    height0 = len(array0)
    width0 = len(array0[0])

    width1 = int(width0 * scale)
    height1 = int(height0 * scale)

    if array0.ndim == 2:
        array1 = np.empty([height1, width1])
    elif array0.ndim == 3:
        array1 = np.empty([height1, width1, 3])
    for h in range(width1):
        for w in range(height1):
            array1[w][h] = array0[int(width0 * w / width1)][int(height0 * h / height1)]
    return array1


# G5 | Image shrinking using scale
def shrink(array, scale):
    return enlarge(array, (1 / scale))
