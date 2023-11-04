from PIL import Image
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

# histogram

def histogram_f(array, channel):
    print("something is happening")
    plt.hist(array.ravel(), bins=256)
    print("histogram is done")
    return plt.gcf()
