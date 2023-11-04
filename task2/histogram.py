from PIL import Image
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

# histogram

def histogram_f(array, channel):

    if array.ndim == 2:
        plt.hist(array.ravel(), bins=256, color="gray")
    if array.ndim == 3:
        if channel == "R":
            plt.hist(array[:,:,0].ravel(), bins=256, color="red")
        elif channel == "G":
             plt.hist(array[:,:,1].ravel(), bins=256, color="green")
        elif channel == "B":
             plt.hist(array[:,:,2].ravel(), bins=256, color="blue")
    return plt.gcf()
