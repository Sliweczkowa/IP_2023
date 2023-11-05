from PIL import Image
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

# histogram

def histogram_f(array, channel):
    if channel not in ("greyscale", "R", "G", "B"):
        print("Incorrect channel for histogram!")
        exit()
    if array.ndim == 2 and channel == "greyscale":
        plt.hist(array.ravel(), bins=256, color="gray")
    if array.ndim == 3 and channel in ("R", "G", "B"):
        if channel == "R":
            plt.hist(array[:,:,0].ravel(), bins=256, color="red")
        elif channel == "G":
             plt.hist(array[:,:,1].ravel(), bins=256, color="green")
        elif channel == "B":
             plt.hist(array[:,:,2].ravel(), bins=256, color="blue")
    return plt.gcf()


# H1 | Uniform final probability density function
