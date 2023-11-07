import numpy as np
import matplotlib.pyplot as plt

# image characteristics

# C1.1 | Mean

def cmean(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram,bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
    to_sum=0

    to_sum = np.sum(np.arange(levels) * histogram)
    b = 1 / (width * height * to_sum)
    return b

# C1.2 | Variance

def cvariance(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram,bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
    b = cmean(arr, levels)
    to_sum=0
    to_sum = np.sum(np.square(np.arange(levels)-b) * histogram)
    variance = 1 / (width * height * to_sum)
    return variance

# C2.1 | Standard deviation

def cstdev(arr, levels):
    d_sq = cvariance(arr, levels)
    d = np.sqrt(d_sq)
    return d

# C2.2 | Variation coefficient I

def cvarcoi(arr):
    return

# C3 | Assymetry coefficient

def casyco(arr):
    return

# C4 | Flattening coefficient

def cflaco(arr):
    return

# C5 | Variation coefficient II

def cvarcoii(arr):
    return

# C6 | Information source entropy

def centropy(arr):
    return