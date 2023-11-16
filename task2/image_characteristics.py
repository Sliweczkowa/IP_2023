import numpy as np


# image characteristics


# C1.1 | Mean
def cmean(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
    to_sum = 0

    to_sum = np.sum(np.arange(levels) * histogram)
    b = 1 / (width * height) * to_sum
    return b


# C1.2 | Variance
def cvariance(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
    b = cmean(arr, levels)
    to_sum = 0
    to_sum = np.sum(np.square(np.arange(levels)-b) * histogram)
    variance = 1 / (width * height) * to_sum
    return variance


# C2.1 | Standard deviation
def cstdev(arr, levels):
    d_sq = cvariance(arr, levels)
    d = np.sqrt(d_sq)
    return d


# C2.2 | Variation coefficient I
def cvarcoi(arr, levels):
    d = cstdev(arr, levels)
    b = cmean(arr, levels)
    b_z = d / b
    return b_z


# C3 | Assymetry coefficient
def casyco(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram,bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))

    s = cstdev(arr, levels)
    b = cmean(arr, levels)
    b_s = (1 / np.power(s, 3)) * 1 / width * height * np.sum(np.power((np.arange(levels)-b), 3) * histogram)
    return b_s


# C4 | Flattening coefficient
def cflaco(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))

    s = cstdev(arr, levels)
    b = cmean(arr, levels)
    b_k = (1 / np.power(s, 4)) * 1 / width * height * np.sum(np.power((np.arange(levels)-b), 4) * histogram - 3)
    return b_k


# C5 | Variation coefficient II
def cvarcoii(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))

    b_n = np.square(1 / (width * height)) * np.sum(np.square(histogram))
    return b_n


# C6 | Information source entropy
def centropy(arr, levels):
    width = len(arr[0])
    height = len(arr)

    histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
    epsilon = 1e-10
    histogram = histogram + epsilon
   
    b_e = -1 / (width * height) * np.sum(histogram * np.log2(histogram / (width * height)))
    return b_e
