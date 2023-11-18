import numpy as np
import matplotlib.pyplot as plt


# histogram


def histogram_f(array, channel):
    if channel not in ("greyscale", "R", "G", "B"):
        print("Incorrect channel for histogram!")
        exit()
    if array.ndim == 2 and channel == "greyscale":
        histogram, bins = np.histogram(array.ravel(), bins=256, range=(0, 255))
        plt.bar(bins[:-1], histogram, width=1, color="gray")
    if array.ndim == 3 and channel in ("R", "G", "B"):
        if channel == "R":
            histogram, bins = np.histogram(array[:, :, 0].ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], histogram, width=1, color="red")
        elif channel == "G":
            histogram, bins = np.histogram(array[:, :, 1].ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], histogram, width=1, color="green")
        elif channel == "B":
            histogram, bins = np.histogram(array[:, :, 2].ravel(), bins=256, range=(0, 255))
            plt.bar(bins[:-1], histogram, width=1, color="blue")
    return plt.gcf(), histogram


# H1 | Uniform final probability density function
def huniform(arr, g_min, g_max):
    width = len(arr[0])
    height = len(arr)
    
    new_arr = np.zeros_like(arr)

    if arr.ndim == 2:
        histogram, bins = np.histogram(arr.ravel(), bins=256, range=(0, 255))

        for i in range(height):
            for j in range(width):
                new_arr[i, j] = g_min + (g_max-g_min) * (1/(height*width) * np.cumsum(histogram)[arr[i, j]])

    elif arr.ndim == 3:
        histogram_r, bins = np.histogram(arr[:, :, 0].ravel(), bins=256, range=(0, 255))
        histogram_g, bins = np.histogram(arr[:, :, 1].ravel(), bins=256, range=(0, 255))
        histogram_b, bins = np.histogram(arr[:, :, 2].ravel(), bins=256, range=(0, 255))

        histogram = (histogram_r, histogram_g, histogram_b)

        for c in range(3):
            for x in range(width):
                for y in range(height):
                    new_arr[x, y, c] = g_min + (g_max-g_min) * (1/(height*width) * np.cumsum(histogram[c])[arr[x, y, c]])

    return new_arr
