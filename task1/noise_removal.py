import numpy as np


# N4.1 | Midpoint filter
def mid(array, size):
    height = len(array)
    width = len(array[0])

    border = size // 2

    if array.ndim == 2:
        filtered_array = np.empty([height, width])
        for i in range(border, height - border):
            for j in range(border, width - border):
                neighborhood = array[i - border:i + border + 1, j - border:j + border + 1]
                midpoint = (np.amin(neighborhood) + np.amax(neighborhood)) // 2
                filtered_array[i, j] = midpoint

    if array.ndim == 3:
        filtered_array = np.empty([height, width, 3])

        for c in range(3):
            for i in range(border, height - border):
                for j in range(border, width - border):
                    neighborhood = array[i - border:i + border + 1, j - border:j + border + 1, c]
                    midpoint = (np.amin(neighborhood) + np.amax(neighborhood)) // 2
                    filtered_array[i, j, c] = midpoint

    for i in range(border):
        filtered_array[i, :] = array[i, :]
        filtered_array[height - border + i, :] = array[height - border + i, :]
        filtered_array[:, i] = array[:, i]
        filtered_array[:, width - border + i] = array[:, width - border + i]

    return filtered_array


# N4.2 | Arithmetic mean filter
def amean(array, size):
    height = len(array)
    width = len(array[0])

    border = size // 2

    if array.ndim == 2:
        filtered_array = np.empty([height, width])
        for i in range(border, height - border):
            for j in range(border, width - border):
                neighborhood = array[i - border:i + border + 1, j - border:j + border + 1]
                amean = np.mean(neighborhood)
                filtered_array[i, j] = amean

    if array.ndim == 3:
        filtered_array = np.empty([height, width, 3])

        for c in range(3):
            for i in range(border, height - border):
                for j in range(border, width - border):
                    neighborhood = array[i - border:i + border + 1, j - border:j + border + 1, c]
                    amean = (np.mean(neighborhood))
                    filtered_array[i, j, c] = amean

    for i in range(border):
        filtered_array[i, :] = array[i, :]
        filtered_array[height - border + i, :] = array[height - border + i, :]
        filtered_array[:, i] = array[:, i]
        filtered_array[:, width - border + i] = array[:, width - border + i]

    return filtered_array
