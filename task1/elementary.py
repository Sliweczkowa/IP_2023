import numpy as np


# B1 | Image brightness modification
def brightness(array, num):
    if array.ndim == 2:
        for xIndex, x in enumerate(array):
            for yIndex, y in enumerate(x):
                if y + num >= 255:
                    array[xIndex][yIndex] = 255
                elif y + num <= 0:
                    array[xIndex][yIndex] = 0
                else:
                    array[xIndex][yIndex] = y + num
    elif array.ndim == 3:
        for xIndex, x in enumerate(array):
            for yIndex, y in enumerate(x):
                for c in range(3):
                    if y[c] + num >= 255:
                        array[xIndex][yIndex][c] = 255
                    elif y[c] + num <= 0:
                        array[xIndex][yIndex][c] = 0
                    else:
                        array[xIndex][yIndex][c] = y[c] + num

    return array


# B2 | Image contrast modification
def contrast(array, num):
    if array.ndim == 2:
        for xIndex, x in enumerate(array):
            for yIndex, y in enumerate(x):
                current_pixel = array[xIndex][yIndex]
                new_pixel = pow((current_pixel / 255), num) * 255
                array[xIndex][yIndex] = np.clip(new_pixel, 0, 255)
    elif array.ndim == 3:
        for xIndex, x in enumerate(array):
            for yIndex, y in enumerate(x):
                for c in range(3):
                    current_pixel = array[xIndex][yIndex][c]
                    new_pixel = pow((current_pixel / 255), num) * 255
                    array[xIndex][yIndex][c] = np.clip(new_pixel, 0, 255)
    return array


# B3 | Negative
def negative(array):
    array[:] = 255 - array
    return array
