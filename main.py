from PIL import Image
import numpy as np


# HELP | Commands (+ detailed description of arguments) print
def helpp():
    print("")


# B1 | Image brightness modification
# to improve while saturated
# to fix while grayscale
def brightness(array, num):
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            for rgbIndex, rgb in enumerate(y):
                if rgb + num >= 255:
                    array[xIndex][yIndex][rgbIndex] = 255
                elif rgb + num <= 0:
                    array[xIndex][yIndex][rgbIndex] = 0
                else:
                    array[xIndex][yIndex][rgbIndex] = rgb + num
    return array


# B2 | Image contrast modification
# to fix while greyscale
def contrast(array, num):
    num /= 2
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            for rgbIndex, rgb in enumerate(y):
                if rgb - num <= 0:
                    array[xIndex][yIndex][rgbIndex] = 0
                elif rgb + num >= 255:
                    array[xIndex][yIndex][rgbIndex] = 255
                elif rgb <= 127:
                    array[xIndex][yIndex][rgbIndex] = rgb - num
                elif rgb >= 128:
                    array[xIndex][yIndex][rgbIndex] = rgb + num
    return array


# B3 | Negative
def negative(array):
    return 255 - array


# G1 | Horizontal flip
def hflip(array):
    i = 0
    for y in array:
        while len(y) / 2 > i:
            array[:, [i, len(y) - 1 - i]] = array[:, [len(y) - 1 - i, i]]
            i += 1
    return array


# G2 | Vertical flip
def vflip(array):
    i = 0
    for x in array:
        while len(x) / 2 > i:
            array[[i, len(x) - 1 - i], :] = array[[len(x) - 1 - i, i], :]
            i += 1
    return array


# G3 | Diagonal flip
def dflip(array):
    vflip(hflip(array))
    return array


# main
image = Image.open("lena.bmp")
arr = np.array(image.getdata())
if arr.ndim == 1: #grayscale
    arr = arr.reshape(image.size[1], image.size[0])
    numColorChannels = 1
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(image.size[1], image.size[0], numColorChannels)

dflip(arr)

newImage = Image.fromarray(arr.astype(np.uint8))
newImage.show()
newImage.save("result.bmp")
