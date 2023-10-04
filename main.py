from PIL import Image
import numpy as np


def helpp():
    print("")


# to improve while saturated
def brightness(array, num):
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            for rgbIndex, rgb in enumerate(y):
                if rgb + num >= 255:
                    array[xIndex][xIndex][rgbIndex] = 255
                elif rgb + num <= 0:
                    array[xIndex][xIndex][rgbIndex] = 0
                else:
                    array[xIndex][xIndex][rgbIndex] = rgb + num
    return array


def contrast():
    print("")


def negative():
    print("")


def hflip(array):
    i = 0
    for y in array:
        while len(y) / 2 > i:
            array[:, [i, len(y) - 1 - i], :] = array[:, [len(y) - 1 - i, i], :]
            i += 1
    return array


def vflip(array):
    i = 0
    for x in array:
        while len(x) / 2 > i:
            array[[i, len(x) - 1 - i], :, :] = array[[len(x) - 1 - i, i], :, :]
            i += 1
    return array


def dflip(array):
    vflip(hflip(array))
    return array


image = Image.open("lenac.bmp")
arr = np.array(image.getdata())
if arr.ndim == 1: #grayscale
    arr = arr.reshape(image.size[1], image.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(image.size[1], image.size[0], numColorChannels)

# arr = brightness(arr, 0)
# arr = dflip(arr)

newImage = Image.fromarray(arr.astype(np.uint8))
newImage.show()
newImage.save("result.bmp")
