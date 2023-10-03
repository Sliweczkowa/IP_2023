from PIL import Image
import numpy as np


def helpp():
    print("")


# to improve while saturated
def brightness(array, num):
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            for pixelIndex, pixel in enumerate(y):
                if pixel + num > 255:
                    array[xIndex][yIndex][pixelIndex] = 255
                elif pixel + num < 0:
                    array[xIndex][yIndex][pixelIndex] = 0
                else:
                    array[xIndex][yIndex][pixelIndex] = pixel + num
    return array


def contrast():
    print("")


def negative():
    print("")


def hflip():
    print("")


image = Image.open("lenac.bmp")
arr = np.array(image.getdata())
if arr.ndim == 1: #grayscale
    arr = arr.reshape(image.size[1], image.size[0])
else:
    numColorChannels = arr.shape[1]
    arr = arr.reshape(image.size[1], image.size[0], numColorChannels)

# arr = brightness(arr, 100)

newImage = Image.fromarray(arr.astype(np.uint8))
newImage.show()
newImage.save("result.bmp")
