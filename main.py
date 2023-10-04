from PIL import Image
import numpy as np


def helpp():
    print("")


# to improve while saturated
def brightness(array, num):
    for xIndex, x in enumerate(array):
        for pixelIndex, pixel in enumerate(x):
            for rgbIndex, rgb in enumerate(pixel):
                print(pixel)
                if rgb + num >= 255:
                    array[xIndex][pixelIndex][rgbIndex] = 255
                elif rgb + num <= 0:
                    array[xIndex][pixelIndex][rgbIndex] = 0
                else:
                    array[xIndex][pixelIndex][rgbIndex] = rgb + num
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
