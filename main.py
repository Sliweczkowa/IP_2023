from PIL import Image
import numpy as np
import argparse


# B1 | Image brightness modification
# to improve while saturated
def brightness(array, num):
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            if y + num >= 255:
                array[xIndex][yIndex] = 255
            elif y + num <= 0:
                array[xIndex][yIndex] = 0
            else:
                array[xIndex][yIndex] = y + num
    return array


# B2 | Image contrast modification
# to improve while saturated
def contrast(array, num):
    num /= 2
    for xIndex, x in enumerate(array):
        for yIndex, y in enumerate(x):
            if y - num <= 0:
                array[xIndex][yIndex] = 0
            elif y + num >= 255:
                array[xIndex][yIndex] = 255
            elif y <= 127:
                array[xIndex][yIndex] = y - num
            elif y >= 128:
                array[xIndex][yIndex] = y + num
    return array


# B3 | Negative
def negative(array):
    array[:] = 255 - array
    return array


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


# G4 | Image shrinking
def shrink(array, num):
    return array


# G5 | Image enlargement
def enlarge(array, num):
    return array


# N4.1 | Midpoint filter
def mid(array):
    return array


# N4.2 | Arithmetic mean filter
def amean(array):
    return array


# main
value = 0
imgPath = ""

parser = argparse.ArgumentParser()
parser.add_argument('--brightness', help='change of brightness', type=int)
parser.add_argument('--contrast', help='change of contrast', type=int)
parser.add_argument('--negative', help='negative of the image', action="store_true")
parser.add_argument('--hflip', help='horizontal flip', action="store_true")
parser.add_argument('--vflip', help='vertical flip', action="store_true")
parser.add_argument('--dflip', help='diagonal flip', action="store_true")
parser.add_argument('--load', help='loads an image from a given path', required=1)
parser.add_argument('--save', help='saves edited image in a specified folder under a specified name', required=1)

args = parser.parse_args()

if args.load:
    imgPath = args.load
    image = Image.open(imgPath)
    arr = np.array(image.getdata())
    if arr.ndim == 1: #grayscale
        numColorChannels = 1
        arr = arr.reshape(image.size[1], image.size[0])
    else:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(image.size[1], image.size[0], numColorChannels)

if args.brightness:
    value = args.brightness
    brightness(arr, value)

if args.contrast:
    value = args.contrast
    contrast(arr, value)

if args.negative:
    negative(arr)

if args.hflip:
    hflip(arr)
        
if args.vflip:
    vflip(arr)

if args.dflip:
    dflip(arr)

if args.save:
    newImage = Image.fromarray(arr.astype(np.uint8))
    newImage.save(args.save)
    newImage.show()
