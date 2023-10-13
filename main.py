from PIL import Image
import numpy as np
import argparse


def loadImg(path):
    image = Image.open(path)
    arr = np.array(image.getdata())
    if arr.ndim == 1: #grayscale
        numColorChannels = 1
        arr = arr.reshape(image.size[1], image.size[0])
    else:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(image.size[1], image.size[0], numColorChannels)
    return arr

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


# G4/G5 | Image shrinking/enlargement
def resize(array0, width1, height1):
    width0 = len(array0)
    height0 = len(array0[0])

    if width1 == 0:
        width1 = int(width0 * height1 / height0)
    elif height1 == 0:
        height1 = int(height0 * width1 / width0)

    if array0.ndim == 2:
        array1 = np.empty([height1, width1])
    elif array0.ndim == 3:
        array1 = np.empty([height1, width1, 3])
    for h in range(width1):
        for w in range(height1):
            array1[w][h] = array0[int(width0 * w / width1)][int(height0 * h / height1)]
    return array1


# N4.1 | Midpoint filter
def mid(array, size):
    height = len(array[0])
    width = len(array)

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

    return filtered_array


# N4.2 | Arithmetic mean filter
def amean(array, size):
    height = len(array[0])
    width = len(array)

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

    return filtered_array


# squared differences sum
def sqd_dif_sum(img1, img2):
    sqd_dif = np.square(img1 - img2)
    sum = np.sum(sqd_dif)
    return sum

# E1 | Mean square error 
def mse(org_img, noise_img, fil_img):
    height = len(org_img[0])
    width = len(org_img)

    #original and filtered
    sum_org_fil = sqd_dif_sum(org_img, fil_img)
    err_org_fil = sum_org_fil / (width*height) 

    #original and noise
    sum_org_noise = sqd_dif_sum(org_img, noise_img)
    err_org_noise = sum_org_noise / (width*height)

    return err_org_noise, err_org_fil

# E2 | Peak mean square error 
def pmse(org_img, noise_img, fil_img):
    max_org_img = np.max(org_img)

    mse_res = mse(org_img, noise_img, fil_img)

    #original and filtered
    err_org_fil = mse_res[0] / np.square(max_org_img)

    #original anf noise
    err_org_noise = mse_res[1] / np.square(max_org_img)

    return err_org_fil, err_org_noise

# E3 | Signal to noise ratio 
# E4 | Peak signal to noise ratio 
# E5 | Maximum difference


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
parser.add_argument('--mid', help='midpoint filter', type=int)
parser.add_argument('--amean', help='arithmetic mean filter', type=int)
parser.add_argument('--mse', help='mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3)
parser.add_argument('--pmse', help='peak mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3)
parser.add_argument('--load', help='loads an image from a given path')
parser.add_argument('--save', help='saves edited image in a specified folder under a specified name')

args = parser.parse_args()

if args.load:
    imgPath = args.load
    arr = loadImg(imgPath)


if args.brightness and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.brightness:
    value = args.brightness
    brightness(arr, value)

if args.contrast and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.contrast:
    value = args.contrast
    contrast(arr, value)

if args.negative and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.negative:
    negative(arr)

if args.hflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.hflip:
    hflip(arr)
        
if args.vflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.vflip:
    vflip(arr)

if args.dflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.dflip:
    dflip(arr)

if args.mid and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.mid:
    value = args.mid
    arr = mid(arr, value)

if args.amean and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.amean:
    value = args.amean
    arr = amean(arr, value)

if args.mse:
    original = loadImg(args.mse[0])
    noise = loadImg(args.mse[1])
    filtered = loadImg(args.mse[2])
    result = mse(original, noise, filtered)
    print("original/noise: " + str(result[0]) + ", original/filter: " + str(result[1]))

if args.pmse:
    original = loadImg(args.pmse[0])
    noise = loadImg(args.pmse[1])
    filtered = loadImg(args.pmse[2])
    result = pmse(original, noise, filtered)
    print("original/noise: " + str(result[0]) + ", original/filter: " + str(result[1]))

if args.save:
    newImage = Image.fromarray(arr.astype(np.uint8))
    newImage.save(args.save)
    newImage.show()
