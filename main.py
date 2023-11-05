from PIL import Image
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

from task1 import elementary            # B1-B3
from task1 import geometric             # G1-G5
from task1 import noise_removal         # N4
from task1 import similarity_measures   # E1-E5
from task2 import histogram             # H


def loadImg(path):
    image = Image.open(path)
    arr = np.array(image.getdata())
    if arr.ndim == 1: 
        numColorChannels = 1
        arr = arr.reshape(image.size[1], image.size[0])
    elif arr.ndim == 2:
        numColorChannels = arr.shape[1]
        arr = arr.reshape(image.size[1], image.size[0], numColorChannels)
    return arr


# main
value = 0
is_histogram = 0
imgPath = ""

parser = argparse.ArgumentParser()
parser.add_argument('--brightness', help='change of brightness', type=int, metavar=('Value'))
parser.add_argument('--contrast', help='change of contrast', type=float, metavar=('Value'))
parser.add_argument('--negative', help='negative of the image', action="store_true")
parser.add_argument('--hflip', help='horizontal flip', action="store_true")
parser.add_argument('--vflip', help='vertical flip', action="store_true")
parser.add_argument('--dflip', help='diagonal flip', action="store_true")
parser.add_argument('--shrink', help='image shrinking', type=int, metavar=('Scale'))
parser.add_argument('--enlarge', help='image enlargement', type=int, metavar=('Scale'))
parser.add_argument('--mid', help='midpoint filter', type=int, metavar=('Cluster size'))
parser.add_argument('--amean', help='arithmetic mean filter', type=int, metavar=('Cluster size'))
parser.add_argument('--mse', help='mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--pmse', help='peak mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--snr', help='signal to noise ratio, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--psnr', help='peak signal to noise ratio, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--md', help='maximum difference error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--histogram', help='generates a histogram for a chosen chanel of a chosen image', type=str, metavar="Channel (R/G/B/greyscale)")
parser.add_argument('--load', help='loads an image from a given path', metavar=('Path'))
parser.add_argument('--save', help='saves edited image in a specified folder under a specified name', metavar=('Path'))

args = parser.parse_args()

if args.load:
    try:
        imgPath = args.load
        arr = loadImg(imgPath)
    except (IOError, ValueError, FileNotFoundError) as e:
        print(f"Error: Unable to load the image. Please check the file extension and try again.")
        sys.exit(1)

if args.brightness and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.brightness:
    value = args.brightness
    elementary.brightness(arr, value)

if args.contrast and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.contrast:
    value = args.contrast
    elementary.contrast(arr, value)

if args.negative and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.negative:
    elementary.negative(arr)

if args.hflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.hflip:
    geometric.hflip(arr)
        
if args.vflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.vflip:
    geometric.vflip(arr)

if args.dflip and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.dflip:
    geometric.dflip(arr)

if args.shrink and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.shrink:
    value = args.shrink
    arr = geometric.shrink(arr, value)

if args.enlarge and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.enlarge:
    value = args.enlarge
    arr = geometric.enlarge(arr, value)

if args.mid and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.mid:
    value = args.mid
    arr = noise_removal.mid(arr, value)

if args.amean and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.amean:
    value = args.amean
    arr = noise_removal.amean(arr, value)

if args.mse:
    original = loadImg(args.mse[0])
    noise = loadImg(args.mse[1])
    filtered = loadImg(args.mse[2])
    result = similarity_measures.mse(original, noise, filtered)
    print("original/noise (greyscale or R,G,B): " + str(result[0]) + ", original/filter(greyscale or R,G,B): " + str(result[1]))

if args.pmse:
    original = loadImg(args.pmse[0])
    noise = loadImg(args.pmse[1])
    filtered = loadImg(args.pmse[2])
    result = similarity_measures.pmse(original, noise, filtered)
    print("original/noise (greyscale or R,G,B): " + str(result[1]) + ", original/filter (greyscale or R,G,B): " + str(result[0]))

if args.snr:
    original = loadImg(args.snr[0])
    noise = loadImg(args.snr[1])
    filtered = loadImg(args.snr[2])
    result = similarity_measures.snr(original, noise, filtered)
    print("original/noise (greyscale or R,G,B): " + str(result[1]) + ", original/filter (greyscale or R,G,B): " + str(result[0]))

if args.psnr:
    original = loadImg(args.psnr[0])
    noise = loadImg(args.psnr[1])
    filtered = loadImg(args.psnr[2])
    result = similarity_measures.psnr(original, noise, filtered)
    print("original/noise (greyscale or R,G,B): " + str(result[1]) + ", original/filter (greyscale or R,G,B): " + str(result[0]))

if args.md:
    original = loadImg(args.md[0])
    noise = loadImg(args.md[1])
    filtered = loadImg(args.md[2])
    result = similarity_measures.md(original, noise, filtered)
    print("original/noise (greyscale or R,G,B): " + str(result[1]) + ", original/filter (greyscale or R,G,B): " + str(result[0]))

if args.histogram and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.histogram:
    histogramImg = histogram.histogram_f(arr, args.histogram)
    is_histogram = 1

if args.save:
    try:
        if is_histogram == 1:
            histogramImg.savefig(args.save)
            plt.show()
        elif is_histogram != 1:
            newImage = Image.fromarray(arr.astype(np.uint8))
            newImage.save(args.save)
            newImage.show()
    except (IOError, ValueError) as e:
        print(f"Error: Unable to save the image. Please check the file extension and try again.")
