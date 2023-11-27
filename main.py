from PIL import Image
import numpy as np
import argparse
import sys
import matplotlib.pyplot as plt

from task1 import elementary                # B1-B3
from task1 import geometric                 # G1-G5
from task1 import noise_removal             # N4
from task1 import similarity_measures       # E1-E5

from task2 import histogram                 # H1
from task2 import image_characteristics     # C1-C6
from task2 import image_filtration          # S4, O3

from task3 import morphological             # dilation, erosion, opening, closing, HMT transformation, M7
from task3 import structural_elements
from task3.structural_elements import StructuralElement


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
parser.add_argument('--brightness', help='change of brightness', type=int, metavar='Value')
parser.add_argument('--contrast', help='change of contrast', type=float, metavar='Value')
parser.add_argument('--negative', help='negative of the image', action="store_true")
parser.add_argument('--hflip', help='horizontal flip', action="store_true")
parser.add_argument('--vflip', help='vertical flip', action="store_true")
parser.add_argument('--dflip', help='diagonal flip', action="store_true")
parser.add_argument('--shrink', help='image shrinking', type=int, metavar='Scale')
parser.add_argument('--enlarge', help='image enlargement', type=int, metavar='Scale')
parser.add_argument('--mid', help='midpoint filter', type=int, metavar='Cluster size')
parser.add_argument('--amean', help='arithmetic mean filter', type=int, metavar='Cluster size')
parser.add_argument('--mse', help='mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--pmse', help='peak mean squared error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--snr', help='signal to noise ratio, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--psnr', help='peak signal to noise ratio, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--md', help='maximum difference error, arg1=original image, arg2=noise image, arg3=filtered image', nargs=3, metavar=('Original', 'Noise', 'Filtered'))
parser.add_argument('--histogram', help='generates a histogram for a chosen chanel of a chosen image', type=str, metavar="Channel (R/G/B/greyscale)")
parser.add_argument('--huniform', help='Uniform final probability density function', nargs=2, type=int, metavar=('new minimum brightness', 'new maximum brightness'))
parser.add_argument('--cmean', help='mean, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--cvariance', help='variance, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--cstdev', help='standard deviation, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--cvarcoi', help='variation coefficient I, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--casyco', help='asymmetry coefficient, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--cflaco', help='flattening coefficient, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "brightness levels"))
parser.add_argument('--cvarcoii', help='variation coefficient II, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "[brightness levels]"))
parser.add_argument('--centropy', help='information source entropy, arg1=original image, arg2=improved image, arg3=brightness levels', nargs=3, metavar=('Original', 'Improved', "[brightness levels]"))
parser.add_argument('--sexdetii', help='Linear filtration in spatial domain (convolution) with extraction of details. Choices for convolution masks: S/SW/W/NW', metavar='[convolution mask]')
parser.add_argument('--sexdetii2', help='Improved linear filtration in spatial domain (convolution) with extraction of details.', action="store_true")
parser.add_argument('--osobel', help='Non-linear filtration in spatial domain. Sobel operator.', action="store_true")
parser.add_argument('--dilation', help='Dilation', type=int, metavar="[Structural element (1-10)]")
parser.add_argument('--erosion', help='Erosion', type=int, metavar="[Structural element (1-10)]")
parser.add_argument('--opening', help='Opening', type=int, metavar="[Structural element (1-10)]")
parser.add_argument('--closing', help='Closing', type=int, metavar="[Structural element (1-10)]")
parser.add_argument('--hmt', help='Hit and Miss Transform', type=int, metavar="[Structural element (1-12)]")
parser.add_argument('--m7', help='M7 operation', type=int, metavar="[Structural element (1-12)]")
parser.add_argument('--load', help='loads an image from a given path', metavar='Path')
parser.add_argument('--save', help='saves edited image in a specified folder under a specified name', metavar='Path')

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

if args.huniform and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.huniform:
    new_min = args.huniform[0]
    new_max = args.huniform[1]
    arr = histogram.huniform(arr, new_min, new_max)

if args.cmean:
    original = loadImg(args.cmean[0])
    improved = loadImg(args.cmean[1])
    brightness_lvl = int(args.cmean[2])
    result_original = image_characteristics.cmean(original, brightness_lvl)
    result_improved = image_characteristics.cmean(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.cvariance:
    original = loadImg(args.cvariance[0])
    improved = loadImg(args.cvariance[1])
    brightness_lvl = int(args.cvariance[2])
    result_original = image_characteristics.cvariance(original, brightness_lvl)
    result_improved = image_characteristics.cvariance(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.cstdev:
    original = loadImg(args.cstdev[0])
    improved = loadImg(args.cstdev[1])
    brightness_lvl = int(args.cstdev[2])
    result_original = image_characteristics.cstdev(original, brightness_lvl)
    result_improved = image_characteristics.cstdev(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.cvarcoi:
    original = loadImg(args.cvarcoi[0])
    improved = loadImg(args.cvarcoi[1])
    brightness_lvl = int(args.cvarcoi[2])
    result_original = image_characteristics.cvarcoi(original, brightness_lvl)
    result_improved = image_characteristics.cvarcoi(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.casyco:
    original = loadImg(args.casyco[0])
    improved = loadImg(args.casyco[1])
    brightness_lvl = int(args.casyco[2])
    result_original = image_characteristics.casyco(original, brightness_lvl)
    result_improved = image_characteristics.casyco(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.cflaco:
    original = loadImg(args.cflaco[0])
    improved = loadImg(args.cflaco[1])
    brightness_lvl = int(args.cflaco[2])
    result_original = image_characteristics.cflaco(original, brightness_lvl)
    result_improved = image_characteristics.cflaco(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.cvarcoii:
    original = loadImg(args.cvarcoii[0])
    improved = loadImg(args.cvarcoii[1])
    brightness_lvl = int(args.cvarcoii[2])
    result_original = image_characteristics.cvarcoii(original, brightness_lvl)
    result_improved = image_characteristics.cvarcoii(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.centropy:
    original = loadImg(args.centropy[0])
    improved = loadImg(args.centropy[1])
    brightness_lvl = int(args.centropy[2])
    result_original = image_characteristics.centropy(original, brightness_lvl)
    result_improved = image_characteristics.centropy(improved, brightness_lvl)
    print("original: " + str(result_original) + ", improved: " + str(result_improved))

if args.sexdetii and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.sexdetii:
    mask = args.sexdetii
    arr = image_filtration.sexdetii(arr, mask)

if args.sexdetii2 and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.sexdetii2:
    arr = image_filtration.sexdetii2(arr)

if args.osobel and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.osobel:
    arr = image_filtration.osobel(arr)

numberToStructural = {
    1:structural_elements.I,
    2:structural_elements.II,
    3:structural_elements.III,
    4:structural_elements.IV,
    5:structural_elements.V, 
    6:structural_elements.VI,
    7:structural_elements.VII,
    8:structural_elements.VIII,
    9:structural_elements.IX,
    10:structural_elements.X,
    11:structural_elements.XI,
    12:structural_elements.XII}

if args.dilation and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.dilation and args.dilation not in range(1,11):
    parser.error("incorrect number of structural element")
elif args.dilation:
    arr = morphological.dilation(numberToStructural[args.dilation], arr)

if args.erosion and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.erosion and args.erosion not in range(1,11):
    parser.error("incorrect number of structural element")
elif args.erosion:
    arr = morphological.erosion(numberToStructural[args.erosion], arr)

if args.opening and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.opening and args.opening not in range(1,11):
    parser.error("incorrect number of structural element")
elif args.opening:
    arr = morphological.opening(numberToStructural[args.opening], arr)

if args.closing and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.closing and args.closing not in range(1,11):
    parser.error("incorrect number of structural element")
elif args.closing:
    arr = morphological.closing(numberToStructural[args.closing], arr)

if args.hmt and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.hmt and args.hmt not in range(1,13):
    parser.error("incorrect number of structural element")
elif args.hmt:
    arr = morphological.hmt(numberToStructural[args.hmt], arr)

if args.m7 and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.m7 and args.m7 not in range(1,11):
    parser.error("incorrect number of structural element")
elif args.m7:
    arr = morphological.m7(numberToStructural[args.m7], arr)

if args.histogram and (args.load is None or args.save is None):
    parser.error("--load and --save arguments are required for this operation.")
elif args.histogram:
    histogramImg = histogram.histogram_f(arr, args.histogram)
    is_histogram = 1

if args.save:
    try:
        if is_histogram == 1:
            histogramImg[0].savefig(args.save)
            plt.show()
        elif is_histogram != 1:
            newImage = Image.fromarray(arr.astype(np.uint8))
            newImage.save(args.save)
            newImage.show()
    except (IOError, ValueError) as e:
        print(f"Error: Unable to save the image. Please check the file extension and try again.")
