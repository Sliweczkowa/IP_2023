import numpy as np

from task4 import fourier_transform


# Function cropping image to square
def cropArrayImageToSquare(arrayImage: np.ndarray, sideLength: int = None) -> np.ndarray:
    if sideLength == None:
        if len(arrayImage) != len(arrayImage[0]):
            if len(arrayImage) < len(arrayImage[0]):
                sideLength = len(arrayImage)
            elif len(arrayImage) > len(arrayImage[0]):
                sideLength = len(arrayImage[0])
    arrayImage = arrayImage[:sideLength, :sideLength]
    return arrayImage


# Function creating hamming window of given length
def createHammingWindow(imageSideLength: int, bandSize: int):
    hamming1D = np.hamming(imageSideLength)[:, None]
    hamming2D = np.sqrt(np.dot(hamming1D, hamming1D.T)) ** bandSize
    return hamming2D


# Function creating mask of phase modifying filter
def createPhaseMask(imageHeight: int, imageWidth: int, l: int, k: int, j: int) -> np.ndarray:
    mask = np.zeros((imageHeight, imageWidth), dtype=float)

    for n in range(imageHeight):
        for m in range(imageWidth):
            firstParenthesis = (-1) * n * k * 2 * np.pi / imageHeight
            secondParenthesis = (-1) * m * l * 2 * np.pi / imageWidth
            mask[n, m] = np.exp(j * (firstParenthesis + secondParenthesis + ((k + l) * np.pi)))

    return mask


# F1 | Low-pass filter (high-cut filter) for 2D array
def lpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    arrayImage = fourier_transform.fft2d(arrayImage)[1]
    arrayImage = np.fft.fftshift(arrayImage)
    arrayImage *= createHammingWindow(len(arrayImage), bandSize)
    arrayImage = np.fft.ifftshift(arrayImage)
    arrayImage = fourier_transform.ifft2d(arrayImage)
    arrayImage = np.abs(arrayImage)
    return arrayImage


# F2 | High-pass filter (low-cut filter) for 2D array
def hpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    return arrayImage - lpfForOneChannel(bandSize, arrayImage)


# F3 | Band-pass filter for 2D array
def bpfForOneChannel(bandSizeLow: int, bandSizeHigh, arrayImage: np.ndarray) -> np.ndarray:
    return abs(arrayImage - bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage))


# F4 | Band-cut filter for 2D array
def bcfForOneChannel(bandSizeLow: int, bandSizeHigh, arrayImage: np.ndarray) -> np.ndarray:
    return lpfForOneChannel(bandSizeLow, arrayImage) + hpfForOneChannel(bandSizeHigh, arrayImage)


#  F1 | Low-pass filter (high-cut filter)
def lpf(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = lpfForOneChannel(bandSize, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = lpfForOneChannel(bandSize, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = lpfForOneChannel(bandSize, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = lpfForOneChannel(bandSize, arrayImage[:, :, 2])

    return arrayImage


# F2 | High-pass filter (low-cut filter)
def hpf(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = hpfForOneChannel(bandSize, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = hpfForOneChannel(bandSize, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = hpfForOneChannel(bandSize, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = hpfForOneChannel(bandSize, arrayImage[:, :, 2])

    return arrayImage


# F3 | Band-pass filter
def bpf(bandSizeLow: int, bandSizeHigh: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 2])

    return arrayImage


# F4 | Band-cut filter
def bcf(bandSizeLow: int, bandSizeHigh: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 2])

    return arrayImage


# F6 | Phase modifying filter
def pmf(input_array, k, l):
    array = fourier_transform.fft2d(input_array)[1]
    arr = np.fft.fftshift(array)
    N, M = arr.shape
    mask = np.zeros((N, M), dtype=np.complex128)

    for n in range(N):
        for m in range(M):
            phase = (-n*k*2*np.pi/N) + (-m*l*2*np.pi/M) + (k + l)*np.pi
            mask[n, m] = np.exp(1j * phase)

    filtered_spectrum = arr * mask 
    arrayImage = np.fft.ifftshift(filtered_spectrum)
    arrayImage = fourier_transform.ifft2d(arrayImage)
    arrayImage = np.abs(arrayImage)
    return arrayImage
