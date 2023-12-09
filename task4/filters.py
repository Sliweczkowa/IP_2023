import numpy as np

from task4 import fourier_transform


# Function cropping image to square
def cropArrayImageToSquare(arrayImage: np.ndarray) -> np.ndarray:
    if len(arrayImage) != len(arrayImage[0]):
        if len(arrayImage) < len(arrayImage[0]):
            shorterImageSideLength = len(arrayImage)
        elif len(arrayImage) > len(arrayImage[0]):
            shorterImageSideLength = len(arrayImage[0])
        arrayImage = arrayImage[:shorterImageSideLength, :shorterImageSideLength]
    return arrayImage


# Function creating hamming window of given length
def createHammingWindow(imageSideLength: int, bandSize: int):
    hamming1D = np.hamming(imageSideLength)[:, None]
    hamming2D = np.sqrt(np.dot(hamming1D, hamming1D.T)) ** bandSize
    return hamming2D


# F1 | Low-pass filter (low-cut filter) for 2D array
def lpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    arrayImageWoj = fourier_transform.fft2d(arrayImage)[1]
    arrayImage = np.fft.fftshift(arrayImageWoj)
    arrayImage *= createHammingWindow(len(arrayImage), bandSize)
    arrayImage = np.fft.ifftshift(arrayImage)
    arrayImage = fourier_transform.ifft2d(arrayImage)
    arrayImage = np.abs(arrayImage)
    return arrayImage


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


# F2 | High-pass filter (low-cut filter) for 2D array
def hpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    return arrayImage - lpfForOneChannel(bandSize, arrayImage)


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
