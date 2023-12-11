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


# F1 | Low-pass filter (high-cut filter) for 2D array
def lpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    arrayImage = fourier_transform.fft2d(arrayImage)[1]
    arrayImage = np.fft.fftshift(arrayImage)
    arrayImage *= createHammingWindow(len(arrayImage), bandSize)
    arrayImage = np.fft.ifftshift(arrayImage)
    arrayImage = fourier_transform.ifft2d(arrayImage)
    arrayImage = np.abs(arrayImage)
    arrayImage -= arrayImage.min()
    arrayImage = arrayImage * 255 / arrayImage.max()
    return arrayImage.clip(0, 255)


# F2 | High-pass filter (low-cut filter) for 2D array
def hpfForOneChannel(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:
    return (arrayImage - lpfForOneChannel(bandSize, arrayImage)).clip(0, 255)


# F3 | Band-pass filter for 2D array
def bpfForOneChannel(bandSizeLow: int, bandSizeHigh, arrayImage: np.ndarray) -> np.ndarray:
    return abs(arrayImage - bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage)).clip(0, 255)


# F4 | Band-cut filter for 2D array
def bcfForOneChannel(bandSizeLow: int, bandSizeHigh, arrayImage: np.ndarray) -> np.ndarray:
    return (lpfForOneChannel(bandSizeLow, arrayImage) + hpfForOneChannel(bandSizeHigh, arrayImage)).clip(0, 255)


# F5 | High-pass filter with detection of edge direction for 2D array [image size - 256x256]
def hpfEdgeDetectionForOneChannel(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arrayImage = fourier_transform.fft2d(image)[1]
    arrayImageLPF = arrayImage * (mask / 255)
    arrayImage -= arrayImageLPF
    arrayImage = np.fft.fftshift(arrayImage)
    arrayImage = fourier_transform.ifft2d(arrayImage)
    arrayImage = np.abs(arrayImage)
    arrayImage -= arrayImage.min()
    arrayImage = arrayImage * 255 / arrayImage.max()
    return arrayImage.clip(0, 255)


#  F1 | Low-pass filter (high-cut filter)
def lpf(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = lpfForOneChannel(bandSize, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = lpfForOneChannel(bandSize, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = lpfForOneChannel(bandSize, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = lpfForOneChannel(bandSize, arrayImage[:, :, 2])

    return arrayImage.astype(np.uint8)


# F2 | High-pass filter (low-cut filter)
def hpf(bandSize: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = hpfForOneChannel(bandSize, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = hpfForOneChannel(bandSize, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = hpfForOneChannel(bandSize, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = hpfForOneChannel(bandSize, arrayImage[:, :, 2])

    return arrayImage.astype(np.uint8)


# F3 | Band-pass filter
def bpf(bandSizeLow: int, bandSizeHigh: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = bpfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 2])

    return arrayImage.astype(np.uint8)


# F4 | Band-cut filter
def bcf(bandSizeLow: int, bandSizeHigh: int, arrayImage: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage)

    if arrayImage.ndim == 2:
        arrayImage = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 0])
        arrayImage[:, :, 1] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 1])
        arrayImage[:, :, 2] = bcfForOneChannel(bandSizeLow, bandSizeHigh, arrayImage[:, :, 2])

    return arrayImage.astype(np.uint8)


# F5 | High-pass filter with detection of edge direction [image size - len(mask)xlen(mask)]
def hpf_edge(arrayImage: np.ndarray, mask: np.ndarray) -> np.ndarray:

    arrayImage = cropArrayImageToSquare(arrayImage, len(mask))

    if arrayImage.ndim == 2:
        arrayImage = hpfEdgeDetectionForOneChannel(arrayImage, mask)

    elif arrayImage.ndim == 3:
        arrayImage[:, :, 0] = hpfEdgeDetectionForOneChannel(arrayImage[:, :, 0], mask)
        arrayImage[:, :, 1] = hpfEdgeDetectionForOneChannel(arrayImage[:, :, 1], mask)
        arrayImage[:, :, 2] = hpfEdgeDetectionForOneChannel(arrayImage[:, :, 2], mask)

    return arrayImage.astype(np.uint8)


# F6 | Phase modifying filter
def pmf(input_array, k, l):

    def phaseShift(input_array, k, l):
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
    if input_array.ndim == 2:
       arrayImage = phaseShift(input_array, k, l)

    if input_array.ndim == 3:
        input_r = input_array[:,:,0]
        input_g = input_array[:,:,1]
        input_b = input_array[:,:,2]
        arrayImage = np.zeros_like(input_array)
        arrayImage[:,:, 0] = phaseShift(input_r, k, l)
        arrayImage[:,:,1] = phaseShift(input_g, k, l)
        arrayImage[:,:, 2] = phaseShift(input_b, k, l)
        
    return arrayImage
