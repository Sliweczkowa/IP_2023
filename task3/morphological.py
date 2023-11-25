import numpy as np

from task3.structural_elements import StructuralElement


def checkMatch(kernel: StructuralElement, arrayImage: np.ndarray) -> str:
    listOfValues = []

    for x in range(len(kernel.array)):
        for y in range(len(kernel.array[0])):
            if kernel.array[x, y]:
                listOfValues.append(arrayImage[x, y])

    # perfect match - all true kernel cells correspond to ones
    if listOfValues.count(255) == len(listOfValues):
        return 'perfect match'

    # no match - all true kernel cells correspond to zeros
    elif listOfValues.count(0) == len(listOfValues):
        return 'no match'

    # some match - some true kernel cells correspond to ones and some to zeros
    else:
        return 'some match'


def dilation(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    arrayNewImage = np.zeros((len(arrayImage), len(arrayImage[0])))
    # What to do with missing border? Crop?

    for xArray in range(kernel.origin[0], len(arrayImage) - len(kernel.array) - 1 + kernel.origin[0] + 1):
        for yArray in range(kernel.origin[1], len(arrayImage[0]) - 1 - len(kernel.array[0]) - 1 + kernel.origin[1] + 1):
            arrayImagePart = arrayImage[xArray - kernel.origin[0]:xArray + len(kernel.array) - kernel.origin[0],
                             yArray - kernel.origin[1]:yArray + len(kernel.array[0]) - kernel.origin[1]]

            if checkMatch(kernel, arrayImagePart) == 'no match':
                arrayNewImage[xArray, yArray] = 0
            else:
                arrayNewImage[xArray, yArray] = 255
    return arrayNewImage


def erosion(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    arrayNewImage = np.zeros((len(arrayImage), len(arrayImage[0])))
    # What to do with missing border? Crop?

    for xArray in range(kernel.origin[0], len(arrayImage) - len(kernel.array) - 1 + kernel.origin[0] + 1):
        for yArray in range(kernel.origin[1], len(arrayImage[0]) - 1 - len(kernel.array[0]) - 1 + kernel.origin[1] + 1):
            arrayImagePart = arrayImage[xArray - kernel.origin[0]:xArray + len(kernel.array) - kernel.origin[0],
                             yArray - kernel.origin[1]:yArray + len(kernel.array[0]) - kernel.origin[1]]

            if checkMatch(kernel, arrayImagePart) == 'perfect match':
                arrayNewImage[xArray, yArray] = 255
            else:
                arrayNewImage[xArray, yArray] = 0
    return arrayNewImage


def opening(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    return dilation(kernel, erosion(kernel, arrayImage))


def closing(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    return erosion(kernel, dilation(kernel, arrayImage))
