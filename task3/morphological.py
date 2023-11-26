import numpy as np

from task3.structural_elements import StructuralElement
from task3 import structural_elements


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
    
def checkThreeStatesMatch(kernel: StructuralElement, arrayImage: np.ndarray) -> bool:
    for x in range(len(kernel.array)):
        for y in range(len(kernel.array[0])):
            if kernel.array[x, y]==1:
                if arrayImage[x, y]!=255:
                    return 0
            elif kernel.array[x, y]==0:
                if arrayImage[x,y]!=0:
                    return 0
    return 1

def dilation(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    arrayNewImage = np.zeros((len(arrayImage), len(arrayImage[0])))
    # What to do with missing border? Crop?

    for xArray in range(kernel.origin[0], len(arrayImage) - len(kernel.array) + kernel.origin[0]):
        for yArray in range(kernel.origin[1], len(arrayImage[0]) - len(kernel.array[0]) + kernel.origin[1]):
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

    for xArray in range(kernel.origin[0], len(arrayImage) - len(kernel.array) + kernel.origin[0]):
        for yArray in range(kernel.origin[1], len(arrayImage[0]) - len(kernel.array[0]) + kernel.origin[1]):
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

def hmt(kernel: StructuralElement, arrayImage: np.ndarray) -> np.ndarray:
    arrayNewImage = np.zeros((len(arrayImage), len(arrayImage[0])))

    if isinstance(kernel, tuple):
        newImages = (np.zeros((len(arrayImage), len(arrayImage[0]))))
        for i in range(len(kernel)):
            for xArray in range(kernel[i].origin[0], len(arrayImage) - len(kernel[i].array) + kernel[i].origin[0] + 1):
                for yArray in range(kernel[i].origin[1], len(arrayImage[0]) - len(kernel[i].array[0]) + kernel[i].origin[1] + 1):
                    arrayImagePart = arrayImage[xArray - kernel[i].origin[0]:xArray + len(kernel[i].array) - kernel[i].origin[0],
                                    yArray - kernel[i].origin[1]:yArray + len(kernel[i].array[0]) - kernel[i].origin[1]]
                    isMatch = checkThreeStatesMatch(kernel[i], arrayImagePart)               
                    if isMatch == 1:
                        arrayNewImage[xArray, yArray] = 255
                    elif isMatch == 0:
                        arrayNewImage[xArray, yArray] = 0
            newImages = np.maximum(newImages, arrayNewImage) 
        arrayNewImage = newImages
                        
    else:
        for xArray in range(kernel.origin[0], len(arrayImage) - len(kernel.array) + kernel.origin[0]):
                for yArray in range(kernel.origin[1], len(arrayImage[0]) - len(kernel.array[0]) + kernel.origin[1]):
                    arrayImagePart = arrayImage[xArray - kernel.origin[0]:xArray + len(kernel.array) - kernel.origin[0],
                                    yArray - kernel.origin[1]:yArray + len(kernel.array[0]) - kernel.origin[1]]
                    if np.all(arrayImagePart.astype(bool) == kernel.array):
                        arrayNewImage[xArray, yArray] = 255
                    else:
                        arrayNewImage[xArray, yArray] = 0 
    return arrayNewImage

    