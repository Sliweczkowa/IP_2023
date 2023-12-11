import numpy as np


# TODO: conditionValue to CLI

# R1 | Region growing (merging)
def regionGrowingForOneChannel(seedPointList: list[(int, int)], arrayImage: np.ndarray, conditionValue: int) -> list[np.ndarray]:
    region = []
    for i in range(len(seedPointList)):
        region.append(np.zeros_like(arrayImage))

    for i, seedPoint in enumerate(seedPointList):
        visited = np.zeros((len(arrayImage), len(arrayImage[0])), dtype=bool)
        stack = [seedPoint]

        while stack:
            currentPoint = stack.pop()

            if not visited[currentPoint]:
                if (arrayImage[currentPoint] >= arrayImage[seedPoint] - conditionValue and
                        arrayImage[currentPoint] <= arrayImage[seedPoint] + conditionValue):
                    region[i][currentPoint] = arrayImage[currentPoint]

                    if not currentPoint[1] + 1 == len(arrayImage[0]):
                        stack.append((currentPoint[0], currentPoint[1] + 1))
                    if not currentPoint[1] - 1 == 0:
                        stack.append((currentPoint[0], currentPoint[1] - 1))
                    if not currentPoint[0] + 1 == len(arrayImage[0]):
                        stack.append((currentPoint[0] + 1, currentPoint[1]))
                    if not currentPoint[0] - 1 == 0:
                        stack.append((currentPoint[0] - 1, currentPoint[1]))

                visited[currentPoint] = 1

    return region


def regionGrowing(seedPointList: list[(int, int)], arrayImage: np.ndarray, conditionValue: int) -> np.ndarray:

    if arrayImage.ndim == 2:
        region = regionGrowingForOneChannel(seedPointList, arrayImage, conditionValue)

    elif arrayImage.ndim == 3:
        mean = (arrayImage[:, :, 0] + arrayImage[:, :, 1] + arrayImage[:, :, 2]) // 3

        intencityRegionGrowing = regionGrowingForOneChannel(seedPointList, mean, conditionValue)

        region = []
        k = []

        for i in range(len(seedPointList)):
            region.append(np.zeros_like(arrayImage))
            k.append(np.zeros_like(arrayImage))

        for i in range(len(seedPointList)):
            k[i] = intencityRegionGrowing[i] / mean

        for i in range(len(seedPointList)):
            region[i][:, :, 2] = np.clip(arrayImage[:, :, 2] * k[i], 0, 255)
            region[i][:, :, 1] = np.clip(arrayImage[:, :, 1] * k[i], 0, 255)
            region[i][:, :, 0] = np.clip(arrayImage[:, :, 0] * k[i], 0, 255)

    for i in range(len(region)-1):
        j = i + 1
        while j < len(region):
            if np.array_equal(region[i], region[j]):
                region = np.delete(region, j, axis=0)
            j += 1

    return region[0].astype(np.uint8)
