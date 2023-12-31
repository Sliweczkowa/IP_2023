import numpy as np


# R1 | Region growing (merging)
def regionGrowingForOneChannel(seedPointList: list[(int, int)], arrayImage: np.ndarray, conditionValue: int) -> list[np.ndarray]:
    region = []
    for i in range(len(seedPointList)):
        region.append(np.full((len(arrayImage), len(arrayImage[0])), -1))

    for i, seedPoint in enumerate(seedPointList):
        visited = np.zeros((len(arrayImage), len(arrayImage[0])), dtype=bool)
        stack = [seedPoint]

        while stack:
            currentPoint = stack.pop()

            if not visited[currentPoint]:
                if (arrayImage[currentPoint] >= arrayImage[seedPoint] - conditionValue and
                        arrayImage[currentPoint] <= arrayImage[seedPoint] + conditionValue):
                    region[i][currentPoint] = arrayImage[currentPoint]

                    if currentPoint[1] + 1 < len(arrayImage[0]):
                        stack.append((currentPoint[0], currentPoint[1] + 1))
                    if currentPoint[1] - 1 >= 0:
                        stack.append((currentPoint[0], currentPoint[1] - 1))
                    if currentPoint[0] + 1 < len(arrayImage):
                        stack.append((currentPoint[0] + 1, currentPoint[1]))
                    if currentPoint[0] - 1 >= 0:
                        stack.append((currentPoint[0] - 1, currentPoint[1]))

                visited[currentPoint] = 1

    return region


def regionGrowing(seedPointList: list[(int, int)], arrayImage: np.ndarray, conditionValue: int) -> np.ndarray:

    if arrayImage.ndim == 2:
        region = regionGrowingForOneChannel(seedPointList, arrayImage, conditionValue)

    elif arrayImage.ndim == 3:
        mean = (arrayImage[:, :, 0] + arrayImage[:, :, 1] + arrayImage[:, :, 2]) // 3

        intencityRegionGrowing = regionGrowingForOneChannel(seedPointList, mean, conditionValue)
        red = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 0], conditionValue)
        green = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 1], conditionValue)
        blue = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 2], conditionValue)

        region = []
        k = []

        for i in range(len(seedPointList)):
            region.append(np.zeros_like(arrayImage))
            k.append(np.zeros_like(arrayImage))

            for x in range(len(arrayImage)):
                for y in range(len(arrayImage[0])):
                    if mean[x, y] != 0:
                        k[i][x, y] = intencityRegionGrowing[i][x, y] / mean[x, y]

            region[i][:, :, 2] = arrayImage[:, :, 2] * k[i][:, :, 2]
            region[i][:, :, 1] = arrayImage[:, :, 1] * k[i][:, :, 1]
            region[i][:, :, 0] = arrayImage[:, :, 0] * k[i][:, :, 0]

            for x in range(len(arrayImage)):
                for y in range(len(arrayImage[0])):
                    if red[i][x, y] == -1 or green[i][x, y] == -1 or blue[i][x, y] == -1:
                        region[i][x, y, :] = -1

            region[i] = np.clip(region[i], 0, 255)

    for i in range(len(region)-1):
        j = i + 1
        while j < len(region):
            if np.array_equal(region[i], region[j]):
                region = np.delete(region, j, axis=0)
            j += 1

    return region[0].astype(np.uint8)
