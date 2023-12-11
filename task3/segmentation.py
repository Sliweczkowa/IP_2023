import numpy as np


# TODO: region growing for rgb, repeating regions deletion for rgb, conditionValue to CLI

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

    # for i in range(len(region)):
    #     j = 0
    #     while j < len(region):
    #         if i < j and np.array_equal(region[i], region[j]):
    #             region = np.delete(region, j, axis=0)
    #         j += 1

    return region


def regionGrowing(seedPointList: list[(int, int)], arrayImage: np.ndarray, conditionValue: int) -> np.ndarray:

    if arrayImage.ndim == 2:
        return regionGrowingForOneChannel(seedPointList, arrayImage, conditionValue)[0]

    elif arrayImage.ndim == 3:
        regionOfOneChannel0 = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 0], conditionValue)
        regionOfOneChannel1 = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 1], conditionValue)
        regionOfOneChannel2 = regionGrowingForOneChannel(seedPointList, arrayImage[:, :, 2], conditionValue)

        region = []

        for i in range(len(seedPointList)):
            region.append(np.zeros_like(arrayImage))

        for x in range(len(arrayImage)):
            for y in range(len(arrayImage[0])):
                for i in range(len(seedPointList)):
                    region[i][x, y, 0] = regionOfOneChannel0[i][x, y]
                    region[i][x, y, 1] = regionOfOneChannel1[i][x, y]
                    region[i][x, y, 2] = regionOfOneChannel2[i][x, y]

        return region[0].astype(np.uint8)
