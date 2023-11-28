import numpy as np


# R1 | Region growing (merging)
def regionGrowing(seedPointList: list[(int, int)], arrayImage: np.ndarray) -> list[np.ndarray]:
    region = [[], [], []]
    for i in range(len(seedPointList)):
        region[i] = np.zeros_like(arrayImage)

    for i, seedPoint in enumerate(seedPointList):
        visited = np.zeros_like(arrayImage)
        stack = [seedPoint]

        while stack:
            currentPoint = stack.pop()

            if not visited[currentPoint]:
                if arrayImage[currentPoint] == arrayImage[seedPoint]:
                    region[i][currentPoint] = 255

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
