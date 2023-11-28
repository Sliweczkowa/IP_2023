import numpy as np


# R1 | Region growing (merging)
def regionGrowing(seedPoint: (int, int), arrayImage: np.ndarray) -> np.ndarray:
    region = np.zeros_like(arrayImage)

    visited = np.zeros_like(arrayImage)

    stack = [seedPoint]

    while stack:
        currentPoint = stack.pop()

        if not visited[currentPoint]:
            if arrayImage[currentPoint] == arrayImage[seedPoint]:
                region[currentPoint] = 255

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
