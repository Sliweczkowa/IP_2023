import numpy as np


# linear image filtration algorithm in spatial domain


# S4.1 | Extraction of details (S, SW, W, NW filters)
def sexdetii(arr, mask):
    P = len(arr)
    Q = len(arr[0])

    if mask == "S":
        h = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])
    elif mask == "SW":
        h = np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]])
    elif mask == "W":
        h = np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]])
    elif mask == "NW":
        h = np.array([[1, 1, 1], [1, -2, -1], [1, -1, -1]])

    M = len(h)

    border = M // 2

    result = np.zeros_like(arr)

    if arr.ndim == 2:
        for p in range(border, P-border):
            for q in range(border, Q-border):
                neighborhood = arr[p - border: p + border + 1, q - border: q + border + 1]
                result[p,q] = abs(np.sum(neighborhood * h))
    
    elif arr.ndim == 3:
        for c in range(3):
            for p in range(border, P-border):
                for q in range(border, Q-border):
                    neighborhood = arr[p - border: p + border + 1, q - border: q + border + 1, c]
                    result[p, q, c] = abs(np.sum(neighborhood * h))

    return np.clip(result, 0, 255)


# S4.2 | Optimized extraction of details (S, SW, W, NW filters)
def sexdetii2(arr: np.array):
    def sexdetii_opt(arr):
        mask = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])

        maskH, maskW = mask.shape
        arrH, arrW = arr.shape
        h, w = arrH + 1 - maskH, arrW + 1 - maskW

        filter1 = np.arange(maskW) + np.arange(h)[:, np.newaxis] #1

        intermediate = arr[filter1]                              #2
        intermediate = np.transpose(intermediate, (0, 2, 1))     #3

        filter2 = np.arange(maskH) + np.arange(w)[:, np.newaxis] #4

        intermediate = intermediate[:, filter2]                  #5
        intermediate = np.transpose(intermediate, (0, 1, 3, 2))  #6

        product = intermediate * mask                            #7

        return abs(product.sum(axis=(2, 3)))                     #8

    if arr.ndim == 2:
        result = sexdetii_opt(arr)
    elif arr.ndim == 3: 
        result = np.zeros((510, 510, 3))
        result[:, :, 0] = sexdetii_opt(arr[:, :, 0])
        result[:, :, 1] = sexdetii_opt(arr[:, :, 1])
        result[:, :, 2] = sexdetii_opt(arr[:, :, 2])

    return np.clip(result, 0, 255)


# O3 | Sobel operator
def osobel(arr):
    P = len(arr)
    Q = len(arr[0])

    border = 1

    result = np.zeros_like(arr)

    if arr.ndim == 2:
        for p in range(border, P-border):
            for q in range(border, Q-border):
                neighborhood = arr[p - border: p + border + 1, q - border: q + border + 1]
                X = (neighborhood[0, 2] + 2 * neighborhood[1, 2] + neighborhood[2, 2]) - (neighborhood[0, 0] + 2 * neighborhood[1, 0] + neighborhood[2, 0])
                Y = (neighborhood[0, 0] + 2 * neighborhood[0, 1] + neighborhood[0, 2]) - (neighborhood[2, 0] + 2 * neighborhood[2, 1] + neighborhood[2, 2])
                result[p, q] = abs(np.sqrt(np.square(X)+np.square(Y)))
    
    elif arr.ndim == 3:
        for c in range(3):
            for p in range(border, P-border):
                for q in range(border, Q-border):
                    neighborhood = arr[p - border: p + border + 1, q - border: q + border + 1, c]
                    X = (neighborhood[0, 2] + 2 * neighborhood[1, 2] + neighborhood[2, 2]) - (neighborhood[0, 0] + 2 * neighborhood[1, 0] + neighborhood[2, 0])
                    Y = (neighborhood[0, 0] + 2 * neighborhood[0, 1] + neighborhood[0, 2]) - (neighborhood[2, 0] + 2 * neighborhood[2, 1] + neighborhood[2, 2])
                    result[p, q, c] = abs(np.sqrt(np.square(X)+np.square(Y)))

    return np.clip(result, 0, 255)
