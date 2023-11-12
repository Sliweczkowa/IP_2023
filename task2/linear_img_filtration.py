import numpy as np

# linear image filtration algorithm in spatial domain

# S4.1

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

    for p in range(border, P-border):
        for q in range(border, Q-border):
            neighborhood = arr[p - border: p + border + 1, q - border: q + border + 1]
            result[p,q] = np.sum(neighborhood * h)

    for i in range(border):
        result[i, :] = arr[i, :]
        result[P - border + i, :] = arr[P - border + i, :]
        result[:, i] = arr[:, i]
        result[:, Q - border + i] = arr[:, Q - border + i]

    return result

# S4.2

def sexdetii2(arr):
    P = len(arr)
    Q = len(arr[0])

    h = np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]])

    result = np.zeros_like(arr)

    for p in range(1, P-1):
        for q in range(1, Q-1):
            neighborhood = arr[p - 1: p + 2, q - 1: q + 2]
            result[p,q] = np.sum(neighborhood * h)

    for i in range(1):
        result[i, :] = arr[i, :]
        result[P - 1 + i, :] = arr[P - 1 + i, :]
        result[:, i] = arr[:, i]
        result[:, Q - 1 + i] = arr[:, Q - 1 + i]

    return result
