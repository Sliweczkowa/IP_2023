import numpy as np


# image characteristics


# C1.1 | Mean
def cmean(arr, levels):
    width = arr.shape[1]
    height = arr.shape[0]

    if arr.ndim == 2:
        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
        to_sum = np.sum(np.arange(levels) * histogram)
        b = [1 / (width * height) * to_sum]
    
    elif arr.ndim == 3:
        b = []
        for channel in range(arr.shape[2]):
            channel_data = arr[:, :, channel]
            histogram, bins = np.histogram(channel_data.ravel(), bins=levels, range=(0, levels-1))
            to_sum = np.sum(np.arange(levels) * histogram)
            mean_channel = 1 / (width * height) * to_sum
            b.append(mean_channel)

    return b


# C1.2 | Variance
def cvariance(arr, levels):
    width = len(arr[0])
    height = len(arr)

    if arr.ndim == 2:
        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
        b = cmean(arr, levels)
        to_sum = np.sum(np.square(np.arange(levels) - b) * histogram)
        variance = 1 / (width * height) * to_sum
    
    elif arr.ndim == 3:
        variance = []
        for channel in range(arr.shape[2]):
            channel_data = arr[:, :, channel]
            histogram, bins = np.histogram(channel_data.ravel(), bins=levels, range=(0, levels-1))
            b_channel = cmean(channel_data, levels)
            to_sum_channel = np.sum(np.square(np.arange(levels) - b_channel) * histogram)
            variance_channel = 1 / (width * height) * to_sum_channel
            variance.append(variance_channel)

    return variance


# C2.1 | Standard deviation
def cstdev(arr, levels):
    d_sq = cvariance(arr, levels)
    d = np.sqrt(d_sq)
    return d


# C2.2 | Variation coefficient I
def cvarcoi(arr, levels):
    d = cstdev(arr, levels)
    b = cmean(arr, levels)
    b_z = d / b
    return b_z


# C3 | Asymmetry coefficient
def casyco(arr, levels):
    width = arr.shape[1]
    height = arr.shape[0]

    if arr.ndim == 2: 
        s = cstdev(arr, levels)
        b = cmean(arr, levels)

        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
        b_s = (1 / np.power(s, 3)) * (1 / (width * height)) * np.sum(np.power((np.arange(levels) - b), 3) * histogram)
    
    elif arr.ndim == 3: 
        b_s = np.zeros(arr.shape[2])
        for channel in range(arr.shape[2]):
            channel_data = arr[:, :, channel]
            histogram, bins = np.histogram(channel_data.ravel(), bins=levels, range=(0, levels-1))
            s_channel = cstdev(channel_data, levels)
            b_channel = cmean(channel_data, levels)
            b_s[channel] = (1 / np.power(s_channel, 3)) * ((1 / (width * height)) * np.sum(np.power((np.arange(levels) - b_channel), 3) * histogram))

    return b_s


# C4 | Flattening coefficient
def cflaco(arr, levels):
    width = len(arr[0])
    height = len(arr)

    if arr.ndim == 2:
        s = cstdev(arr, levels)
        b = cmean(arr, levels)
        
        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
        b_k = (1 / np.power(s, 4)) * (1 / (width * height)) * np.sum(np.power((np.arange(levels) - b), 4) * histogram - 3)
    
    elif arr.ndim == 3: 
        b_k = np.zeros(arr.shape[2])
        for channel in range(arr.shape[2]):
            channel_data = arr[:, :, channel]
            histogram, bins = np.histogram(channel_data.ravel(), bins=levels, range=(0, levels-1))
            s_channel = cstdev(channel_data, levels)
            b_channel = cmean(channel_data, levels)
            
            b_k[channel] = (1 / np.power(s_channel, 4)) * (1 / (width * height)) * np.sum(np.power((np.arange(levels) - b_channel), 4) * histogram - 3)
    
    return b_k


# C5 | Variation coefficient II
def cvarcoii(arr, levels):
    width = len(arr[0])
    height = len(arr)

    if arr.ndim == 2:
        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))

        b_n = np.square(1 / (width * height)) * np.sum(np.square(histogram))

    if arr.ndim == 3: 
        b_n = np.zeros(arr.shape[2])
        for c in range(3):
            channnel_data = arr[:, :, c]
            histogram, bins = np.histogram(channnel_data.ravel(), bins=levels, range=(0, levels-1))
            b_n[c] = np.square(1 / (width * height)) * np.sum(np.square(histogram))
    return b_n


# C6 | Information source entropy
def centropy(arr, levels):
    width = len(arr[0])
    height = len(arr)

    if arr.ndim == 2:
        histogram, bins = np.histogram(arr.ravel(), bins=levels, range=(0, levels-1))
        epsilon = 1e-10
        histogram = histogram + epsilon
        b_e = -1 / (width * height) * np.sum(histogram * np.log2(histogram / (width * height)))

    elif arr.ndim == 3:
        b_e = np.zeros(arr.shape[2])
        for c in range(3):
            channel_data = arr[:, :, c]
            epsilon = 1e-10
            histogram, bins = np.histogram(channel_data.ravel(), bins=levels, range=(0, levels-1))
            histogram = histogram + epsilon
            b_e[c] = -1 / (width * height) * np.sum(histogram * np.log2(histogram / (width * height)))

    return b_e
