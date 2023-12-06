import numpy as np

# T2 | Normal and inverse fast Fourier transform with decimation in frequency domain.

def dft(arr: np.complex128):
    N = len(arr)
    M = len(arr[0])
    res_arr = np.zeros((N, M), dtype=np.complex128)

    for p in range(N):
        print("calculating"+str(p)+" of "+str(N))
        for q in range(M):
            res_arr[p, q] = 1/(np.sqrt(N*M)) * (np.sum(arr * (np.exp(-1j*2*np.pi*np.arange(N).reshape(-1, 1)*p / N) * 
                              np.exp(-1j*2*np.pi*np.arange(M)*q / M))))
    res_arrs = np.fft.fftshift(res_arr)
    lres_arrs = 10 * np.log(1+np.abs(res_arrs))
    return lres_arrs, res_arr

def idft(arr: np.complex128):
    N = len(arr)
    M = len(arr[0])
    res_arr = np.zeros((N, M), dtype=np.complex128)

    for p in range(N):
        print("calculating"+str(p)+" of "+str(N))
        for q in range(M):
            res_arr[p, q] = 1/(np.sqrt(N*M)) * (np.sum(arr * (np.exp(-1j*2*np.pi*np.arange(N).reshape(-1, 1)*p / N) * 
                              np.exp(1j*2*np.pi*np.arange(M)*q / M))))

    return res_arr

def fft(arr: np.complex128):
    return