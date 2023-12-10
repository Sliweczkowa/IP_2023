import numpy as np

# T2 | Normal and inverse fast Fourier transform with decimation in frequency domain.

def dft(arr: np.complex128):
    N = len(arr)
    M = len(arr[0])
    res_arr = np.zeros_like(arr, dtype=np.complex128)

    if arr.ndim == 2:
        for p in range(N):
            for q in range(M):
                res_arr[p, q] = (np.sum(arr * (np.exp(-1j*2*np.pi*np.arange(N).reshape(-1, 1)*p / N) * 
                                np.exp(-1j*2*np.pi*np.arange(M)*q / M))))
        res_arr_shift = np.fft.fftshift(res_arr)
        res_arr_adjusted = 10 * np.log(1+np.abs(res_arr_shift))
                
    elif arr.ndim == 3:
        for c in range(3):
            for p in range(N):
                for q in range(M):
                    res_arr[p, q, c] = (np.sum(arr[:,:,c] * (np.exp(-1j*2*np.pi*np.arange(N).reshape(-1, 1)*p / N) * 
                                    np.exp(-1j*2*np.pi*np.arange(M)*q / M))))
        res_arr_adjusted = np.zeros_like(res_arr, dtype=np.complex128)
        res_arr_adjusted[:,:,0] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,0])))
        res_arr_adjusted[:,:,1] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,1])))
        res_arr_adjusted[:,:,2] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,2])))
    
    return res_arr_adjusted, res_arr

def idft(arr: np.complex128):
    N = len(arr)
    M = len(arr[0])
    res_arr = np.zeros_like(arr, dtype=np.complex128)

    if arr.ndim == 2:
        for p in range(N):
            for q in range(M):
                res_arr[p, q] = (np.sum(arr * (np.exp(1j*2*np.pi*(np.arange(N)/N).reshape(-1, 1)*p)) * 
                                np.exp(1j*2*np.pi*(np.arange(M)/M)*q)))
    elif arr.ndim == 3:
        for c in range(3):
            for p in range(N):
                for q in range(M):
                    res_arr[p, q, c] = (np.sum(arr[:,:,c] * (np.exp(1j*2*np.pi*(np.arange(N)/N).reshape(-1, 1)*p)) * 
                                np.exp(1j*2*np.pi*(np.arange(M)/M)*q)))
    return res_arr / (M * N)

def fft(x: np.complex128):
  N = len(x)
  if N <= 1:
      return x
  even = fft(x[0::2])
  odd = fft(x[1::2])
  T = np.exp(-2j * np.pi * np.arange(N) / N)
  return np.concatenate([even + T[:N//2] * odd,
                       even + T[N//2:] * odd])

def fft2d(x: np.complex128):
    fft_rows = np.array([fft(row) for row in x], dtype=np.complex128)
    res_arr = np.array([fft(col) for col in fft_rows.T], dtype=np.complex128).T

    _ = ""
    return _, res_arr

def fourierVisualise(arr):
    res_arr = np.zeros_like(arr)

    if arr.ndim == 2:
        res_arr = fft2d(arr)[1]
        res_arr_shift = np.fft.fftshift(res_arr)
        res_arr_adjusted = 10 * np.log(1+np.abs(res_arr_shift))
    elif arr.ndim == 3:
        res_arr[:,:,0] = fft2d(arr[:,:,0])[1]
        res_arr[:,:,1] = fft2d(arr[:,:,1])[1]
        res_arr[:,:,2] = fft2d(arr[:,:,2])[1]
        res_arr_adjusted = np.zeros_like(arr)
        res_arr_adjusted[:,:,0] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,0])))
        res_arr_adjusted[:,:,1] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,1])))
        res_arr_adjusted[:,:,2] = 10 * np.log(1+np.abs(np.fft.fftshift(res_arr[:,:,2])))
    return res_arr_adjusted

def ifft(x, final=True):
  N = len(x)
  if N <= 1:
      return x
  even = ifft(x[0::2], final=False)
  odd = ifft(x[1::2], final=False)
  T = np.exp(2j * np.pi * np.arange(N) / N)
  result = np.concatenate([even + T[:N//2] * odd,
                          even + T[N//2:] * odd])
  return result / N if final else result

def ifft2d(x):
   fft_rows = np.array([ifft(row) for row in x], dtype=np.complex128)
   res_arr = np.array([ifft(col) for col in fft_rows.T], dtype=np.complex128).T

   return res_arr