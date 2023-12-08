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
            res_arr[p, q] = (np.sum(arr * (np.exp(-1j*2*np.pi*np.arange(N).reshape(-1, 1)*p / N) * 
                              np.exp(1j*2*np.pi*np.arange(M)*q / M))))

    return res_arr

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
    x_row_fft = np.array([fft(row) for row in x], dtype=np.complex128)

    x_col_fft = np.array([fft(col) for col in x_row_fft.T], dtype=np.complex128).T

    res_arrs = np.fft.fftshift(x_col_fft)
    lres_arrs = 10 * np.log(1+np.abs(res_arrs))
    return lres_arrs, x_col_fft

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
   x_row_ifft = np.array([ifft(row) for row in x], dtype=np.complex128)
   x_col_ifft = np.array([ifft(col) for col in x_row_ifft.T], dtype=np.complex128).T

   return x_col_ifft