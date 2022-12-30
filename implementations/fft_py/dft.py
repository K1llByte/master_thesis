import cmath
import numpy as np
from math import sqrt

def unpack(m):
    N = len(m)
    f1 = []
    f2 = []
    j = complex(0,-1)
    for k, yk in enumerate(m):
        ynk = m[N-k-1].real - m[N-k-1].imag
        f1.append((yk + ynk)/2)
        f2.append(j*(yk - ynk)/2)
    return (f1,f2)



def dft(x):
    x = np.asarray(x, dtype=type(x[0]))
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N) / sqrt(N)
    return np.dot(M, x)

def idft(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N) / sqrt(N)
    return np.dot(M, x)

def __fft(p, inverse):
    n = len(p)
    # This is a pre condition (if n is not multiple of 2)
    if not (n and not (n & (n-1))):
        raise Exception("Input must length N as power of 2")
    elif n == 1:
        return p
    else:
        pe, po = p[::2], p[1::2]
        ye, yo = __fft(pe,inverse), __fft(po,inverse)
        y = [0] * n

        if inverse:
            w = cmath.exp(complex(0, cmath.pi * 2. / n))
        else:
            w = cmath.exp(complex(0, -cmath.pi * 2. / n))

        half_n = n//2
        for j in range(half_n):
            wj = pow(w, j)
            y[j] = ye[j] + wj*yo[j]
            y[j + half_n] = ye[j] - wj*yo[j]
        return y

def fft(p, inverse=False):
    return __fft(p, inverse)
    # if inverse:
    #     return [ c / len(p) for c in __fft(p, inverse)]
    # else:
    #     return __fft(p, inverse)


def fft_np(x, inverse=False):
    N = len(x)
    if N == 1:
        return x
    else:
        X_even = fft_np(x[::2],inverse)
        X_odd = fft_np(x[1::2],inverse)

        factor = np.exp(2j*np.pi*np.arange(N)/ N) if inverse else np.exp(-2j*np.pi*np.arange(N)/ N)

        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

if __name__ == '__main__':
    # x = np.random.random(4)
    x = np.array([0,1,2,3])
    # x = np.array([1, 0.707, 0, -0.707, -1, -0.707, 0, 0.707])

    print("========= ARRAY IMPLEMENTATION =========")
    freq1 = fft(x)
    print("fft: ", freq1)
    val1 = fft(freq1, inverse=True)
    print("ifft: ", val1)

    # print("NUMPY IMPLEMENTATION")
    # freq2 = fft_np(x)
    # val2 = fft_np(freq2, inverse=True)
    # print("fft: ", freq2)
    # print("ifft: ", val2)

    print("========= DFT =========")
    freq3 = dft(x)
    print("dft: ", freq3)
    val3 = idft(freq3)
    print("idft: ", val3)

    # # print("NUMPY REAL FFT")
    # freq4 = np.fft.fft(x)
    # val4 = np.fft.ifft(freq4)
    # # print("fft: ", freq4)
    # # print("ifft: ", val4)

    print(np.allclose(freq1, freq3))
    print(np.allclose(val1, val3))

    # N = 4
    # print(np.exp(-2j*np.pi*np.arange(N)/ N))

    # ws = np.array([1]*N)
    # w = cmath.exp(complex(0, -cmath.pi * 2. / N))
    # ws = pow(w, ws)
    # print(ws)