"""
Numerical core of the project. It implements the Fourier
transform from scratch:
  - the naive discrete Fourier transform (DFT) in O(n^2),
  - the fast Fourier transform (FFT) via Cooley-Tukey in O(n log n),
  - the inverse FFT.
"""

import numpy as np

# Naive DFT from the definition, O(n^2)
# F_k = sum_n f_n * exp(-2*pi*i*k*n / N)
# Matrix form F = W f, where W[k,n] = exp(-2*pi*i*k*n / N)
def dft(x):

    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)   
    return W @ x

# Recursive radix-2 Cooley-Tukey. Requires len(x) to be a power of two.
# Splits the DFT into even- and odd-indexed halves, recombined with
# twiddle factors exp(-2*pi*i*k/N). O(n^2) -> O(n log n).
def _fft_recursive(x):

    N = x.shape[0]

    if N == 1:
        return x

    even = _fft_recursive(x[0::2])
    odd = _fft_recursive(x[1::2])

    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    first_half = even + factor[:N // 2] * odd
    second_half = even + factor[N // 2:] * odd

    return np.concatenate([first_half, second_half])

# Smallest power of two that is >= n
def _next_power_of_two(n):

    power = 1
    while power < n:
        power *= 2
    return power

# Fast Fourier transform (radix-2 Cooley-Tukey).
# Zero-pads to the next power of two if needed.
def fft(x, zero_pad=True):

    x = np.asarray(x, dtype=np.complex128)
    N = x.shape[0]

    if N & (N - 1) != 0:   
        if zero_pad:
            new_N = _next_power_of_two(N)
            x = np.concatenate([x, np.zeros(new_N - N, dtype=np.complex128)])
        else:
            raise ValueError(
                f'Length {N} is not a power of two. '
                f'Set zero_pad=True or pad the signal manually.')

    return _fft_recursive(x)

# Inverse FFT via the standard trick: ifft(X) = conj(fft(conj(X))) / N
def ifft(X):

    X = np.asarray(X, dtype=np.complex128)
    N = X.shape[0]

    return np.conj(fft(np.conj(X))) / N

# Frequencies matching the FFT coefficients.
def freq_axis(N, Fs):

    return np.fft.fftfreq(N, 1.0 / Fs)

# Max absolute error between our fft and numpy.fft.fft.
def compare_with_numpy(x):

    ours = fft(x)
    N = len(ours)
    numpy_result = np.fft.fft(x, N)

    return np.max(np.abs(ours - numpy_result))