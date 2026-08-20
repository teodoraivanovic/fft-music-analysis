# TODO:
# 1. Autocorrelation
# 2. HPS

import numpy as np

# Autocorrelation measures the similarity of the signal with a time-shifted copy of itself:
# r[lag] = dot(x, shift(x, lag))

# Direct autocorrelation by definition, in O(n^2)
# Used only to show equivalence with the faster FFT version and to compare running times
def autocorrelation_direct(signal):

    N = len(signal)

    r = np.zeros(N)
    for lag in range(N):
        r[lag] = np.sum(signal[:N - lag] * signal[lag:])

    return r

# Autocorrelation can be computed in O(nlogn) instead of directly in O(n^2):
# r = IFFT(|FFT(x)|^2)
# a.k.a. the Wiener-Khinchin theorem
def autocorrelation(signal):

    N = len(signal)
    
    # Zero-pad to 2*N to avoid circular (cyclic) correlation
    spectrum = np.fft.fft(signal, 2 * N)
    power = spectrum * np.conj(spectrum)          # |FFT(x)|^2
    r = np.fft.ifft(power).real[:N]               # keep the first N lags

    return r

# Detect the fundamental frequency using the autocorrelation method
# Looks for the largest autocorrelation peak within the lag range that
# corresponds to the allowed frequency range [fmin, fmax], then converts
# it to a frequency:
# f = Fs / lag
def detect_pitch_autocorrelation(signal, Fs, fmin=50, fmax=2000, interpolation=True):

    r = autocorrelation(signal)

    # Lag range corresponding to the requested frequency range
    min_lag = int(Fs / fmax)
    max_lag = int(Fs / fmin)
    max_lag = min(max_lag, len(r) - 1)

    if min_lag >= max_lag:
        return None

    # Look for the largest peak (avoiding lag=0, which is always the maximum)
    segment = r[min_lag:max_lag]
    lag = min_lag + int(np.argmax(segment))

    if interpolation:
        # TODO - add parabolic interpolation
        None

    if lag == 0:
        return None
    
    return Fs / lag