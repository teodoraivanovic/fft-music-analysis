# Here we implement and compare two methods for detecting the fundamental
# frequency (F0) of a monophonic signal:
# 1. Autocorrelation - looks for periodicity of the signal in the time domain.
# 2. HPS (Harmonic Product Spectrum) - uses the fact that harmonics are
#    integer multiples of the fundamental frequency, so multiplying
#    "compressed" copies of the spectrum makes the fundamental frequency
#    stand out as the strongest peak.

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

# Parabolic interpolation around the peak for a more precise period estimate.
# Parabolic interpolation is a numerical optimization technique that fits a
# second-order polynomial (a parabola) through three points to estimate a
# function's maximum or minimum.
def _parabolic_interpolation(r, lag):

    if lag <= 0 or lag >= len(r) - 1:
        return float(lag)
    a, b, c = r[lag - 1], r[lag], r[lag + 1]
    denom = a - 2 * b + c
    if denom == 0:
        return float(lag)
    shift = 0.5 * (a - c) / denom
    return lag + shift

# Detect the fundamental frequency using the autocorrelation method.
# Looks for the largest autocorrelation peak within the lag range that
# corresponds to the allowed frequency range [fmin, fmax], then converts
# it to a frequency: f = Fs / lag
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
        lag = _parabolic_interpolation(r, lag)

    if lag == 0:
        return None
    
    return Fs / lag

# Compute the Harmonic Product Spectrum of the signal:
# 1. Compute the magnitude spectrum |FFT(x)|.
# 2. For each harmonic h = 2, 3, ..., n, build a "compressed" version of
#    the spectrum (taking every h-th element) and multiply it into the
#    original.
# 3. Wherever the harmonics of the fundamental frequency line up, the
#    product produces a pronounced peak at the fundamental frequency.
# Returns (freqs, hps), where hps is defined over positive frequencies.
def harmonic_product_spectrum(signal, Fs, num_harmonics=5):

    N = len(signal)
    spectrum = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(N, 1.0 / Fs)

    hps = spectrum.copy()
    for h in range(2, num_harmonics + 1):
        compressed = spectrum[::h]              # every h-th element
        hps[:len(compressed)] *= compressed     # multiply by the "compressed" spectrum

    return freqs, hps

# Detect the fundamental frequency using the HPS method
def detect_pitch_hps(signal, Fs, num_harmonics=5, fmin=50, fmax=2000):

    freqs, hps = harmonic_product_spectrum(signal, Fs, num_harmonics)

    # Restrict the search to the allowed frequency range
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None

    indices = np.where(mask)[0]
    strongest = indices[np.argmax(hps[indices])]
    return freqs[strongest]

# Helper function for evaluation
def errors(measured, true):

    measured = np.asarray(measured, dtype=float)
    true = np.asarray(true, dtype=float)

    mae = np.mean(np.abs(measured - true))
    rmse = np.sqrt(np.mean((measured - true) ** 2))

    return mae, rmse