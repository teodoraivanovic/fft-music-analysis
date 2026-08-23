"""
Short-time Fourier transform and spectrogram.
A plain FFT shows WHICH frequencies are present, not WHEN. The STFT applies an
FFT to short overlapping windows, giving a spectrogram (time x frequency x magnitude).
"""

import numpy as np

# Window function of length M.
# Tapers the segment edges to zero to reduce spectral leakage.
def make_window(name, M):

    n = np.arange(M)
    if name == 'rectangular':
        return np.ones(M)
    elif name == 'hann':
        return 0.5 - 0.5 * np.cos(2 * np.pi * n / (M - 1))
    elif name == 'hamming':
        return 0.54 - 0.46 * np.cos(2 * np.pi * n / (M - 1))
    elif name == 'blackman':
        return (0.42 - 0.5 * np.cos(2 * np.pi * n / (M - 1))
                + 0.08 * np.cos(4 * np.pi * n / (M - 1)))
    else:
        raise ValueError(f'Unknown window: {name}')

# Short-time Fourier transform.
# Returns (freqs, times, S); S is the complex STFT matrix [freqs x times].
def stft(signal, Fs, window_size=2048, overlap=0.5, window='hann'):

    M = window_size
    hop = int(M * (1 - overlap))   
    w = make_window(window, M)

    starts = range(0, len(signal) - M + 1, hop)

    columns = []
    for p in starts:
        segment = signal[p:p + M] * w
        columns.append(np.fft.rfft(segment))   

    S = np.array(columns).T   

    freqs = np.fft.rfftfreq(M, 1.0 / Fs)
    times = (np.array(list(starts)) + M / 2) / Fs

    return freqs, times, S

# Magnitude spectrogram ready for plotting.
def spectrogram(signal, Fs, window_size=2048, overlap=0.5,
                window='hann', in_decibels=True):

    freqs, times, S = stft(signal, Fs, window_size, overlap, window)
    magnitudes = np.abs(S)
    if in_decibels:
        magnitudes = 20 * np.log10(magnitudes + 1e-10)   
    return freqs, times, magnitudes

# Plot the spectrogram of a signal.
def plot_spectrogram(signal, Fs, window_size=2048, overlap=0.5,
                     window='hann', ymax=2500, title=None):

    from matplotlib import pyplot as plt
    f, t, mag = spectrogram(signal, Fs, window_size, overlap, window)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f, mag, shading='auto', cmap='magma')
    plt.colorbar(label='Magnitude [dB]')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.ylim(0, ymax)
    if title is None:
        title = (f'Spectrogram (window={window}, '
                 f'size={window_size}, overlap={overlap})')
    plt.title(title)
    plt.tight_layout()
