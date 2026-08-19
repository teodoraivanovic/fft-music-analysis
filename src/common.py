"""
This file contains some shared helper functions that will be used in the project, including:
Note/frequency conversion, synthetic tone/chord generation, .wav loading and plotting utilities.
"""

import numpy as np
from matplotlib import pyplot as plt

# Musical system constants 

# Reference pitch: A4 = 440 Hz (MIDI number 69).
A4_FREQ = 440.0
A4_MIDI = 69

# The 12 pitch classes in equal temperament, starting from C.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Note to frequency conversion 
# Frequency (Hz) of a note given in its MIDI number is calculated as: f = 440 * 2^((m - 69) / 12)
def note_to_freq(midi_number):
    return A4_FREQ * 2.0 ** ((midi_number - A4_MIDI) / 12.0)

# Frequency to note conversion
# Return (name, octave, midi) of the nearest note: m = round(69 + 12 * log2(f / 440))
def freq_to_note(f):
    if f <= 0:
        return None, None, None
    midi = int(round(A4_MIDI + 12 * np.log2(f / A4_FREQ)))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return name, octave, midi

# Return the pitch class index (0=C, ..., 11=B), folding across all octaves
def freq_to_chroma(f):
    _, _, midi = freq_to_note(f)
    if midi is None:
        return None
    return midi % 12

# Generate a synthetic tone at fundamental frequency f0, plus harmonics
def generate_tone(f0, duration, Fs, num_harmonics=1, decay=0.6, noise_std=0.0):
    t = np.arange(0, duration, 1.0 / Fs)
    signal = np.zeros_like(t)
    for h in range(1, num_harmonics + 1):
        amplitude = decay ** (h - 1)
        signal += amplitude * np.sin(2 * np.pi * (h * f0) * t)
    if noise_std > 0:
        signal += noise_std * np.random.randn(len(t))
    return signal

# Generate a chord as the sum of several tones (with harmonics)
def generate_chord(frequencies, duration, Fs, num_harmonics=4, noise_std=0.0):
    signal = None
    for f in frequencies:
        tone = generate_tone(f, duration, Fs, num_harmonics=num_harmonics)
        signal = tone if signal is None else signal + tone
    if noise_std > 0:
        signal += noise_std * np.random.randn(len(signal))
    return signal / len(frequencies)

# Loads a .wav file and return (Fs, signal), mono-averaged and normalized to [-1, 1]
def load_wav(path):
    from scipy.io import wavfile
    Fs, data = wavfile.read(path)

    if data.ndim == 2:
        data = data.mean(axis=1)

    data = data.astype(np.float64)
    if np.max(np.abs(data)) > 0:
        data = data / np.max(np.abs(data))

    return Fs, data

# Visualize signal waveform
def plot_signal(signal, Fs, title='Signal in time domain', xlim=None):
    t = np.arange(len(signal)) / Fs
    plt.figure(figsize=(12, 3))
    plt.plot(t, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()

# Visualize magnitude spectrum
def plot_spectrum(freq, coefficients, title='Magnitude spectrum', xlim=None):
    plt.figure(figsize=(12, 4))
    plt.plot(freq, np.abs(coefficients))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title(title)
    if xlim is not None:
        plt.xlim(xlim)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()