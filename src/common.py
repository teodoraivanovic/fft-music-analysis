"""
common.py - shared helpers used by both parts of the project.

Note/frequency conversion, synthetic tone/chord generation, .wav loading
and plotting utilities.
"""

import numpy as np
from matplotlib import pyplot as plt

#Musical system constants 

# Reference pitch: A4 = 440 Hz (MIDI number 69).
A4_FREQ = 440.0
A4_MIDI = 69

# The 12 pitch classes in equal temperament, starting from C.
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F',
              'F#', 'G', 'G#', 'A', 'A#', 'B']


#Note <-> frequency conversion 

def note_to_freq(midi_number):
    """Frequency (Hz) of a note given its MIDI number: f = 440 * 2^((m - 69) / 12)."""
    return A4_FREQ * 2.0 ** ((midi_number - A4_MIDI) / 12.0)


def freq_to_note(f):
    """Return (name, octave, midi) of the nearest note: m = round(69 + 12 * log2(f / 440))."""
    if f <= 0:
        return None, None, None
    midi = int(round(A4_MIDI + 12 * np.log2(f / A4_FREQ)))
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return name, octave, midi


def freq_to_chroma(f):
    """Return the pitch class index (0=C, ..., 11=B), folding across all octaves."""
    _, _, midi = freq_to_note(f)
    if midi is None:
        return None
    return midi % 12


#Synthetic signal generation 

def generate_tone(f0, duration, Fs, num_harmonics=1, decay=0.6, noise_std=0.0):
    """Generate a synthetic tone at fundamental frequency f0, plus harmonics."""
    t = np.arange(0, duration, 1.0 / Fs)
    signal = np.zeros_like(t)
    for h in range(1, num_harmonics + 1):
        amplitude = decay ** (h - 1)
        signal += amplitude * np.sin(2 * np.pi * (h * f0) * t)
    if noise_std > 0:
        signal += noise_std * np.random.randn(len(t))
    return signal
