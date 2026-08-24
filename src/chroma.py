import numpy as np

from common import NOTE_NAMES, A4_FREQ, A4_MIDI
from stft import stft
from matplotlib import pyplot as plt

# Compute the chromagram of the signal.
# For each STFT window, the energy of every frequency bin is assigned to
# the corresponding chroma (0=C ... 11=B) using the formula: chroma = round(12 * log2(f / 440)) + 9   (mod 12)
# (since A4=440 has MIDI 69, and 69 mod 12 = 9 = the index of note A).
# Returns:
#     times      : array of time instants
#     chromagram : matrix [12 x number_of_windows]
def chromagram(signal, Fs, window_size=4096, overlap=0.5,
               fmin=65.0, fmax=2000.0):

    freqs, times, S = stft(signal, Fs, window_size, overlap, window='hann')
    magnitude = np.abs(S)  # [frequency x time]

    # For each frequency bin, precompute which chroma it belongs to
    chroma_per_bin = np.full(len(freqs), -1, dtype=int)
    for i, f in enumerate(freqs):
        if fmin <= f <= fmax:
            midi = A4_MIDI + 12 * np.log2(f / A4_FREQ)
            chroma_per_bin[i] = int(round(midi)) % 12

    # Sum the energy per chroma
    num_windows = magnitude.shape[1]
    H = np.zeros((12, num_windows))
    for chroma in range(12):
        mask = chroma_per_bin == chroma
        H[chroma, :] = magnitude[mask, :].sum(axis=0)

    # Normalize each column (each time instant) to sum to 1
    sums = H.sum(axis=0, keepdims=True)
    sums[sums == 0] = 1
    H = H / sums

    return times, H

def plot_chromagram(times, H, title='Chromagram'):

    plt.figure(figsize=(12, 4))
    plt.imshow(H, aspect='auto', origin='lower', cmap='magma',
               extent=[times[0], times[-1], -0.5, 11.5])
    plt.yticks(range(12), NOTE_NAMES)
    plt.xlabel('Time [s]')
    plt.ylabel('Chroma')
    plt.title(title)
    plt.colorbar(label='Relative strength')
    plt.tight_layout()

# Return a dict {chord_name: 12-dim binary vector}.
# For each of the 12 roots we build a major and a minor triad:
# - major: root, major third (+4 semitones), fifth (+7)
# - minor: root, minor third (+3 semitones), fifth (+7)
def make_chord_templates():

    templates = {}
    for root in range(12):
        # Major triad
        major = np.zeros(12)
        for interval in [0, 4, 7]:
            major[(root + interval) % 12] = 1
        templates[NOTE_NAMES[root] + ':maj'] = major

        # Minor triad
        minor = np.zeros(12)
        for interval in [0, 3, 7]:
            minor[(root + interval) % 12] = 1
        templates[NOTE_NAMES[root] + ':min'] = minor

    return templates

# Return (chord_name, similarity) for the given 12-dim chroma vector.
# We use cosine similarity between the chroma vector and each template as the similarity metrics, and pick the chord with the highest similarity.
def detect_chord(chroma_vector, templates=None):

    if templates is None:
        templates = make_chord_templates()

    v = np.asarray(chroma_vector, dtype=float)
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return None, 0.0
    v = v / norm_v

    best_chord, best_similarity = None, -1.0
    for name, template in templates.items():
        t = template / np.linalg.norm(template)
        similarity = float(np.dot(v, t))
        if similarity > best_similarity:
            best_similarity = similarity
            best_chord = name

    return best_chord, best_similarity

# Detect a chord for every time instant of the chromagram
# Returns a list of (time, chord_name, similarity)
def detect_chord_sequence(times, H, templates=None):

    if templates is None:
        templates = make_chord_templates()

    result = []
    for j in range(H.shape[1]):
        name, similarity = detect_chord(H[:, j], templates)
        result.append((times[j], name, similarity))

    return result

# Evaluation
# Return the fraction of correctly detected chords (0..1)
# Arguments are lists of chord names of equal length
def detection_accuracy(detected, true):

    detected = list(detected)
    true = list(true)

    if len(detected) != len(true) or len(true) == 0:
        raise ValueError('The lists must be of equal, non-zero length.')
    
    hits = sum(1 for d, t in zip(detected, true) if d == t)

    return hits / len(true)