import numpy as np


def get_spectrogram(samples, freq, step=256):
    spectrogram = []

    for i in range(0, samples.size + 1, step):
        segment = samples[i: i + step]
        if segment.size != step:
            break
        segment = np.fft.fft(segment)
        segment = np.real(segment * np.conj(segment))
        segment = 10 * np.log10(np.abs(segment))

        spectrogram.append(segment[:step // 2])
    spectrogram = np.stack(spectrogram)
    return spectrogram
