import numpy as np
import scipy

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


def get_mel_spectrogram(samples, freq, filters_count=30, lower=80, upper=8000, step=256):
    # Createion of filters
    lower_mel = 1125 * np.log(1 + lower / 700)
    upper_mel = 1125 * np.log(1 + upper / 700)
    filter_point = 700 * (np.exp(np.linspace(lower_mel, upper_mel, filters_count+2) / 1125) - 1)
    freqs = np.fft.fftfreq(step, d=1 / freq)[:step // 2]

    points = np.floor(filter_point / freq * step)
    filters = np.zeros((filters_count, step // 2))
    idxs = np.array(range(step // 2))

    for i in range(1, filters_count+1):
        cur = np.zeros(step // 2)
        cur = np.where((points[i - 1] <= idxs) & (idxs <= points[i]),
                       (idxs - points[i - 1]) / (points[i] - points[i - 1]), cur)
        cur = np.where((points[i] <= idxs) & (idxs <= points[i + 1]),
                       (points[i + 1] - idxs) / (points[i + 1] - points[i]), cur)
        filters[i-1] = cur

    # Applying filters to spectrogram
    amplitudes = get_spectrogram(samples, freq, step)
    amplitudes = np.where(amplitudes < -100, 0, amplitudes)
    result = np.zeros((amplitudes.shape[0], filters_count))

    for i in range(amplitudes.shape[0]):
        result[i] = np.log(np.sum(amplitudes[i] ** 2 * filters, axis=1)+1)


    mel_spectrogram = 10 * np.log10(np.abs(scipy.fft.dct(result,type=2)))
    mel_spectrogram = np.where(mel_spectrogram < -100, 0.01, mel_spectrogram)

    return mel_spectrogram
