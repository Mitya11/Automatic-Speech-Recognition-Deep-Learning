import torch
import itertools
import python_speech_features as pf
import numpy as np

alphabet = {'а': 1, 'б': 2, 'в': 3, 'г': 4, 'д': 5, 'е': 6, 'ж': 7, 'з': 8, 'и': 9, 'й': 10, 'к': 11, 'л': 12, 'м': 13,
            'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18, 'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25,
            'щ': 26, 'ъ': 27, 'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, " ": 33, 1: 'а', 2: 'б', 3: 'в', 4: 'г',
            5: 'д', 6: 'е', 7: 'ж', 8: 'з', 9: 'и', 10: 'й', 11: 'к', 12: 'л', 13: 'м', 14: 'н', 15: 'о', 16: 'п',
            17: 'р', 18: 'с', 19: 'т', 20: 'у', 21: 'ф', 22: 'х', 23: 'ц', 24: 'ч', 25: 'ш', 26: 'щ', 27: 'ъ', 28: 'ы',
            29: 'ь', 30: 'э', 31: 'ю', 32: 'я', 33: " ", 0: "-",34:"",35:"|"}

def decode_result(nn_output):
    text = "".join(list(map(lambda x: alphabet[x], nn_output)))
    result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
    return result
def get_features(samp,freq):
    split_size = 1

    n_fft = 512
    hop = 160
    mfcc = pf.mfcc(samp, freq, nfilt=40, nfft=256, winlen=0.015, winstep=0.01)
    delta_mfcc = pf.delta(mfcc, 2)
    a_mfcc = pf.delta(delta_mfcc, 2)
    features = torch.tensor(np.concatenate([mfcc, delta_mfcc, a_mfcc], axis=1))

    # spectrogram = torch.tensor(get_mel_spectrogram(samp, freq))
    """plt.imshow(spectrogram.transpose(0,1), origin = "lower")
    plt.show()"""
    sequence = torch.split(features, split_size)

    if sequence[-1].size()[0] != split_size:
        sequence = sequence[:-1]

    sequence = torch.stack(sequence)
    # standarize
    l = sequence.min()
    # sequence -=sequence.min()
    # sequence /= 20
    sequence = torch.squeeze(sequence)
    assert sequence.isnan().any().item() == 0
    return sequence
