import torch
import itertools
import python_speech_features as pf
import numpy as np
import random
from ru_transcript import RuTranscript

alphabet = {'а': 1, 'б': 2, 'в': 3, 'г': 4, 'д': 5, 'е': 6, 'ё': 6, 'ж': 7, 'з': 8, 'и': 9, 'й': 10, 'к': 11, 'л': 12,
            'м': 13,
            'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18, 'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25,
            'щ': 26, 'ъ': 27, 'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, " ": 33, 1: 'а', 2: 'б', 3: 'в', 4: 'г',
            5: 'д', 6: 'е', 7: 'ж', 8: 'з', 9: 'и', 10: 'й', 11: 'к', 12: 'л', 13: 'м', 14: 'н', 15: 'о', 16: 'п',
            17: 'р', 18: 'с', 19: 'т', 20: 'у', 21: 'ф', 22: 'х', 23: 'ц', 24: 'ч', 25: 'ш', 26: 'щ', 27: 'ъ', 28: 'ы',
            29: 'ь', 30: 'э', 31: 'ю', 32: 'я', 33: " ", 0: "-", 34: "", 35: "|"}

transcript = {0: 'a', 1: 'ɑ', 2: 'æ', 3: 'æ.', 4: 'ɐ.', 5: 'ɐ', 6: 'ə', 7: 'ʌ', 8: 'b', 9: 'bʷ', 10: 'bː', 11: 'bːʷ',
              12: 'bˠ', 13: 'bʲ', 14: 'bᶣ', 15: 'v', 16: 'vʷ', 17: 'vˠ', 18: 'vʲ', 19: 'vᶣ', 20: 'ɡ', 21: 'ɡʷ',
              22: 'ɡˠ', 23: 'ɡʲ', 24: 'ɡᶣ', 25: 'ɡː', 26: 'γ', 27: 'γʷ', 28: 'd', 29: 'dʷ', 30: 'dˠ', 31: 'dʲ',
              32: 'dᶣ', 33: 'dː', 34: 'dːʷ', 35: 'dːˠ', 36: 'dʲː', 37: 'dːᶣ', 38: 'ʐ', 39: 'ʐʷ', 40: 'ʐˠ', 41: 'ʑː',
              42: 'ʑːʷ', 43: 'ʑːˠ', 44: 'ʑʲː', 45: 'ʑːᶣ', 46: 'd͡ʒ', 47: 'd͡ʒᶣ', 48: 'z', 49: 'zʷ', 50: 'zˠ', 51: 'zʲ',
              52: 'zᶣ', 53: 'zː', 54: 'zʲː', 55: 'i', 56: 'ɪ', 57: 'ɪ.', 58: 'j', 59: 'ʝ', 60: 'jʷ', 61: 'jᶣ', 62: 'ʝʷ',
              63: 'ʝᶣ', 64: 'k', 65: 'kʷ', 66: 'kˠ', 67: 'kʲ', 68: 'kː', 69: 'kːʷ', 70: 'kʲː', 71: 'kᶣ', 72: 'l',
              73: 'lʷ', 74: 'lˠ', 75: 'lʲ', 76: 'lᶣ', 77: 'lː', 78: 'lːʷ', 79: 'lʲː', 80: 'lːᶣ', 81: 'm', 82: 'mʷ',
              83: 'mˠ', 84: 'mʲ', 85: 'mː', 86: 'mːʷ', 87: 'mʲː', 88: 'mːˠ', 89: 'mᶣ', 90: 'ɱ', 91: 'ɱʲ', 92: 'n',
              93: 'nʷ', 94: 'nˠ', 95: 'nʲ', 96: 'nː', 97: 'nːʷ', 98: 'nːˠ', 99: 'nʲː', 100: 'nᶣ', 101: 'o', 102: 'ɵ',
              103: 'p', 104: 'pʷ', 105: 'pː', 106: 'pːʷ', 107: 'pʲː', 108: 'pˠ', 109: 'pʲ', 110: 'pᶣ', 111: 'r',
              112: 'rʷ', 113: 'rˠ', 114: 'rʲ', 115: 'rː', 116: 'rːʷ', 117: 'rʲː', 118: 'rᶣ', 119: 'r̥', 120: 'r̥ʲ',
              121: 's', 122: 'sʷ', 123: 'sˠ', 124: 'sʲ', 125: 'sː', 126: 'sːʷ', 127: 'sʲː', 128: 'sᶣ', 129: 't',
              130: 'tʷ', 131: 'tˠ', 132: 'tʲ', 133: 'tː', 134: 'tʲː', 135: 'tᶣ', 136: 'u', 137: 'ʉ', 138: 'ʊ', 139: 'ᵿ',
              140: 'f', 141: 'fʷ', 142: 'fˠ', 143: 'fʲ', 144: 'fʲː', 145: 'fᶣ', 146: 'x', 147: 'xʷ', 148: 'xˠ',
              149: 'xʲ', 150: 'xᶣ', 151: 't͡s', 152: 't͡sʷ', 153: 't͡sˠ', 154: 't͡sː', 155: 't͡sːʷ', 156: 't͡sːˠ',
              157: 'd̻͡z̪', 158: 't͡ɕ', 159: 't͡ɕᶣ', 160: 't͡ɕː', 161: 't͡ɕːᶣ', 162: 'ʂ', 163: 'ʂʷ', 164: 'ʂˠ',
              165: 'ʂː', 166: 'ʂːʷ', 167: 'ʂːˠ', 168: 'ɕː', 169: 'ɕːᶣ', 170: 'ɨ', 171: 'ɨ̟', 172: 'ɯ̟ɨ̟', 173: 'ᵻ',
              174: 'ɛ', 175: 'e', 176: 'ʔ','f:' : 177, 'a': 0, 'ɑ': 1, 'æ': 2, 'æ.': 3, 'ɐ.': 4, 'ɐ': 5, 'ə': 6, 'ʌ': 7, 'b': 8,
              'bʷ': 9, 'bː': 10, 'bːʷ': 11, 'bˠ': 12, 'bʲ': 13, 'bᶣ': 14, 'v': 15, 'vʷ': 16, 'vˠ': 17, 'vʲ': 18,
              'vᶣ': 19, 'ɡ': 20, 'ɡʷ': 21, 'ɡˠ': 22, 'ɡʲ': 23, 'ɡᶣ': 24, 'ɡː': 25, 'γ': 26, 'γʷ': 27, 'd': 28, 'dʷ': 29,
              'dˠ': 30, 'dʲ': 31, 'dᶣ': 32, 'dː': 33, 'dːʷ': 34, 'dːˠ': 35, 'dʲː': 36, 'dːᶣ': 37, 'ʐ': 38, 'ʐʷ': 39,
              'ʐˠ': 40, 'ʑː': 41, 'ʑːʷ': 42, 'ʑːˠ': 43, 'ʑʲː': 44, 'ʑːᶣ': 45, 'd͡ʒ': 46, 'd͡ʒᶣ': 47, 'z': 48, 'zʷ': 49,
              'zˠ': 50, 'zʲ': 51, 'zᶣ': 52, 'zː': 53, 'zʲː': 54, 'i': 55, 'ɪ': 56, 'ɪ.': 57, 'j': 58, 'ʝ': 59, 'jʷ': 60,
              'jᶣ': 61, 'ʝʷ': 62, 'ʝᶣ': 63, 'k': 64, 'kʷ': 65, 'kˠ': 66, 'kʲ': 67, 'kː': 68, 'kːʷ': 69, 'kʲː': 70,
              'kᶣ': 71, 'l': 72, 'lʷ': 73, 'lˠ': 74, 'lʲ': 75, 'lᶣ': 76, 'lː': 77, 'lːʷ': 78, 'lʲː': 79, 'lːᶣ': 80,
              'm': 81, 'mʷ': 82, 'mˠ': 83, 'mʲ': 84, 'mː': 85, 'mːʷ': 86, 'mʲː': 87, 'mːˠ': 88, 'mᶣ': 89, 'ɱ': 90,
              'ɱʲ': 91, 'n': 92, 'nʷ': 93, 'nˠ': 94, 'nʲ': 95, 'nː': 96, 'nːʷ': 97, 'nːˠ': 98, 'nʲː': 99, 'nᶣ': 100,
              'o': 101, 'ɵ': 102, 'p': 103, 'pʷ': 104, 'pː': 105, 'pːʷ': 106, 'pʲː': 107, 'pˠ': 108, 'pʲ': 109,
              'pᶣ': 110, 'r': 111, 'rʷ': 112, 'rˠ': 113, 'rʲ': 114, 'rː': 115, 'rːʷ': 116, 'rʲː': 117, 'rᶣ': 118,
              'r̥': 119, 'r̥ʲ': 120, 's': 121, 'sʷ': 122, 'sˠ': 123, 'sʲ': 124, 'sː': 125, 'sːʷ': 126, 'sʲː': 127,
              'sᶣ': 128, 't': 129, 'tʷ': 130, 'tˠ': 131, 'tʲ': 132, 'tː': 133, 'tʲː': 134, 'tᶣ': 135, 'u': 136,
              'ʉ': 137, 'ʊ': 138, 'ᵿ': 139, 'f': 140, 'fʷ': 141, 'fˠ': 142, 'fʲ': 143, 'fʲː': 144, 'fᶣ': 145, 'x': 146,
              'xʷ': 147, 'xˠ': 148, 'xʲ': 149, 'xᶣ': 150, 't͡s': 151, 't͡sʷ': 152, 't͡sˠ': 153, 't͡sː': 154,
              't͡sːʷ': 155, 't͡sːˠ': 156, 'd̻͡z̪': 157, 't͡ɕ': 158, 't͡ɕᶣ': 159, 't͡ɕː': 160, 't͡ɕːᶣ': 161, 'ʂ': 162,
              'ʂʷ': 163, 'ʂˠ': 164, 'ʂː': 165, 'ʂːʷ': 166, 'ʂːˠ': 167, 'ɕː': 168, 'ɕːᶣ': 169, 'ɨ': 170, 'ɨ̟': 171,
              'ɯ̟ɨ̟': 172, 'ᵻ': 173, 'ɛ': 174, 'e': 175, 'ʔ': 176,177:'f:', 178: "|",179:""}


def decode_result(nn_output):
    text = "".join(list(map(lambda x: transcript[x], nn_output)))
    result = "".join([c for c, k in itertools.groupby(text)]).replace("-", "")
    return result


def get_features(samp, freq):
    split_size = 1

    n_fft = 512
    hop = 160
    mfcc = pf.mfcc(samp, freq, nfilt=40, nfft=350, winlen=0.015, winstep=0.01)
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


def beam_search(model, prev_output, args, depth, width):
    prev = torch.nn.functional.softmax(prev_output[0], dim=-1)
    args[2] = (args[2][0][:, 0:1], args[2][1][:, 0:1])

    best_pred = torch.topk(prev, width)
    candidate = best_pred.indices
    prohability = best_pred.values

    for i in range(width):
        output = candidate[i]
        arg_c = args.copy()
        for j in range(depth):
            arg_c[1] = output.unsqueeze(dim=0)

            pr_output, hidden, _, _ = model.decoder(*arg_c)
            pr_output = torch.nn.functional.softmax(pr_output, dim=-1)[0]
            arg_c[2] = hidden
            output = torch.argmax(pr_output)
            prohability[i] *= pr_output[output]
    return candidate[torch.argmax(prohability)]


def shuffle_packets(lst, packet_size):
    assert len(lst) % packet_size == 0, "batches intersect!"
    result = []

    packets = [lst[i:i + packet_size] for i in range(0, len(lst), packet_size)]

    random.shuffle(packets)

    for packet in packets:
        result.extend(packet)

    return result


def get_transcript(text):
    ru_transcript = RuTranscript(text)
    ru_transcript.transcribe()

    tokens = [transcript[i] for i in ru_transcript.get_phonemes()] + [178]

    return tokens
