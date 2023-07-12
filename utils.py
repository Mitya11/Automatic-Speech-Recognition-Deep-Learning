import torch
import itertools
alphabet = {'а': 1, 'б': 2, 'в': 3, 'г': 4, 'д': 5, 'е': 6, 'ж': 7, 'з': 8, 'и': 9, 'й': 10, 'к': 11, 'л': 12, 'м': 13,
            'н': 14, 'о': 15, 'п': 16, 'р': 17, 'с': 18, 'т': 19, 'у': 20, 'ф': 21, 'х': 22, 'ц': 23, 'ч': 24, 'ш': 25,
            'щ': 26, 'ъ': 27, 'ы': 28, 'ь': 29, 'э': 30, 'ю': 31, 'я': 32, " ": 33, 1: 'а', 2: 'б', 3: 'в', 4: 'г',
            5: 'д', 6: 'е', 7: 'ж', 8: 'з', 9: 'и', 10: 'й', 11: 'к', 12: 'л', 13: 'м', 14: 'н', 15: 'о', 16: 'п',
            17: 'р', 18: 'с', 19: 'т', 20: 'у', 21: 'ф', 22: 'х', 23: 'ц', 24: 'ч', 25: 'ш', 26: 'щ', 27: 'ъ', 28: 'ы',
            29: 'ь', 30: 'э', 31: 'ю', 32: 'я', 33: " ", 34: "-"}

def decode_result(nn_output):
    nn_output = torch.squeeze(nn_output)
    best_symbols = torch.argmax(nn_output,dim=1)
    best_symbols = torch.where(best_symbols ==0,34,best_symbols)
    text = "".join(list(map(lambda x:alphabet[x],best_symbols.tolist())))
    result = "".join([c for c , k in itertools.groupby(text)]).replace("-","")
    print(result, len(text))
    return best_symbols

