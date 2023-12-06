from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
import torch
from utils import decode_result
from transforms import RandomOffset
from datetime import datetime
import librosa
from transforms import RandomOffset
import python_speech_features as pf
from SpeechRecognition import SpeechRecognition
from tqdm import tqdm
from glob import glob,iglob
import json
import re


def replace_file_extension(file_path):
    # Открываем файл для чтения
    with open(file_path, 'r') as file:
        content = file.read()

    # Заменяем все вхождения подстрок .txt на .opus
    modified_content = re.sub(r'\.txt', '.opus', content)

    # Открываем файл для записи и записываем измененный контент
    with open(file_path, 'w') as file:
        file.write(modified_content)

    print('Замена успешно выполнена!')


# Пример вызова функции для замены расширений в файле example.txt
replace_file_extension('F:/SpeechDataset/train/manifest.csv')