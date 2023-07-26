from scipy.io import wavfile
import matplotlib.pyplot as plt
from spectrogram import get_spectrogram,get_mel_spectrogram
import numpy as np
from WavDataset import WavDataSet
from model import ASR
import torch
from utils import decode_result
from transforms import RandomOffset
from datetime import datetime
import librosa
from transforms import RandomOffset
import python_speech_features as pf
from SpeechRecognition import SpeechRecognition
freq, samp = wavfile.read("recorded.wav","r")#Салют вызов Светлане Васильевне
#samp = RandomOffset()(samp)
samp = np.array(samp, dtype=np.float64)

model = SpeechRecognition()

result = model(samp,freq)
print("Тест:")
decode_result(torch.exp(result))
