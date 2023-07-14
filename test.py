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

freq, samp = wavfile.read("WavTrain/crowd/WIN_20230714_01_38_40_Pro.wav","r")#Салют вызов Светлане Васильевне
#samp = RandomOffset()(samp)
samp = np.array(samp, dtype=np.float64)

model = ASR()
model.cuda()
model.load_state_dict(torch.load("ASR"))
spectrogram = torch.tensor(librosa.feature.mfcc(y=samp,sr=16000,S=None,n_mfcc=28,n_fft=512,hop_length=256)).transpose(0,1)

sequence = torch.split(spectrogram, 2)

if sequence[-1].size()[0] != 2:
    sequence = sequence[:-1]
sequence = torch.stack(sequence).cuda()

sequence = sequence.reshape(-1, 28 * 2)

sequence =torch.unsqueeze(sequence,dim=1).cuda()

sequence = sequence.type(torch.float32)
#237100
result = model(sequence)
print("Тест:")
decode_result(torch.exp(result))
